"""Same-weight, same-loss comparison of pointwise and 3x3 history cross attention."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml

from src.residual_diffusion_stage.compare_coarse_smoke import save_preview
from src.trainer.rollout_dry_constraints import assess_candidate


DEFAULT_CHECKPOINT = "runs/rollout_dry_pro6000_0Q6klO/dry_guarded/checkpoints/checkpoint_1"
ARMS = ("pointwise", "neighborhood_3x3")


def export_comparison(output: Path, steps: int) -> None:
    reports = {}
    for name in ARMS:
        records = map(json.loads, (output / name / "fixed_validation_metrics.jsonl").read_text().splitlines())
        reports[name] = {row["step"]: row for row in records}
        if steps not in reports[name]:
            raise ValueError(f"Incomplete experiment arm: {name}")
    audits = [json.loads((output / name / "gradient_audit.jsonl").read_text().splitlines()[-1]) for name in ARMS]
    scopes = [json.loads((output / name / "trainable_scope.json").read_text()) for name in ARMS]
    if scopes[0] != scopes[1]:
        raise ValueError("Trainable scope or frozen initialization differs")
    if audits[0]["first_two_updates_batches"] != audits[1]["first_two_updates_batches"]:
        raise ValueError("Training batches differ")
    for name in ARMS:
        if not audits[ARMS.index(name)]["frozen_unchanged"]:
            raise ValueError("Frozen parameters changed")
    initial_a, initial_b = (reports[name][0] for name in ARMS)
    if initial_a["samples"] != initial_b["samples"] or initial_a["manifest_source"] != initial_b["manifest_source"]:
        raise ValueError("Validation protocols differ")
    with (output / ARMS[0] / "config/config_total.yaml").open() as handle:
        selection_cfg = yaml.safe_load(handle)["train"]["constraint_selection"]
    selection = {
        "vs_common_initial_pointwise": assess_candidate(reports[ARMS[1]][steps], initial_a, selection_cfg),
        "vs_matched_step_pointwise": assess_candidate(reports[ARMS[1]][steps], reports[ARMS[0]][steps], selection_cfg),
    }
    (output / "comparison.json").write_text(json.dumps({"reports": reports, "selection": selection}, indent=2))
    lines = ["# Pointwise / 3x3 cross-attention ablation", "",
             "Both arms use the same checkpoint, losses, seed, batch schedule and trainable scope.",
             "Only history search radius differs. Step-zero predictions may differ because the attention operator changes.",
             "Fresh optimizers; raw model validation with four free-rollout frames and no future observations.",
             "Single seed and rain-filtered development windows; this is not a generalization test.", "",
             "| Step | Lead | Threshold mm/min | CSI point/3x3 | POD point/3x3 | FAR point/3x3 |",
             "|---|---|---|---|---|---|"]
    for step in sorted(set(reports[ARMS[0]]) & set(reports[ARMS[1]])):
        for a, b in zip(reports[ARMS[0]][step]["deterministic"]["by_lead"], reports[ARMS[1]][step]["deterministic"]["by_lead"], strict=True):
            for threshold in ("0.01", "0.1", "0.2", "0.5"):
                x, y = a[f"threshold@{threshold}"], b[f"threshold@{threshold}"]
                values = " | ".join(f"{x[key]:.5f} / {y[key]:.5f}" for key in ("csi", "pod", "far"))
                lines.append(f"| {step} | {a['lead_index'] + 1} | {threshold} | {values} |")
    lines += ["", "## Final evolution", "", "| Lead | Measure | Pointwise | 3x3 |", "|---|---|---|---|"]
    for lead, a in reports[ARMS[0]][steps]["evolution"].items():
        b = reports[ARMS[1]][steps]["evolution"][lead]
        for key in ("active_delta_mae", "change_amplitude_ratio", "decay_direction_recall", "core_0.5_mae", "core_0.5_intensity_ratio"):
            lines.append(f"| {lead} | {key} | {a[key]} | {b[key]} |")
    lines += ["", "## Selection checks", "", "```json", json.dumps(selection, indent=2), "```"]
    for path in sorted((output / ARMS[0] / "fixed_validation" / f"step_{steps:08d}").glob("rain_index_*.npz")):
        fields = {}
        for name in ARMS:
            with np.load(output / name / "fixed_validation" / f"step_{steps:08d}" / path.name) as data:
                if "GT" in fields:
                    np.testing.assert_array_equal(fields["GT"], data["GT"])
                fields["GT"] = data["GT"].copy()
                fields[name] = data["Coarse"].copy()
        index = int(path.stem.removeprefix("rain_index_"))
        image_path = output / f"comparison_index_{index}.png"
        save_preview(image_path, fields, index)
        lines += ["", f"![Sample {index}]({image_path.name})"]
    (output / "comparison.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT))
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--val-every", type=int, default=125)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--query-chunk-size", type=int, default=256)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.report_only:
        export_comparison(output, args.steps)
        return
    if args.steps < 2 or args.val_every <= 0 or args.steps % args.val_every or not 1 <= args.samples <= 60 or args.query_chunk_size <= 0:
        raise ValueError("Require steps >=2 divisible by val-every, 1..60 samples and positive chunk size")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Choose an empty output directory")
    checkpoint_path = args.checkpoint.resolve()
    with (checkpoint_path / "model.safetensors").open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    manifest = Path("runs/ccc_fixed_comparison_20261005/comparison.json").resolve()
    (output / "protocol.json").write_text(json.dumps({
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": digest,
        "manifest": str(manifest), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "steps": args.steps, "samples": args.samples, "seed": 2025, "batch_size": 1, "accumulation": 8,
        "query_chunk_size": args.query_chunk_size, "gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "difference": "Only neighborhood radius 0 versus 1; same standard MHA weights, LayerNorm and local MLP",
        "loss": "Same guarded dry-rollout loss in both arms", "initial_function_equal": False,
    }, indent=2))
    for name, radius in zip(ARMS, (0, 1), strict=True):
        command = [sys.executable, "-m", "src.trainer.rain_trainer_rollout_dry", "--config-name=rain_trainer_neighborhood_cross",
                   f"train.init_model_path={checkpoint_path}", f"rain_prediction_model.neighborhood_radius={radius}",
                   f"rain_prediction_model.neighborhood_query_chunk_size={args.query_chunk_size}",
                   f"train.max_steps={args.steps}", f"train.save_every={args.val_every}", f"val.val_duration={args.val_every}",
                   f"val.max_val_iters={args.samples}", "train.log.log_every=1", f"hydra.run.dir={output / name}"]
        (output / "status.json").write_text(json.dumps({"state": "running", "arm": name, "command": command}))
        print(f"Starting {name}; log: {output / (name + '.log')}", flush=True)
        with (output / f"{name}.log").open("w") as log:
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if any(marker in line for marker in ("[Train][", "[FixedVal]", "[FrozenScope]", "[ConstraintSelection]", "[Checkpoint]", "Training finished", "Error", "Traceback")):
                        print(line, end="", flush=True)
                returncode = process.wait()
        if returncode:
            (output / "status.json").write_text(json.dumps({"state": "failed", "arm": name, "returncode": returncode}))
            raise RuntimeError(f"{name} failed; inspect {output / (name + '.log')}")
    export_comparison(output, args.steps)
    (output / "status.json").write_text(json.dumps({"state": "complete"}))
    print(f"Completed: {output / 'comparison.md'}", flush=True)


if __name__ == "__main__":
    main()
