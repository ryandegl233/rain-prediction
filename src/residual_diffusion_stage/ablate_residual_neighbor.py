"""Compare additional pointwise and 3x3 correction paths with identical initial forecasts."""

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


ARMS = ("point_correction", "neighbor_correction")
INITIAL_CHECKPOINT = "runs/rollout_dry_pro6000_0Q6klO/dry_guarded/checkpoints/checkpoint_1"


def export_comparison(output: Path, steps: int) -> None:
    reports, audits, scopes = {}, [], []
    for name in ARMS:
        reports[name] = {row["step"]: row for row in map(json.loads, (output / name / "fixed_validation_metrics.jsonl").read_text().splitlines())}
        if steps not in reports[name]:
            raise ValueError(f"Incomplete arm: {name}")
        audits.append(json.loads((output / name / "gradient_audit.jsonl").read_text().splitlines()[-1]))
        scopes.append(json.loads((output / name / "trainable_scope.json").read_text()))
    if scopes[0] != scopes[1] or audits[0]["first_two_updates_batches"] != audits[1]["first_two_updates_batches"]:
        raise ValueError("Trainable scope, original weights or training batches differ")
    if reports[ARMS[0]][0] != reports[ARMS[1]][0]:
        raise ValueError("Initial forecasts must have exactly equal validation reports")
    for name in ARMS:
        for path in (output / name / "fixed_validation/step_00000000").glob("rain_index_*.npz"):
            with np.load(path) as a, np.load(output / ARMS[0] / "fixed_validation/step_00000000" / path.name) as b:
                np.testing.assert_array_equal(a["Coarse"], b["Coarse"])
    with (output / ARMS[0] / "config/config_total.yaml").open() as handle:
        selection_cfg = yaml.safe_load(handle)["train"]["constraint_selection"]
    reference = reports[ARMS[0]][0]
    selection = {name: assess_candidate(reports[name][steps], reference, selection_cfg) for name in ARMS}
    selection["neighbor_vs_point_final"] = assess_candidate(reports[ARMS[1]][steps], reports[ARMS[0]][steps], selection_cfg)
    (output / "comparison.json").write_text(json.dumps({"reports": reports, "selection": selection}, indent=2))
    lines = ["# Independent correction ablation", "", "Original model frozen, including old cross/local modules.",
             "Only added correction trains; initial predictions and weights match. Seed, loss, samples and batches match.",
             "Point correction and 3x3 correction differ only in history search radius. No new positional encoding in this round.",
             "Compare to step zero (unchanged original model) and matched-step control. One seed, 60 rain-filtered development windows.", "",
             "| Step | Lead | Threshold | CSI point/neighbor | POD point/neighbor | FAR point/neighbor |",
             "|---|---|---|---|---|---|"]
    for step in sorted(reports[ARMS[0]]):
        for a, b in zip(reports[ARMS[0]][step]["deterministic"]["by_lead"], reports[ARMS[1]][step]["deterministic"]["by_lead"], strict=True):
            for threshold in ("0.01", "0.1", "0.2", "0.5"):
                x, y = a[f"threshold@{threshold}"], b[f"threshold@{threshold}"]
                cells = " | ".join(f"{x[key]:.5f}/{y[key]:.5f}" for key in ("csi", "pod", "far"))
                lines.append(f"| {step} | {a['lead_index']+1} | {threshold} | {cells} |")
    lines += ["", "## Final evolution", "", "| Lead | Measure | Point correction | Neighbor correction |", "|---|---|---|---|"]
    for lead, a in reports[ARMS[0]][steps]["evolution"].items():
        b = reports[ARMS[1]][steps]["evolution"][lead]
        for key in ("active_delta_mae", "change_amplitude_ratio", "decay_direction_recall", "core_0.5_mae", "core_0.5_intensity_ratio"):
            lines.append(f"| {lead} | {key} | {a[key]} | {b[key]} |")
    lines += ["", "## Eligibility", "", "```json", json.dumps(selection, indent=2), "```"]
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
        with np.load(output / ARMS[0] / "fixed_validation/step_00000000" / path.name) as initial:
            baseline_fields = {"GT": fields["GT"], "Original frozen": initial["Coarse"].copy(), "Neighbor correction": fields[ARMS[1]]}
        save_preview(output / f"baseline_vs_neighbor_{index}.png", baseline_fields, index)
    (output / "comparison.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.report_only:
        export_comparison(output, args.steps)
        return
    if args.steps < 2 or args.val_every <= 0 or args.steps % args.val_every or not 1 <= args.samples <= 60:
        raise ValueError("Require steps >=2 divisible by val-every; 1..60 validation samples")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Use a new empty output directory")
    checkpoint = Path(INITIAL_CHECKPOINT).resolve()
    with (checkpoint / "model.safetensors").open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    manifest = Path("runs/ccc_fixed_comparison_20261005/comparison.json").resolve()
    (output / "protocol.json").write_text(json.dumps({
        "checkpoint": str(checkpoint), "checkpoint_sha256": digest, "gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "manifest": str(manifest), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "steps": args.steps, "samples": args.samples, "seed": 2025, "batch_size": 1, "accumulation": 8,
        "trainable": "Only cross_modal_adapter.correction; original cross/local and all other original weights frozen",
        "initial_function_equal": True, "difference": "Added correction radius 0 versus 1", "position_encoding": False,
    }, indent=2))
    for name, radius in zip(ARMS, (0, 1), strict=True):
        command = [sys.executable, "-m", "src.trainer.rain_trainer_residual_neighbor",
                   f"rain_prediction_model.correction_radius={radius}", f"train.max_steps={args.steps}",
                   f"train.save_every={args.val_every}", f"val.val_duration={args.val_every}",
                   f"val.max_val_iters={args.samples}", "train.log.log_every=1", f"hydra.run.dir={output/name}"]
        (output / "status.json").write_text(json.dumps({"state": "running", "arm": name, "command": command}))
        print(f"Starting {name}: {output / (name + '.log')}", flush=True)
        with (output / f"{name}.log").open("w") as log:
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if any(marker in line for marker in ("[Train][", "[FixedVal]", "[CorrectionScope]", "[ConstraintSelection]", "Training finished", "Error", "Traceback")):
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
