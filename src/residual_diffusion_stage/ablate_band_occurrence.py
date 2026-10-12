"""Compare weak-rain-only occurrence suppression strengths with matched training."""

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


INITIAL_CHECKPOINT = "runs/rollout_dry_pro6000_0Q6klO/dry_guarded/checkpoints/checkpoint_1"
ARMS = {
    "lambda0p25": {"max_suppression": 0.25},
    "lambda0p50": {"max_suppression": 0.50},
    "lambda0p75": {"max_suppression": 0.75},
}
PROTECTED_THRESHOLDS = (0.1, 0.2, 0.5)
PROTECTED_METRICS = ("csi", "pod", "far", "f1", "hss")


def _assert_protected_metrics(candidate: dict, reference: dict, arm: str) -> None:
    candidate_det = candidate["deterministic"]
    reference_det = reference["deterministic"]
    groups = [("overall", candidate_det["overall"], reference_det["overall"])]
    groups.extend(
        (f"t+{index + 1}", row, reference_det["by_lead"][index])
        for index, row in enumerate(candidate_det["by_lead"])
    )
    for label, row, reference_row in groups:
        for threshold in PROTECTED_THRESHOLDS:
            key = f"threshold@{threshold}"
            for metric in PROTECTED_METRICS:
                if row[key][metric] != reference_row[key][metric]:
                    raise AssertionError(
                        f"{arm} changed protected {label} {key} {metric}: "
                        f"{row[key][metric]} vs {reference_row[key][metric]}"
                    )


def export_comparison(output: Path, steps: int) -> None:
    training = output / "training"
    reference_rows = {
        row["step"]: row
        for row in map(json.loads, (training / "fixed_validation_metrics.jsonl").read_text().splitlines())
    }
    if 0 not in reference_rows:
        raise ValueError("Training run has no step-zero frozen reference")
    reference = reference_rows[0]
    reports = {}
    for name in ARMS:
        metrics_path = training / name / "fixed_validation_metrics.jsonl"
        reports[name] = {row["step"]: row for row in map(json.loads, metrics_path.read_text().splitlines())}
        if steps not in reports[name]:
            raise ValueError(f"Incomplete arm: {name}")
    scope = json.loads((training / "trainable_scope.json").read_text())
    audit = json.loads((training / "gradient_audit.jsonl").read_text().splitlines()[-1])
    if scope["parameter_counts"]["occurrence"] <= 0 or not audit["frozen_unchanged"]:
        raise ValueError("Occurrence scope or frozen-parameter audit failed")
    for name in ARMS:
        _assert_protected_metrics(reports[name][steps], reference, name)

    with (training / "config/config_total.yaml").open() as handle:
        selection_cfg = yaml.safe_load(handle)["train"]["constraint_selection"]
    selection = {name: assess_candidate(reports[name][steps], reference, selection_cfg) for name in ARMS}
    eligible = [name for name, result in selection.items() if result["eligible"]]
    selected = min(eligible, key=lambda name: selection[name]["long_lead_far"]) if eligible else None
    result = {"reports": reports, "selection": selection, "selected": selected}
    (output / "comparison.json").write_text(json.dumps(result, indent=2))

    lines = [
        "# Band-limited occurrence screening",
        "",
        "Only predictions in [0.01, 0.1) mm/min are gated; stronger predictions are an exact bypass.",
        "All autoregressive history uses ungated rain. Only the occurrence head is trainable.",
        f"One shared occurrence head trains for {steps} optimizer steps at lr=1e-4.",
        "Lambda is swept after training with exactly the same learned weights and validation samples.",
        "Protected metrics at 0.1/0.2/0.5 are required to equal the frozen reference exactly.",
        "",
        "| Arm | MAE | mean CSI | CSI@0.01 | POD@0.01 | FAR@0.01 | Long FAR | Eligible |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name in ARMS:
        deterministic = reports[name][steps]["deterministic"]
        overall = deterministic["overall"]
        weak = overall["threshold@0.01"]
        long_far = sum(deterministic["by_lead"][index]["threshold@0.01"]["far"] for index in (2, 3)) / 2
        lines.append(
            f"| {name} | {overall['mae']:.6f} | {overall['mean_csi']:.5f} | "
            f"{weak['csi']:.5f} | {weak['pod']:.5f} | {weak['far']:.5f} | "
            f"{long_far:.5f} | {selection[name]['eligible']} |"
        )
    lines += ["", f"Selected: `{selected}`" if selected else "Selected: none; no candidate passed all safeguards."]
    lines += ["", "## Safeguards", "", "```json", json.dumps(selection, indent=2), "```"]

    final_dir = f"step_{steps:08d}"
    first = next(iter(ARMS))
    for path in sorted((training / first / "fixed_validation" / final_dir).glob("rain_index_*.npz")):
        with np.load(training / "fixed_validation/step_00000000" / path.name) as data:
            target = data["GT"].copy()
            baseline = data["Coarse"].copy()
        index = int(path.stem.removeprefix("rain_index_"))
        for name in ARMS:
            with np.load(training / name / "fixed_validation" / final_dir / path.name) as data:
                np.testing.assert_array_equal(target, data["GT"])
                candidate = data["Coarse"].copy()
            image_path = output / f"comparison_{name}_index_{index}.png"
            save_preview(image_path, {"GT": target, "Original frozen": baseline, name: candidate}, index)
            lines += ["", f"### {name}, sample {index}", "", f"![Sample {index}]({image_path.name})"]
    (output / "comparison.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--val-every", type=int, default=200)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.report_only:
        export_comparison(output, args.steps)
        selected = json.loads((output / "comparison.json").read_text())["selected"]
        (output / "status.json").write_text(json.dumps({"state": "complete", "selected": selected, "report_only": True}))
        return
    if args.steps < 2 or args.val_every <= 0 or args.steps % args.val_every or not 1 <= args.samples <= 60:
        raise ValueError("Require steps >=2 divisible by val-every; 1..60 validation samples")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Use a new empty output directory")

    checkpoint = Path(INITIAL_CHECKPOINT).resolve()
    with (checkpoint / "model.safetensors").open("rb") as handle:
        checkpoint_digest = hashlib.file_digest(handle, "sha256").hexdigest()
    manifest = Path("runs/ccc_fixed_comparison_20261005/comparison.json").resolve()
    (output / "protocol.json").write_text(json.dumps({
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_digest,
        "manifest": str(manifest),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "steps": args.steps,
        "samples": args.samples,
        "seed": 2025,
        "batch_size": 2,
        "accumulation": 4,
        "optimizer_lr": 1.0e-4,
        "trainable": "Only occurrence_head; all original predictor parameters frozen",
        "protected_output": ">=0.1 mm/min exact bypass; ungated autoregressive history",
        "training_suppression": 0.5,
        "post_training_sweep": ARMS,
    }, indent=2))

    command = [
        sys.executable,
        "-m",
        "src.trainer.rain_trainer_band_occurrence",
        "train.soft_occurrence.max_suppression=0.5",
        f"train.max_steps={args.steps}",
        f"train.save_every={args.val_every}",
        f"val.val_duration={args.val_every}",
        f"val.max_val_iters={args.samples}",
        "train.log.log_every=1",
        f"hydra.run.dir={output / 'training'}",
    ]
    (output / "status.json").write_text(json.dumps({"state": "running", "phase": "train_shared_head", "command": command}))
    print(f"Starting shared-head training: {output / 'training.log'}", flush=True)
    with (output / "training.log").open("w") as log:
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
            if process.stdout is None:
                raise RuntimeError("Trainer stdout pipe was not created")
            for line in process.stdout:
                log.write(line)
                log.flush()
                if any(marker in line for marker in (
                    "[Train][", "[FixedVal]", "[OccurrenceScope]", "[ConstraintSelection]",
                    "Training finished", "Error", "Traceback",
                )):
                    print(line, end="", flush=True)
            returncode = process.wait()
    if returncode:
        (output / "status.json").write_text(json.dumps({"state": "failed", "phase": "train_shared_head", "returncode": returncode}))
        raise RuntimeError(f"Shared-head training failed; inspect {output / 'training.log'}")
    export_comparison(output, args.steps)
    selected = json.loads((output / "comparison.json").read_text())["selected"]
    (output / "status.json").write_text(json.dumps({"state": "complete", "selected": selected}))
    print(f"Completed: {output / 'comparison.md'}", flush=True)


if __name__ == "__main__":
    main()
