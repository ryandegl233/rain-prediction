"""Run matched soft-occurrence hyperparameter screening from one frozen checkpoint."""

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
    "gamma1_fp2": {"gamma": 1.0, "false_positive_weight": 2.0},
    "gamma2_fp1": {"gamma": 2.0, "false_positive_weight": 1.0},
    "gamma2_fp2": {"gamma": 2.0, "false_positive_weight": 2.0},
    "gamma2_fp4": {"gamma": 2.0, "false_positive_weight": 4.0},
}


def export_comparison(output: Path, steps: int) -> None:
    reports = {}
    scopes = []
    audits = []
    for name in ARMS:
        metrics_path = output / name / "fixed_validation_metrics.jsonl"
        reports[name] = {row["step"]: row for row in map(json.loads, metrics_path.read_text().splitlines())}
        if steps not in reports[name]:
            raise ValueError(f"Incomplete arm: {name}")
        scopes.append(json.loads((output / name / "trainable_scope.json").read_text()))
        audits.append(json.loads((output / name / "gradient_audit.jsonl").read_text().splitlines()[-1]))
    first = next(iter(ARMS))
    reference = reports[first][0]
    if any(reports[name][0] != reference for name in ARMS):
        raise ValueError("All arms must have exactly equal step-zero forecasts")
    if any(scope != scopes[0] for scope in scopes[1:]):
        raise ValueError("Trainable scopes differ between occurrence arms")
    if any(audit["first_two_updates_batches"] != audits[0]["first_two_updates_batches"] for audit in audits[1:]):
        raise ValueError("Training batches differ between occurrence arms")

    with (output / first / "config/config_total.yaml").open() as handle:
        selection_cfg = yaml.safe_load(handle)["train"]["constraint_selection"]
    selection = {name: assess_candidate(reports[name][steps], reference, selection_cfg) for name in ARMS}
    eligible = [name for name, result in selection.items() if result["eligible"]]
    selected = min(eligible, key=lambda name: selection[name]["long_lead_far"]) if eligible else None
    result = {"reports": reports, "selection": selection, "selected": selected}
    (output / "comparison.json").write_text(json.dumps(result, indent=2))

    lines = [
        "# Soft occurrence screening",
        "",
        "The original predictor is frozen. Only the new occurrence head trains.",
        "Gate strength is zero through step 50, ramps to one by step 100, then uses p^gamma.",
        "All arms use the same seed, batches, baseline checkpoint and 60 fixed validation windows.",
        "",
        "| Arm | MAE | mean CSI | CSI@0.01 | POD@0.01 | FAR@0.01 | CSI@0.1 | POD@0.1 | FAR@0.1 | Eligible |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name in ARMS:
        overall = reports[name][steps]["deterministic"]["overall"]
        weak = overall["threshold@0.01"]
        moderate = overall["threshold@0.1"]
        lines.append(
            f"| {name} | {overall['mae']:.6f} | {overall['mean_csi']:.5f} | "
            f"{weak['csi']:.5f} | {weak['pod']:.5f} | {weak['far']:.5f} | "
            f"{moderate['csi']:.5f} | {moderate['pod']:.5f} | {moderate['far']:.5f} | "
            f"{selection[name]['eligible']} |"
        )
    lines += ["", f"Selected: `{selected}`" if selected else "Selected: none; no candidate passed all safeguards."]
    lines += ["", "## Long-lead weak-rain FAR", "", "| Arm | t+3 | t+4 | Mean |", "|---|---:|---:|---:|"]
    for name in ARMS:
        leads = reports[name][steps]["deterministic"]["by_lead"]
        values = [leads[index]["threshold@0.01"]["far"] for index in (2, 3)]
        lines.append(f"| {name} | {values[0]:.5f} | {values[1]:.5f} | {sum(values) / 2:.5f} |")
    lines += ["", "## Safeguards", "", "```json", json.dumps(selection, indent=2), "```"]

    final_dir = f"step_{steps:08d}"
    for path in sorted((output / first / "fixed_validation" / final_dir).glob("rain_index_*.npz")):
        with np.load(output / first / "fixed_validation/step_00000000" / path.name) as data:
            target = data["GT"].copy()
            baseline = data["Coarse"].copy()
        index = int(path.stem.removeprefix("rain_index_"))
        for name in ARMS:
            with np.load(output / name / "fixed_validation" / final_dir / path.name) as data:
                np.testing.assert_array_equal(target, data["GT"])
                candidate = data["Coarse"].copy()
            image_path = output / f"comparison_{name}_index_{index}.png"
            save_preview(
                image_path,
                {"GT": target, "Original frozen": baseline, name: candidate},
                index,
            )
            lines += ["", f"### {name}, sample {index}", "", f"![Sample {index}]({image_path.name})"]
    (output / "comparison.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--val-every", type=int, default=50)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.report_only:
        export_comparison(output, args.steps)
        selected = json.loads((output / "comparison.json").read_text())["selected"]
        (output / "status.json").write_text(json.dumps({
            "state": "complete",
            "selected": selected,
            "report_only": True,
        }))
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
        "batch_size": 1,
        "accumulation": 8,
        "trainable": "Only occurrence_head; all original predictor parameters frozen",
        "arms": ARMS,
    }, indent=2))

    for name, settings in ARMS.items():
        command = [
            sys.executable,
            "-m",
            "src.trainer.rain_trainer_soft_occurrence",
            f"train.soft_occurrence.gate_gamma={settings['gamma']}",
            f"train.soft_occurrence.false_positive_weight={settings['false_positive_weight']}",
            f"train.max_steps={args.steps}",
            f"train.save_every={args.val_every}",
            f"val.val_duration={args.val_every}",
            f"val.max_val_iters={args.samples}",
            "train.log.log_every=1",
            f"hydra.run.dir={output / name}",
        ]
        (output / "status.json").write_text(json.dumps({"state": "running", "arm": name, "command": command}))
        print(f"Starting {name}: {output / (name + '.log')}", flush=True)
        with (output / f"{name}.log").open("w") as log:
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
            (output / "status.json").write_text(json.dumps({
                "state": "failed", "arm": name, "returncode": returncode,
            }))
            raise RuntimeError(f"{name} failed; inspect {output / (name + '.log')}")
    export_comparison(output, args.steps)
    (output / "status.json").write_text(json.dumps({"state": "complete", "selected": json.loads((output / 'comparison.json').read_text())["selected"]}))
    print(f"Completed: {output / 'comparison.md'}", flush=True)


if __name__ == "__main__":
    main()
