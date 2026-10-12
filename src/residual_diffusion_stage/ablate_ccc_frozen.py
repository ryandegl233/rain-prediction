"""Train paired frozen-backbone CCC arms and compare matched rollout forecasts."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from src.residual_diffusion_stage.compare_coarse_smoke import save_preview


def export_comparison(output: Path, steps: int) -> None:
    arms = ("without_ccc", "with_ccc")
    reports = {}
    for name in arms:
        reports[name] = {}
        for line in (output / name / "fixed_validation_metrics.jsonl").read_text().splitlines():
            record = json.loads(line)
            reports[name][record["step"]] = record
        if steps not in reports[name]:
            raise ValueError(f"Incomplete arm: {name}, expected step {steps}")
    initial_a, initial_b = (reports[name][0] for name in arms)
    if initial_a["samples"] != initial_b["samples"] or initial_a["manifest_source"] != initial_b["manifest_source"]:
        raise ValueError("Validation protocols differ")
    # Both models are identical functions before the first optimizer update.
    for a, b in zip(initial_a["deterministic"]["by_lead"], initial_b["deterministic"]["by_lead"], strict=True):
        for key in ("mae", "rmse", "bias", "mean_csi"):
            np.testing.assert_allclose(a[key], b[key], rtol=1e-5, atol=1e-7)
        for threshold in ("0.01", "0.1", "0.2", "0.5"):
            for metric in ("csi", "pod", "far"):
                np.testing.assert_allclose(a[f"threshold@{threshold}"][metric], b[f"threshold@{threshold}"][metric], rtol=1e-5, atol=1e-7)
    scopes = [json.loads((output / name / "trainable_scope.json").read_text()) for name in arms]
    if scopes[0]["initial_frozen_sha256"] != scopes[1]["initial_frozen_sha256"]:
        raise ValueError("Frozen backbone initialization differs")
    audits = [json.loads((output / name / "gradient_audit.jsonl").read_text().splitlines()[-1]) for name in arms]
    if audits[0]["first_two_updates_batches"] != audits[1]["first_two_updates_batches"]:
        raise ValueError("Training batches differ in the first two optimizer updates")
    lines = [
        "# Frozen-backbone CCC ablation", "",
        "Same cross/local checkpoint, seed, losses and batches; only CCC differs. No future observations in validation.",
        "Single seed and 60 rain-filtered windows by default: a development experiment, not a generalization claim.",
        "Check FAR together with CSI/POD and strong-rain intensity; lower FAR alone is not success.", "",
        "| Step | Lead | Threshold mm/min | CSI no/CCC | POD no/CCC | FAR no/CCC |",
        "|---|---|---|---|---|---|",
    ]
    paired_steps = sorted(set(reports[arms[0]]) & set(reports[arms[1]]))
    for step in paired_steps:
        a, b = (reports[name][step] for name in arms)
        for x, y in zip(a["deterministic"]["by_lead"], b["deterministic"]["by_lead"], strict=True):
            for threshold in ("0.01", "0.1", "0.2", "0.5"):
                first, second = x[f"threshold@{threshold}"], y[f"threshold@{threshold}"]
                cells = [f"{first[key]:.5f}/{second[key]:.5f}" for key in ("csi", "pod", "far")]
                lines.append(f"| {step} | t+{x['lead_index'] + 1} | {threshold} | " + " | ".join(cells) + " |")
    final = {name: reports[name][steps] for name in arms}
    (output / "comparison.json").write_text(json.dumps({"paired_steps": paired_steps, "reports": reports}, indent=2))
    lines += ["", "## Final evolution and intensity", "", "| Lead | Measure | No CCC | CCC |", "|---|---|---|---|"]
    for lead, row in final[arms[0]]["evolution"].items():
        for key in ("active_delta_mae", "change_amplitude_ratio", "decay_direction_recall", "core_0.5_mae", "core_0.5_intensity_ratio"):
            lines.append(f"| t+{lead} | {key} | {row[key]:.5f} | {final[arms[1]]['evolution'][lead][key]:.5f} |")
    first_dir = output / arms[0] / "fixed_validation" / f"step_{steps:08d}"
    for path in sorted(first_dir.glob("rain_index_*.npz")):
        index = int(path.stem.removeprefix("rain_index_"))
        fields = {}
        for name in arms:
            other = output / name / "fixed_validation" / f"step_{steps:08d}" / path.name
            with np.load(other) as data:
                if "GT" in fields:
                    np.testing.assert_array_equal(fields["GT"], data["GT"])
                fields["GT"] = data["GT"].copy()
                fields[name] = data["Coarse"].copy()
        image_path = output / f"comparison_index_{index}.png"
        save_preview(image_path, fields, index)
        lines += ["", f"![Paired sample {index}]({image_path.name})"]
    (output / "comparison.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--val-every", type=int, default=250)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.report_only:
        export_comparison(output, args.steps)
        return
    if args.steps <= 0 or args.val_every <= 0 or args.steps % args.val_every or not 1 <= args.samples <= 60:
        raise ValueError("Use positive steps divisible by val-every, and 1..60 validation samples")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists() or any((output / name).exists() for name in ("without_ccc", "with_ccc")):
        raise FileExistsError("Choose a new output directory; existing experiments will not be overwritten")
    checkpoint = Path("runs/next_frame_cross_local_finetune/2026-08-29/22-03-48_cross_local_finetune/checkpoints/checkpoint_12/model.safetensors")
    manifest = Path("runs/ccc_fixed_comparison_20261005/comparison.json")
    with checkpoint.open("rb") as handle:
        checkpoint_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
    protocol = {
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": checkpoint_sha256,
        "manifest": str(manifest.resolve()), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "steps": args.steps, "validation_samples": args.samples, "seed": 2025,
        "batch_size": 1, "gradient_accumulation": 8, "scheduler_steps": 1000,
        "trainable": "cross attention and local branch in both arms; corrected CCC only in with_ccc",
        "gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "limitation": "Single seed, rain-filtered validation; assess held-out dry/mixed events before claiming robustness.",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2))
    for name, enabled in (("without_ccc", False), ("with_ccc", True)):
        command = [sys.executable, "-m", "src.trainer.rain_trainer_ccc_frozen_ablation",
                   f"rain_prediction_model.consensus_calibration_enabled={str(enabled).lower()}",
                   f"train.max_steps={args.steps}", "train.scheduler.num_training_steps=1000",
                   f"train.save_every={args.val_every}", f"val.val_duration={args.val_every}",
                   f"val.max_val_iters={args.samples}", f"hydra.run.dir={output / name}"]
        (output / "status.json").write_text(json.dumps({"state": "running", "arm": name, "command": command}))
        print(f"Starting {name}; log: {output / (name + '.log')}", flush=True)
        with (output / f"{name}.log").open("w") as log:
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
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
