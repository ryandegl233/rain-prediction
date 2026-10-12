"""Paired continuation from one frozen-backbone checkpoint, varying only dry constraints."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--val-every", type=int, default=125)
    parser.add_argument("--samples", type=int, default=60)
    args = parser.parse_args()
    if args.steps < 2 or args.val_every <= 0 or args.steps % args.val_every or not 1 <= args.samples <= 60:
        raise ValueError("Require steps >= 2, divisible by val-every; 1..60 validation samples")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Use a new empty output directory")
    checkpoint = Path("runs/ccc_frozen_ablation_4090_b112aM/without_ccc/checkpoints/checkpoint_0/model.safetensors").resolve()
    with checkpoint.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    (output / "protocol.json").write_text(json.dumps({
        "checkpoint": str(checkpoint), "sha256": digest, "steps": args.steps, "samples": args.samples,
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"), "seed": 2025, "batch": 1, "accumulation": 8,
        "scope": "Cross/local only; backbone frozen, CCC disabled; fresh identical optimizers in both arms",
        "limitation": "Development validation uses 60 rain-filtered windows; independent dry/mixed events still required",
    }, indent=2))
    reports = {}
    audits = []
    scopes = []
    for name, enabled in (("control", False), ("dry_guarded", True)):
        command = [sys.executable, "-m", "src.trainer.rain_trainer_rollout_dry",
                   f"train.init_model_path={checkpoint.parent}",
                   f"train.rollout_dry_constraint.enabled={str(enabled).lower()}",
                   f"train.max_steps={args.steps}", f"train.save_every={args.val_every}",
                   f"val.val_duration={args.val_every}", f"val.max_val_iters={args.samples}",
                   "train.log.log_every=1", f"hydra.run.dir={output / name}"]
        (output / "status.json").write_text(json.dumps({"state": "running", "arm": name, "command": command}))
        print(f"Starting {name}; log: {output / (name + '.log')}", flush=True)
        with (output / f"{name}.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            (output / "status.json").write_text(json.dumps({"state": "failed", "arm": name}))
            raise RuntimeError(f"{name} failed: {output / (name + '.log')}")
        reports[name] = {row["step"]: row for row in map(json.loads, (output / name / "fixed_validation_metrics.jsonl").read_text().splitlines())}
        if args.steps not in reports[name]:
            raise RuntimeError(f"Incomplete training: {name}")
        audits.append(json.loads((output / name / "gradient_audit.jsonl").read_text().splitlines()[-1]))
        scopes.append(json.loads((output / name / "trainable_scope.json").read_text()))
    if scopes[0]["initial_frozen_sha256"] != scopes[1]["initial_frozen_sha256"]:
        raise ValueError("Frozen initialization differs")
    if audits[0]["first_two_updates_batches"] != audits[1]["first_two_updates_batches"]:
        raise ValueError("Initial training batches differ")
    for a, b in zip(reports["control"][0]["deterministic"]["by_lead"], reports["dry_guarded"][0]["deterministic"]["by_lead"], strict=True):
        for key in ("mae", "rmse", "mean_csi"):
            np.testing.assert_allclose(a[key], b[key], rtol=1e-5, atol=1e-7)
    (output / "comparison.json").write_text(json.dumps(reports, indent=2))
    lines = ["# Rollout dry constraint ablation", "", "Values: control / guarded. Compare every lead, not FAR alone.", "",
             "| Step | Lead | Threshold | CSI | POD | FAR |", "|---|---|---|---|---|---|"]
    for step in sorted(reports["control"]):
        for a, b in zip(reports["control"][step]["deterministic"]["by_lead"], reports["dry_guarded"][step]["deterministic"]["by_lead"], strict=True):
            for threshold in ("0.01", "0.1", "0.2", "0.5"):
                x, y = a[f"threshold@{threshold}"], b[f"threshold@{threshold}"]
                cells = " | ".join(f"{x[key]:.5f} / {y[key]:.5f}" for key in ("csi", "pod", "far"))
                lines.append(f"| {step} | {a['lead_index'] + 1} | {threshold} | {cells} |")
    lines.extend(["", "Full intensity/evolution metrics: comparison.json. Rainfall and change plots: each arm/fixed_validation/.",
                  "Selection gates use the initial checkpoint; additionally compare matched-step control before promoting a model."])
    (output / "comparison.md").write_text("\n".join(lines) + "\n")
    (output / "status.json").write_text(json.dumps({"state": "complete"}))
    print(f"Completed: {output / 'comparison.md'}", flush=True)


if __name__ == "__main__":
    main()
