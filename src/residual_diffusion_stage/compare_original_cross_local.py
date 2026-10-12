import argparse
import hashlib
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import default_collate

from src.residual_diffusion_stage.compare_coarse_smoke import build_evaluator, save_preview
from src.residual_diffusion_stage.diagnose_coarse_evolution import evolution_statistics, save_changes, summarize
from src.residual_diffusion_stage.metrics import DeterministicMetricAccumulator


def write_summary(report: dict, path: Path) -> None:
    lines = [
        "# Original pretrained vs cross-attention + local branch",
        "",
        f"Paired samples: {report['protocol']['samples']}; units: mm/min.",
        "Free rollout, no observed future modalities; raw checkpoint weights, no EMA.",
        "This compares trained systems, not an isolated causal effect of adding branches.",
        "",
        "| Metric | Original | Cross + local | Difference (cross + local minus original) |",
        "|---|---:|---:|---:|",
    ]
    a = report["original"]["deterministic"]["overall"]
    b = report["cross_local"]["deterministic"]["overall"]
    for name in ("mae", "rmse", "bias", "mean_csi"):
        lines.append(f"| {name} | {a[name]:.6f} | {b[name]:.6f} | {b[name] - a[name]:+.6f} |")
    for threshold in report["protocol"]["thresholds"]:
        key = f"threshold@{threshold}"
        for name in ("csi", "pod", "far", "f1", "hss"):
            x, y = a[key][name], b[key][name]
            lines.append(f"| {name}@{threshold} | {x:.6f} | {y:.6f} | {y-x:+.6f} |")
    lines.extend(["", "## By lead", "", "| Lead | Threshold | CSI original/new | POD original/new | FAR original/new |", "|---|---|---|---|---|"])
    for a, b in zip(report["original"]["deterministic"]["by_lead"], report["cross_local"]["deterministic"]["by_lead"], strict=True):
        for threshold in report["protocol"]["thresholds"]:
            key = f"threshold@{threshold}"
            values = " | ".join(f"{a[key][name]:.4f}/{b[key][name]:.4f}" for name in ("csi", "pod", "far"))
            lines.append(f"| t+{a['lead_index']+1} | {threshold} | {values} |")
    lines.extend(["", "## Evolution", "", "| Lead | Metric | Original | Cross + local |", "|---|---|---:|---:|"])
    for lead, row in report["original"]["evolution"].items():
        for name in ("active_delta_mae", "change_amplitude_ratio", "growth_direction_recall", "decay_direction_recall", "core_0.5_intensity_ratio", "core_0.5_mae", "fss_0.1_scale_8"):
            x, y = row[name], report["cross_local"]["evolution"][lead][name]
            x_text = "NA" if x is None else f"{x:.6f}"
            y_text = "NA" if y is None else f"{y:.6f}"
            lines.append(f"| t+{lead} | {name} | {x_text} | {y_text} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate original and cross/local checkpoints on identical free-rollout inputs.")
    parser.add_argument("--original-config", type=Path, required=True)
    parser.add_argument("--original-checkpoint", type=Path, required=True)
    parser.add_argument("--cross-local-config", type=Path, required=True)
    parser.add_argument("--cross-local-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())["manifest"]
    if not 0 < args.samples <= len(manifest):
        raise ValueError("samples must be positive and no larger than the fixed manifest")
    manifest = manifest[:args.samples]
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    trainers = {}
    try:
        trainers["original"] = build_evaluator(args.original_config, args.original_checkpoint, out / "original_runtime")
        trainers["cross_local"] = build_evaluator(args.cross_local_config, args.cross_local_checkpoint, out / "cross_local_runtime")
        original, enhanced = trainers.values()
        for name, trainer in trainers.items():
            config = trainer.cfg.rain_prediction_model
            expected = name == "cross_local"
            if bool(config.get("cross_modal_adapter_enabled", False)) != expected or bool(config.get("local_window_refiner_enabled", False)) != expected:
                raise ValueError(f"Unexpected branch configuration for {name}")
            if config.get("consensus_calibration_enabled", False):
                raise ValueError("CCC must be disabled in both models")
            OmegaConf.save(trainer.cfg, out / f"{name}_evaluation_config.yaml")
        if OmegaConf.to_container(original.dataset_cfg.val, resolve=True) != OmegaConf.to_container(enhanced.dataset_cfg.val, resolve=True):
            raise ValueError("Validation preprocessing differs between models")
        if original._resolve_rain_residual_cfg() != enhanced._resolve_rain_residual_cfg():
            raise ValueError("Rain residual interpretation differs between models")
        thresholds = [0.01, 0.1, 0.2, 0.5]
        accumulators = {name: DeterministicMetricAccumulator(thresholds, 4, torch.device("cpu")) for name in trainers}
        evolution_sums = {name: {} for name in trainers}
        per_sample = []
        strongest_mass = -1.0
        for ordinal, item in enumerate(manifest):
            index = item["index"]
            batch = default_collate([original.val_dataset[index]])
            digest = hashlib.sha256()
            for key in sorted(batch):
                if torch.is_tensor(batch[key]):
                    digest.update(key.encode())
                    digest.update(batch[key].contiguous().numpy().tobytes())
            if digest.hexdigest() != item["sha256"]:
                raise ValueError(f"Fixed input checksum mismatch: {index}")
            fields = {}
            row = {"index": index}
            for name, trainer in trainers.items():
                with trainer.accelerator.autocast():
                    context, target, context_time, future_time = trainer._prepare_val_inference_batch(batch)
                    prediction = trainer._rollout_predict_with_settings(
                        context=context, total_future_frames=4, mode="frame", rollout_block_size=1,
                        detach_history=True, context_time=context_time, future_time=future_time,
                        future_modalities=None, use_gt_future_modalities=False,
                    )["rain"]
                rain = trainer._denormalize_rain_for_metrics(prediction.float()).clamp_min(0).cpu()
                truth = trainer._denormalize_rain_for_metrics(target["rain"].float()).clamp_min(0).cpu()
                anchor = trainer._denormalize_rain_for_metrics(context[:, -trainer.rain_c:, -1:].float()).clamp_min(0).cpu()
                if not torch.isfinite(rain).all():
                    raise FloatingPointError(f"Nonfinite prediction: {name}, {index}")
                if "GT" in fields:
                    torch.testing.assert_close(truth, fields["GT"], rtol=0, atol=0)
                fields["GT"] = truth
                fields[name] = rain
                accumulators[name].update(rain, truth)
                for lead, statistics in evolution_statistics(rain, truth, anchor).items():
                    sums = evolution_sums[name].setdefault(lead, {key: 0.0 for key in statistics})
                    for key, value in statistics.items():
                        sums[key] += value
                row[f"{name}_mae"] = (rain - truth).abs().mean().item()
            per_sample.append(row)
            preview = {name: field[0, 0].numpy() for name, field in fields.items()}
            if ordinal < 3:
                save_preview(out / f"rain_index_{index}.png", preview, index)
                save_changes(out / f"changes_index_{index}.png", fields, anchor)
            mass = truth.sum().item()
            if mass > strongest_mass:
                strongest_mass = mass
                save_preview(out / "largest_target_rain_mass.png", preview, index)
                strongest_index = index
            if (ordinal + 1) % 10 == 0 or ordinal + 1 == len(manifest):
                print(f"Paired samples: {ordinal + 1}/{len(manifest)}", flush=True)
        report = {
            "protocol": {
                "samples": len(manifest), "unit": "mm/min", "thresholds": thresholds,
                "inference": "free rollout; no observed future modalities", "weights": "raw, EMA disabled",
                "selection": "existing fixed validation manifest; existing rain-area filter retained",
                "interpretation": "trained-system comparison; extra fine-tuning is a confound for branch attribution",
                "largest_target_rain_mass_index": strongest_index,
            },
            "checkpoints": {"original": str(args.original_checkpoint), "cross_local": str(args.cross_local_checkpoint)},
            "manifest": manifest, "per_sample": per_sample,
            **{name: {"deterministic": accumulator.result(), "evolution": summarize(evolution_sums[name])} for name, accumulator in accumulators.items()},
        }
        (out / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        write_summary(report, out / "comparison.md")
        print(f"Completed: {out / 'comparison.md'}", flush=True)
    finally:
        for trainer in trainers.values():
            trainer._close_tensorboard_writer()


if __name__ == "__main__":
    main()
