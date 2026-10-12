"""Evaluate free-rollout evolution against GT and a last-observation baseline."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import default_collate

from src.residual_diffusion_stage.compare_coarse_smoke import build_evaluator
from src.residual_diffusion_stage.metrics import DeterministicMetricAccumulator


def evolution_statistics(pred: torch.Tensor, truth: torch.Tensor, anchor: torch.Tensor) -> dict[str, dict[str, float]]:
    pred_previous = torch.cat([anchor, pred[:, :, :-1]], dim=2)
    true_previous = torch.cat([anchor, truth[:, :, :-1]], dim=2)
    dp, dt = pred - pred_previous, truth - true_previous
    result = {}
    for lead in range(truth.shape[2]):
        p, y, a, b = (field[:, :, lead].double() for field in (pred, truth, dp, dt))
        active = (y >= 0.01) | (true_previous[:, :, lead] >= 0.01)
        change = active & (b.abs() >= 0.01)
        growth, decay = change & (b > 0), change & (b < 0)
        row = {
            "delta_abs_error_sum": ((a - b).abs() * active).sum().item(),
            "active_pixels": active.sum().item(),
            "pred_change_sum": (a.abs() * active).sum().item(),
            "gt_change_sum": (b.abs() * active).sum().item(),
            "growth_count": growth.sum().item(),
            "growth_correct": ((a > 0.001) & growth).sum().item(),
            "decay_count": decay.sum().item(),
            "decay_correct": ((a < -0.001) & decay).sum().item(),
        }
        for threshold in (0.1, 0.5, 1.0):
            core = y >= threshold
            key = f"core_{threshold}"
            row.update({
                f"{key}_pixels": core.sum().item(),
                f"{key}_pred_sum": (p * core).sum().item(),
                f"{key}_gt_sum": (y * core).sum().item(),
                f"{key}_error_sum": ((p - y).abs() * core).sum().item(),
            })
        for scale in (1, 4, 8):
            p_fraction = F.avg_pool2d((p >= 0.1).float(), scale, stride=1)
            y_fraction = F.avg_pool2d((y >= 0.1).float(), scale, stride=1)
            row[f"fss_{scale}_error"] = (p_fraction - y_fraction).square().sum().item()
            row[f"fss_{scale}_reference"] = (p_fraction.square() + y_fraction.square()).sum().item()
        result[str(lead + 1)] = row
    return result


def summarize(sums: dict) -> dict:
    result = {}
    for lead, row in sums.items():
        def ratio(numerator: str, denominator: str) -> float | None:
            return row[numerator] / row[denominator] if row[denominator] > 0 else None

        out = {
            "active_delta_mae": ratio("delta_abs_error_sum", "active_pixels"),
            "change_amplitude_ratio": ratio("pred_change_sum", "gt_change_sum"),
            "growth_direction_recall": ratio("growth_correct", "growth_count"),
            "decay_direction_recall": ratio("decay_correct", "decay_count"),
        }
        for threshold in (0.1, 0.5, 1.0):
            key = f"core_{threshold}"
            out[f"{key}_intensity_ratio"] = ratio(f"{key}_pred_sum", f"{key}_gt_sum")
            out[f"{key}_mae"] = ratio(f"{key}_error_sum", f"{key}_pixels")
        for scale in (1, 4, 8):
            score = ratio(f"fss_{scale}_error", f"fss_{scale}_reference")
            out[f"fss_0.1_scale_{scale}"] = None if score is None else 1 - score
        result[lead] = out
    return result


def save_changes(path: Path, fields: dict[str, torch.Tensor], anchor: torch.Tensor) -> None:
    fig, axes = plt.subplots(len(fields), 4, figsize=(14, 3 * len(fields)), constrained_layout=True)
    for row, (name, field) in enumerate(fields.items()):
        difference = field - torch.cat([anchor, field[:, :, :-1]], dim=2)
        for lead in range(4):
            im = axes[row, lead].imshow(difference[0, 0, lead], cmap="RdBu_r", vmin=-0.2, vmax=0.2)
            axes[row, lead].set_title(f"{name} change | t+{lead + 1}")
            axes[row, lead].set_axis_off()
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Rain change (mm/min); red growth, blue decay")
    fig.savefig(path, dpi=130)
    plt.close(fig)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--baseline-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.comparison.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    trainers = {
        name: build_evaluator(
            args.baseline_config if name == "baseline" else args.config,
            Path(checkpoint), args.output / f"{name}_runtime",
        )
        for name, checkpoint in source["checkpoints"].items()
    }
    names = ["persistence", *trainers]
    stats = {name: {} for name in names}
    metrics = {name: DeterministicMetricAccumulator([0.01, 0.1, 0.2, 0.5], 4, torch.device("cpu")) for name in names}
    dataset = next(iter(trainers.values())).val_dataset
    strongest_mass, strongest_fields, strongest_anchor = -1, None, None
    for ordinal, item in enumerate(source["manifest"]):
        batch = default_collate([dataset[item["index"]]])
        digest = hashlib.sha256()
        for key in sorted(batch):
            if torch.is_tensor(batch[key]):
                digest.update(key.encode())
                digest.update(batch[key].contiguous().numpy().tobytes())
        if digest.hexdigest() != item["sha256"]:
            raise ValueError(f"Validation input changed for index {item['index']}")
        fields = {}
        for name, trainer in trainers.items():
            with trainer.accelerator.autocast():
                context, target, context_time, future_time = trainer._prepare_val_inference_batch(batch)
                prediction = trainer._rollout_predict_with_settings(
                    context=context, total_future_frames=4, mode="frame", rollout_block_size=1,
                    detach_history=True, context_time=context_time, future_time=future_time,
                    future_modalities=None, use_gt_future_modalities=False,
                )["rain"]
            fields[name] = trainer._denormalize_rain_for_metrics(prediction.float()).clamp_min(0).cpu()
            truth = trainer._denormalize_rain_for_metrics(target["rain"].float()).clamp_min(0).cpu()
            anchor = trainer._denormalize_rain_for_metrics(context[:, -trainer.rain_c:, -1:].float()).clamp_min(0).cpu()
        fields["persistence"] = anchor.expand_as(truth)
        for name, field in fields.items():
            if not torch.isfinite(field).all():
                raise ValueError(f"Nonfinite forecast: {name}")
            metrics[name].update(field, truth)
            values = evolution_statistics(field, truth, anchor)
            for lead, row in values.items():
                total = stats[name].setdefault(lead, {key: 0.0 for key in row})
                for key, value in row.items():
                    total[key] += value
        display = {"GT": truth, **fields}
        if ordinal < 3:
            save_changes(args.output / f"change_index_{item['index']}.png", display, anchor)
        if truth.sum().item() > strongest_mass:
            strongest_mass, strongest_fields, strongest_anchor = truth.sum().item(), display, anchor
        if (ordinal + 1) % 10 == 0:
            print(f"Evolution diagnostics: {ordinal + 1}/{len(source['manifest'])}", flush=True)
    save_changes(args.output / "largest_rain_mass_changes.png", strongest_fields, strongest_anchor)
    report = {
        "protocol": source["protocol"], "checkpoints": source["checkpoints"],
        "definitions": {
            "delta": "forecast[t]-forecast[t-1], anchored to last observation at t+1; compared to GT increments",
            "active": "GT current or previous rain >=0.01 mm/min",
            "direction": "GT change magnitude >=0.01; forecast signed change must exceed 0.001 mm/min",
            "core_ratio": "sum forecast / sum GT at GT core pixels; penalizes both displacement and intensity error",
            "fss": "0.1 mm/min binary fields, stride-1 valid spatial pooling; pooled squared errors",
            "scope": "60 fixed rain-filtered validation windows, not full validation or an independent test set",
        },
        "evolution": {name: summarize(values) for name, values in stats.items()},
        "deterministic": {name: metric.result() for name, metric in metrics.items()},
        "raw_sums": stats,
    }
    (args.output / "evolution.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for trainer in trainers.values():
        trainer._close_tensorboard_writer()
    print(f"Saved {args.output / 'evolution.json'}", flush=True)


if __name__ == "__main__":
    main()
