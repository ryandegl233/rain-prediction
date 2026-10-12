"""Compare free autoregressive rollout with rain-history teacher forcing."""

import argparse
import json
from pathlib import Path

import torch

from src.residual_diffusion_stage.train import (
    DEFAULT_BASE_CONFIG,
    _build_coarse_trainer,
    _limit,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-config-name", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--max-val-batches", type=int, default=60)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.01, 0.1, 0.2, 0.5])
    return parser.parse_args()


def _new_stats(leads: int, thresholds: list[float]) -> dict:
    return {
        "mae_sum": [0.0] * leads,
        "count": [0] * leads,
        "events": {
            f"{threshold:g}": {
                "hits": [0] * leads,
                "misses": [0] * leads,
                "false_alarms": [0] * leads,
            }
            for threshold in thresholds
        },
    }


def _update(stats: dict, prediction: torch.Tensor, target: torch.Tensor, thresholds: list[float]) -> None:
    prediction = prediction.detach().float().clamp_min(0.0)
    target = target.detach().float().clamp_min(0.0)
    leads = int(target.shape[2])
    for lead in range(leads):
        stats["mae_sum"][lead] += float((prediction[:, :, lead] - target[:, :, lead]).abs().mean().item())
        stats["count"][lead] += 1
        for threshold in thresholds:
            predicted_event = prediction[:, :, lead] >= threshold
            target_event = target[:, :, lead] >= threshold
            key = f"{threshold:g}"
            event = stats["events"][key]
            event["hits"][lead] += int((predicted_event & target_event).sum().item())
            event["misses"][lead] += int((~predicted_event & target_event).sum().item())
            event["false_alarms"][lead] += int((predicted_event & ~target_event).sum().item())


def _finish(stats: dict, thresholds: list[float]) -> dict:
    result = {"mae_by_lead": []}
    for mae, count in zip(stats["mae_sum"], stats["count"]):
        result["mae_by_lead"].append(mae / max(count, 1))
    result["thresholds"] = {}
    for threshold in thresholds:
        key = f"{threshold:g}"
        result["thresholds"][key] = {"csi_by_lead": [], "pod_by_lead": [], "far_by_lead": []}
        event = stats["events"][key]
        for hits, misses, false_alarms in zip(event["hits"], event["misses"], event["false_alarms"]):
            csi_denominator = hits + misses + false_alarms
            pod_denominator = hits + misses
            far_denominator = hits + false_alarms
            result["thresholds"][key]["csi_by_lead"].append(
                hits / csi_denominator if csi_denominator else 0.0
            )
            result["thresholds"][key]["pod_by_lead"].append(
                hits / pod_denominator if pod_denominator else 0.0
            )
            result["thresholds"][key]["far_by_lead"].append(
                false_alarms / far_denominator if far_denominator else 0.0
            )
    return result


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    thresholds = sorted(set(float(value) for value in args.thresholds))
    output_dir = args.output_dir.expanduser().resolve()
    trainer = _build_coarse_trainer(
        args.base_config_name,
        args.coarse_checkpoint,
        output_dir / "rollout_diagnostic_runtime",
        batch_size=1,
        accumulation_steps=1,
        num_workers=0,
    )
    free_stats = None
    forced_stats = None
    evaluated_batches = 0
    for batch in _limit(trainer.val_dataloader, args.max_val_batches):
        context, target, context_time, target_time = trainer._prepare_val_inference_batch(batch)
        target_rain = target["rain"]
        total_future_frames = int(target_rain.shape[2])
        free = trainer._rollout_predict_with_settings(
                context=context,
                total_future_frames=total_future_frames,
                mode="frame",
                rollout_block_size=1,
                detach_history=True,
                context_time=context_time,
                future_time=target_time,
                future_modalities=None,
                use_gt_future_modalities=False,
            )["rain"]
        forced = trainer._rollout_predict_with_settings(
                context=context,
                total_future_frames=total_future_frames,
                mode="frame",
                rollout_block_size=1,
                detach_history=True,
                context_time=context_time,
                future_time=target_time,
                future_modalities=None,
                use_gt_future_modalities=False,
                future_rain=target_rain,
                use_gt_future_rain=True,
            )["rain"]
        free_metric = trainer._denormalize_rain_for_metrics(free)
        forced_metric = trainer._denormalize_rain_for_metrics(forced)
        target_metric = trainer._denormalize_rain_for_metrics(target_rain)
        if free_stats is None:
            free_stats = _new_stats(total_future_frames, thresholds)
            forced_stats = _new_stats(total_future_frames, thresholds)
        _update(free_stats, free_metric, target_metric, thresholds)
        _update(forced_stats, forced_metric, target_metric, thresholds)
        evaluated_batches += 1

    if evaluated_batches == 0 or free_stats is None or forced_stats is None:
        raise RuntimeError("No validation batches were evaluated")
    report = {
        "evaluated_batches": evaluated_batches,
        "thresholds_mm_per_min": thresholds,
        "free_rollout": _finish(free_stats, thresholds),
        "rain_history_teacher_forced": _finish(forced_stats, thresholds),
    }
    report["far_reduction_teacher_forced"] = {}
    for threshold in thresholds:
        key = f"{threshold:g}"
        free_far = report["free_rollout"]["thresholds"][key]["far_by_lead"]
        forced_far = report["rain_history_teacher_forced"]["thresholds"][key]["far_by_lead"]
        report["far_reduction_teacher_forced"][key] = [a - b for a, b in zip(free_far, forced_far)]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rollout_far_diagnostic.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
