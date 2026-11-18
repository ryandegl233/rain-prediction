import csv
import hashlib
import json
from collections.abc import Iterable, Mapping
from itertools import islice
from pathlib import Path

import torch
from omegaconf import DictConfig
from torch import nn

from src.baselines_stage3.contracts import latest_rain, make_inputs, sample_masks, window_times
from src.baselines_stage3.metrics import RainLeadMetrics


@torch.inference_mode()
def evaluate(models: Mapping[str, nn.Module], batches: Iterable[dict], cfg: DictConfig) -> dict:
    scenarios = list(cfg.baseline.scenarios)
    satellite_only = scenarios == ["satellite_only"]
    if "satellite_only" in scenarios and not satellite_only:
        raise ValueError("Evaluate strict satellite-only separately from anchored multimodal scenarios.")
    if satellite_only and any(not getattr(model, "satellite_only", False) for model in models.values()):
        raise ValueError("Satellite-only evaluation requires satellite-only models.")
    limit = cfg.evaluation.max_batches
    if not models or not scenarios or len(set(scenarios)) != len(scenarios) or limit == 0 or limit < -1:
        raise ValueError("Models, unique scenarios, and max_batches=-1 or a positive limit are required.")
    if cfg.evaluation.mixed_precision not in ("no", "bf16", "fp16"):
        raise ValueError("evaluation.mixed_precision must be no, bf16, or fp16.")
    execution = {}
    for name, model in models.items():
        parameter = next(model.parameters(), None)
        execution[name] = {
            "device": str(parameter.device) if parameter is not None else "cpu",
            "inference_precision": cfg.evaluation.mixed_precision if parameter is not None else "no",
        }
    accumulators = {
        (scenario, name): RainLeadMetrics(list(cfg.val.csi_thresholds)) for scenario in scenarios for name in models
    }
    windows = {scenario: [] for scenario in scenarios}
    counts = {scenario: {"rain_observations": [0] * 5, "last_rain_age_steps": [0] * 4} for scenario in scenarios}
    modes = {name: model.training for name, model in models.items()}
    for model in models.values():
        model.eval()
    samples = 0
    try:
        selected = islice(batches, limit) if limit > 0 else batches
        for index, batch in enumerate(selected):
            target = batch["rain_future"].detach().float().cpu()
            if target.ndim != 5 or target.shape[1:3] != (1, 4) or not torch.isfinite(target).all():
                raise ValueError("rain_future must be finite [B,1,4,H,W].")
            past = batch["time_past_timestamp"].detach().cpu()
            future = batch["time_future_timestamp"].detach().cpu()
            if past.shape != (target.shape[0], 4) or future.shape != past.shape:
                raise ValueError("Historical and future timestamps must be [B,4].")
            stamps = torch.cat([past, future], dim=1)
            if not torch.isfinite(stamps).all() or not torch.all(stamps.diff(dim=1) == 1800):
                raise ValueError("Evaluation requires consecutive 30-minute timestamps.")
            for scenario in scenarios:
                mask = sample_masks(
                    past, scenario, cfg.evaluation.mask_seed, cfg.train.next_pred.missing_modality.full_modality_prob
                )
                inputs = make_inputs(batch, mask, any(model.uses_auxiliary_observations for model in models.values()))
                observed = mask[:, 2].sum(dim=1)
                ages = torch.empty(0, dtype=torch.long) if satellite_only else 3 - latest_rain(inputs)[1]
                for value in observed.tolist():
                    counts[scenario]["rain_observations"][value] += 1
                for value in ages.tolist():
                    counts[scenario]["last_rain_age_steps"][value] += 1
                windows[scenario].extend(
                    {"calendar_times": calendar, "timestamps": stamp, "available": availability}
                    for calendar, stamp, availability in zip(
                        window_times(stamps), stamps.tolist(), mask.tolist(), strict=True
                    )
                )
                for name, model in models.items():
                    precision = execution[name]["inference_precision"]
                    with torch.autocast(
                        device_type=torch.device(execution[name]["device"]).type,
                        dtype=torch.bfloat16 if precision == "bf16" else torch.float16,
                        enabled=precision != "no",
                    ):
                        prediction = model(inputs).detach().float().cpu()
                    if prediction.shape != target.shape:
                        raise ValueError(f"Invalid prediction shape for {name}: {tuple(prediction.shape)}")
                    accumulators[scenario, name].update(prediction, target)
            samples += target.shape[0]
            if cfg.evaluation.progress_every > 0 and (index + 1) % cfg.evaluation.progress_every == 0:
                print(f"Evaluated {samples} windows", flush=True)
    finally:
        for name, model in models.items():
            model.train(modes[name])
    if samples == 0:
        raise ValueError("No evaluation windows were observed.")
    results = {scenario: {} for scenario in scenarios}
    for (scenario, name), accumulator in accumulators.items():
        metrics = accumulator.compute()
        metrics["all/mae"] = float(accumulator.errors[:, 0].sum() / (4 * accumulator.pixels))
        metrics["all/mse"] = float(accumulator.errors[:, 1].sum() / (4 * accumulator.pixels))
        results[scenario][name] = {
            **execution[name],
            "parameters": sum(value.numel() for value in models[name].parameters()),
            "uses_arriving_gt_radar_satellite": models[name].uses_auxiliary_observations,
            "metrics": metrics,
        }
        if hasattr(models[name], "checkpoint_metadata"):
            results[scenario][name]["checkpoint"] = models[name].checkpoint_metadata
    return {
        "protocol": "fixed_origin_satellite_only"
        if satellite_only
        else (
            "progressively_observed_radar_satellite"
            if cfg.val.rollout_use_gt_future_modalities
            else "fixed_origin_history_only"
        ),
        "history_frames": 4,
        "lead_minutes": [30, 60, 90, 120],
        "samples": samples,
        "mask_seed": cfg.evaluation.mask_seed,
        "window_identity": "dataset naive calendar time, reconstructed using producer host local time; POSIX timestamps are audit-only",
        "first_frame_all_modalities": not satellite_only,
        "mask_applies_to": "satellite history only; no future observations"
        if satellite_only
        else "historical_frames_only; arriving radar/satellite are fully observed",
        "rain_units": "dataset_native_rain_interpolated",
        "future_rain_as_input": False,
        "availability_counts": counts,
        "window_mask_sha256": {
            scenario: hashlib.sha256(
                json.dumps(
                    sorted(
                        [
                            {"calendar_times": record["calendar_times"], "available": record["available"]}
                            for record in records
                        ],
                        key=lambda x: x["calendar_times"],
                    ),
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            for scenario, records in windows.items()
        },
        "windows": windows,
        "results": results,
        "lk_sparse_policy": "last two observed rain frames, interval-normalized velocity, extrapolation includes age of last observation; one observation uses zero motion",
    }


def write_report(report: dict, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (destination / "windows.json").write_text(json.dumps(report["windows"]) + "\n", encoding="utf-8")
    with (destination / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["scenario", "method", "metric", "value"])
        for scenario, methods in report["results"].items():
            for method, result in methods.items():
                for metric, value in result["metrics"].items():
                    writer.writerow([scenario, method, metric, value])
