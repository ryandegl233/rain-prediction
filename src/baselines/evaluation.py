import csv
import hashlib
import json
import time
from collections.abc import Iterable, Mapping
from itertools import islice
from pathlib import Path

import torch
from torch import nn

from src.baselines.contracts import prepare_history
from src.rain_priority.evaluation import RainLeadMetrics


@torch.inference_mode()
def evaluate_forecasts(
    models: Mapping[str, nn.Module],
    batches: Iterable[Mapping[str, torch.Tensor]],
    combinations: list[tuple[str, ...]],
    thresholds: list[float],
    max_batches: int = -1,
    progress_every: int = 25,
) -> dict:
    if not models or not combinations or max_batches == 0 or max_batches < -1:
        raise ValueError("Models, combinations and a positive limit or -1 are required.")
    labels = ["+".join(names) for names in combinations]
    if len(set(labels)) != len(labels):
        raise ValueError("Duplicate modality combinations.")
    accumulators = {}
    results = {}
    for label, names in zip(labels, combinations, strict=True):
        results[label] = {}
        for name, model in models.items():
            required = model.required_modalities
            if set(required) - set(names):
                results[label][name] = {
                    "status": "skipped",
                    "reason": f"Requires observed {', '.join(required)} history.",
                }
            else:
                accumulators[label, name] = RainLeadMetrics(thresholds)
                results[label][name] = {
                    "status": "ok",
                    "parameters": sum(value.numel() for value in model.parameters()),
                }

    started = time.perf_counter()
    windows = []
    samples = 0
    modes = {name: model.training for name, model in models.items()}
    for model in models.values():
        model.eval()
    try:
        selected = islice(batches, max_batches) if max_batches > 0 else batches
        for index, batch in enumerate(selected):
            target = batch["rain_future"].detach().float().cpu()
            if target.ndim != 5 or target.shape[1:3] != (1, 4) or not torch.isfinite(target).all():
                raise ValueError("rain_future must be finite [B,1,4,H,W].")
            past_stamps = batch["time_past_timestamp"].detach().cpu()
            future_stamps = batch["time_future_timestamp"].detach().cpu()
            if past_stamps.shape != (target.shape[0], 4) or future_stamps.shape != past_stamps.shape:
                raise ValueError("Window timestamps must be [B,4] for both history and future.")
            stamps = torch.cat([past_stamps, future_stamps], dim=1)
            if not torch.isfinite(stamps).all() or not torch.allclose(
                stamps.diff(dim=1), torch.full_like(stamps[:, 1:], 1800)
            ):
                raise ValueError("Evaluation windows must have consecutive 30-minute timestamps.")
            windows.extend(stamps.tolist())
            samples += target.shape[0]
            classical_cache = {}
            for label, names in zip(labels, combinations, strict=True):
                history = prepare_history(batch, names)
                for name, model in models.items():
                    if (label, name) not in accumulators:
                        continue
                    if model.required_modalities == ("rain",) and name in classical_cache:
                        prediction = classical_cache[name]
                    else:
                        prediction = model(history).detach().float().cpu()
                        if model.required_modalities == ("rain",):
                            classical_cache[name] = prediction
                    if prediction.shape != target.shape:
                        raise ValueError(
                            f"{name} prediction shape {tuple(prediction.shape)} != target {tuple(target.shape)}"
                        )
                    accumulators[label, name].update(prediction, target)
            if progress_every > 0 and (index + 1) % progress_every == 0:
                print(f"Evaluated {samples} windows ({index + 1} batches)", flush=True)
    finally:
        for name, model in models.items():
            model.train(modes[name])
    if samples == 0:
        raise ValueError("Evaluation is empty; no windows were observed.")

    for (label, name), accumulator in accumulators.items():
        metrics = accumulator.compute()
        metrics["all/mae"] = float(accumulator.errors[:, 0].sum() / (4 * accumulator.pixels))
        metrics["all/mse"] = float(accumulator.errors[:, 1].sum() / (4 * accumulator.pixels))
        results[label][name].update(metrics=metrics, samples=samples)
    window_payload = json.dumps(sorted(windows), separators=(",", ":"), allow_nan=False)
    return {
        "protocol": "fixed_origin_history_only",
        "history_frames": 4,
        "lead_minutes": [30, 60, 90, 120],
        "samples": samples,
        "window_sha256": hashlib.sha256(window_payload.encode()).hexdigest(),
        "windows": windows,
        "max_batches": max_batches,
        "elapsed_seconds": time.perf_counter() - started,
        "results": results,
    }


def write_report(report: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    (output_dir / "metrics.json").write_text(payload + "\n", encoding="utf-8")
    (output_dir / "windows.json").write_text(json.dumps(report["windows"]) + "\n", encoding="utf-8")
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["modalities", "model", "metric", "value"])
        for combination, models in report["results"].items():
            for name, values in models.items():
                if values["status"] == "skipped":
                    writer.writerow([combination, name, "skipped", values["reason"]])
                for metric, value in values.get("metrics", {}).items():
                    writer.writerow([combination, name, metric, value])
