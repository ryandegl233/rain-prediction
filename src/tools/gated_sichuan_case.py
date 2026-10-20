import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from accelerate import Accelerator
from matplotlib.axes import Axes
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import default_collate

from src.gated_modality_rain.trainer import GatedModalityRainTrainer
from src.utils.visualization.plot_ragion import (
    DEFAULT_GEO_BOUNDS,
    SICHUAN_BOUNDARY_URL,
    _draw_geojson_boundary,
    _load_geojson,
    _rain_colormap,
)


def build_case_times(
    first_future: str, history_frames: int, future_frames: int, interval_minutes: int
) -> tuple[list[datetime], list[datetime]]:
    start = datetime.fromisoformat(first_future)
    if min(history_frames, future_frames, interval_minutes) <= 0:
        raise ValueError("Frame counts and interval must be positive.")
    if (
        start.tzinfo is not None
        or start.second
        or start.microsecond
        or (start.hour * 60 + start.minute) % interval_minutes
    ):
        raise ValueError("Use a naive Beijing timestamp on the dataset interval grid.")
    delta = timedelta(minutes=interval_minutes)
    return (
        [start - delta * i for i in range(history_frames, 0, -1)],
        [start + delta * i for i in range(future_frames)],
    )


def find_case_window(
    pairs: list[tuple[list[datetime], list[datetime]]], past: list[datetime], future: list[datetime]
) -> int:
    matches = [i for i, (history, targets) in enumerate(pairs) if list(history) == past and list(targets) == future]
    if len(matches) != 1:
        raise ValueError(f"Expected one exact continuous window; found {len(matches)}. No nearby sample substituted.")
    return matches[0]


class CaseRuntime(GatedModalityRainTrainer):
    def __init__(self, cfg: DictConfig, model: torch.nn.Module, *, device: str, precision: str) -> None:
        self.accelerator = Accelerator(cpu=device == "cpu", mixed_precision=precision)
        if self.accelerator.num_processes != 1:
            raise ValueError("Run this single-case tool with python, not distributed launch.")
        self.device = self.accelerator.device
        if device == "cuda" and self.device.type != "cuda":
            raise RuntimeError("CUDA was requested but is unavailable.")
        self.model = model.to(self.device).eval()
        self.train_cfg = cfg.train
        self.dataset_cfg = cfg.dataset
        self.radar_c = int(model.radar_out_channels)
        self.satellite_c = int(model.satellite_out_channels)
        self.rain_c = int(model.rain_out_channels)
        self._init_rain_norm_params()

    @torch.inference_mode()
    def predict(self, context: torch.Tensor, past: list[datetime], future: list[datetime]) -> dict[str, torch.Tensor]:
        if len(past) + len(future) > self.model.max_frames:
            raise ValueError("The requested growing-history rollout exceeds model.max_frames.")
        if context.shape[2] != len(past):
            raise ValueError("History tensor and timestamps have different lengths.")
        phases = [
            torch.tensor(
                [[(t.hour * 60 + t.minute) / 1440 for t in times]],
                device=self.device,
                dtype=torch.float32,
            )
            for times in (past, future)
        ]
        output = self._rollout_predict_with_settings(
            context=context.to(self.device, dtype=torch.float32),
            total_future_frames=len(future),
            mode="frame",
            rollout_block_size=1,
            detach_history=True,
            context_time=phases[0],
            future_time=phases[1],
            future_modalities=None,
            use_gt_future_modalities=False,
        )
        return {key: value.detach().float().cpu() for key, value in output.items()}


def _draw_rain(
    ax: Axes,
    rain: np.ndarray,
    title: str,
    boundary: dict,
    geo_bounds: tuple[float, float, float, float],
) -> None:
    cmap, norm = _rain_colormap()
    ax.imshow(
        rain,
        cmap=cmap,
        norm=norm,
        extent=geo_bounds,
        origin="upper",
        interpolation="nearest",
        alpha=1.0,
    )
    _draw_geojson_boundary(ax, boundary, color="#e57373", linewidth=0.55)
    ax.set_xlim(geo_bounds[:2])
    ax.set_ylim(geo_bounds[2:])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def save_case_figures(
    output: Path,
    history: np.ndarray,
    prediction: np.ndarray,
    truth: np.ndarray,
    past: list[datetime],
    future: list[datetime],
    boundary_source: Path,
    geo_bounds: tuple[float, float, float, float],
    *,
    dpi: int = 160,
) -> None:
    boundary = _load_geojson(boundary_source)
    output.mkdir(parents=True, exist_ok=True)
    origin = past[-1]
    for name, arrays, dates in (("prediction", prediction, future), ("truth", truth, future)):
        (output / name).mkdir(exist_ok=True)
        for i, (rain, valid) in enumerate(zip(arrays, dates, strict=True)):
            lead = int((valid - origin).total_seconds() / 60)
            fig, ax = plt.subplots(figsize=(5, 4))
            _draw_rain(ax, rain, f"{name.title()} | {valid:%m-%d %H:%M} | +{lead} min", boundary, geo_bounds)
            fig.tight_layout()
            fig.savefig(output / name / f"lead_{i + 1:02d}_{valid:%Y%m%d_%H%M}.png", dpi=dpi)
            plt.close(fig)

    for name, arrays, dates in (("history", history, past), ("prediction", prediction, future)):
        cols = min(4, len(dates))
        rows = math.ceil(len(dates) / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.3 * rows), squeeze=False)
        for i, ax in enumerate(axes.flat):
            ax.axis("off")
            if i < len(dates):
                _draw_rain(ax, arrays[i], f"{name.title()} | {dates[i]:%m-%d %H:%M}", boundary, geo_bounds)
        fig.suptitle(f"Nominal Beijing times | last observed frame {origin:%Y-%m-%d %H:%M}")
        fig.tight_layout()
        fig.savefig(output / f"{name}_grid.png", dpi=dpi)
        plt.close(fig)

    cols = min(4, len(future))
    groups = math.ceil(len(future) / cols)
    fig, axes = plt.subplots(2 * groups, cols, figsize=(4 * cols, 6.6 * groups), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, valid in enumerate(future):
        row, col = 2 * (i // cols), i % cols
        lead = int((valid - origin).total_seconds() / 60)
        for offset, (label, arrays) in enumerate((("Prediction", prediction), ("Truth", truth))):
            _draw_rain(
                axes[row + offset, col],
                arrays[i],
                f"{label} | {valid:%m-%d %H:%M} | +{lead} min",
                boundary,
                geo_bounds,
            )
    fig.suptitle(f"{len(future)}-step case study | last observed nominal time {origin:%Y-%m-%d %H:%M}")
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    cmap, norm = _rain_colormap()
    color_ax = fig.add_axes((0.28, 0.018, 0.44, 0.012))
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=color_ax, orientation="horizontal")
    color_ax.set_xlabel("Rain value (dataset scale; identical bins for prediction and truth)")
    fig.savefig(output / "comparison_grid.png", dpi=dpi)
    plt.close(fig)

    animation: list[Image.Image] = []
    for i, valid in enumerate(future):
        frames = []
        for name in ("prediction", "truth"):
            with Image.open(output / name / f"lead_{i + 1:02d}_{valid:%Y%m%d_%H%M}.png") as image:
                frames.append(image.convert("RGB"))
        panel = Image.new("RGB", (frames[0].width * 2, frames[0].height), "white")
        panel.paste(frames[0], (0, 0))
        panel.paste(frames[1], (frames[0].width, 0))
        animation.append(panel)
    animation[0].save(
        output / "comparison.gif",
        save_all=True,
        append_images=animation[1:],
        duration=900,
        loop=0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Old gated-model checkpoint: exact-date free rollout and Sichuan maps."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, default=80000)
    parser.add_argument("--data-root", type=Path, default=Path("data2/litdata_train_2025/litdata_interval_30"))
    parser.add_argument("--first-future", default="2025-08-10 23:30:00")
    parser.add_argument("--future-frames", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--boundary-source", default=SICHUAN_BOUNDARY_URL)
    parser.add_argument("--geo-bounds", type=float, nargs=4, default=DEFAULT_GEO_BOUNDS)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--precision", choices=("bf16", "fp16", "no"))
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args()

    config_path = args.run_dir / "config/config_total.yaml"
    weights = args.run_dir / f"checkpoint-{args.step:08d}/model.safetensors"
    cfg = OmegaConf.load(config_path)
    if cfg.rain_prediction_model.get("_target_") != "src.gated_modality_rain.model.GatedModalityRainModel":
        raise ValueError("This tool requires the old GatedModalityRainModel run configuration.")
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if int(cfg.dataset.n_past) != 4:
        raise ValueError("This case study expects the original four-history-frame model.")
    past, future = build_case_times(
        args.first_future, int(cfg.dataset.n_past), args.future_frames, int(cfg.dataset.time_interval)
    )
    if len(past) + len(future) > int(cfg.rain_prediction_model.max_frames):
        raise ValueError("Requested window exceeds max_frames; the architecture will not be changed.")
    months = sorted({t.strftime("%Y%m") for t in past + future})
    dirs = [args.data_root.resolve() / month for month in months]
    metadata = pd.concat([pd.read_parquet(folder / "metadata.parquet") for folder in dirs], ignore_index=True)
    counts = pd.to_datetime(metadata["time"]).value_counts()
    missing = [str(t) for t in past + future if int(counts.get(t, 0)) != 1]
    if missing:
        raise ValueError(f"Missing or duplicate nominal frames: {missing}. No temporal interpolation is performed.")

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {args.output_dir}")
    boundary = _load_geojson(args.boundary_source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    boundary_path = args.output_dir / "sichuan_boundary.geojson"
    boundary_path.write_text(json.dumps(boundary, ensure_ascii=False), encoding="utf-8")

    loader_cfg = OmegaConf.create(OmegaConf.to_container(cfg.dataset.val, resolve=True))
    loader_cfg.update(
        {
            "inp_dirs": [str(folder) for folder in dirs],
            "n_past": len(past),
            "n_futures": len(future),
            "batch_size": 1,
            "num_workers": 0,
            "persistent_workers": False,
            "pin_memory": False,
            "shuffle": False,
            "drop_last": False,
            "is_cycled": False,
            "iter_index_mode": "sequential",
            "rain_ratio_filter_enabled": False,
            "aug_enabled": False,
        }
    )
    dataset, _ = hydra.utils.instantiate(loader_cfg)
    index = find_case_window(dataset.times_pairs, past, future)
    batch = default_collate([dataset[index]])
    context = torch.cat([batch[f"{name}_past"] for name in ("radar", "satellite", "rain")], dim=1)
    print(f"Exact window {index}: history {past[0]} to {past[-1]}, future {future[0]} to {future[-1]}", flush=True)

    model = hydra.utils.instantiate(cfg.rain_prediction_model)
    model.load_state_dict(load_file(str(weights), device="cpu"), strict=True)
    precision = args.precision or ("no" if args.device == "cpu" else cfg.accelerator.get("mixed_precision", "no"))
    runtime = CaseRuntime(cfg, model, device=args.device, precision=precision)
    print(
        f"Loaded model weights at step {args.step}; precision={precision}; rolling {len(future)} frames...", flush=True
    )
    prediction = runtime.predict(context, past, future)
    history_rain = runtime._denormalize_rain_for_metrics(batch["rain_past"]).numpy()[0, 0]
    truth_rain = runtime._denormalize_rain_for_metrics(batch["rain_future"]).numpy()[0, 0]
    prediction_rain = runtime._denormalize_rain_for_metrics(prediction["rain"]).numpy()[0, 0]
    if not all(np.isfinite(array).all() for array in (history_rain, truth_rain, prediction_rain)):
        raise ValueError("Nonfinite rain values detected; refusing to silently replace values for plotting.")
    np.savez_compressed(
        args.output_dir / "forecast_raw.npz",
        history_rain=history_rain,
        prediction_rain=prediction_rain,
        truth_rain=truth_rain,
        prediction_radar=prediction["radar"].numpy(),
        prediction_satellite=prediction["satellite"].numpy(),
        history_times=np.array([t.isoformat() for t in past]),
        future_times=np.array([t.isoformat() for t in future]),
    )
    with weights.open("rb") as stream:
        weights_digest = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        "checkpoint": str(weights.resolve()),
        "step": args.step,
        "weights_kind": "model, not EMA",
        "weights_sha256": weights_digest,
        "config": str(config_path.resolve()),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "precision": precision,
        "dataset_window_index": index,
        "data_dirs": [str(folder) for folder in dirs],
        "history_times": [t.isoformat() for t in past],
        "future_times": [t.isoformat() for t in future],
        "times": "Naive dataset nominal Beijing times; not a claim of operational observation availability.",
        "rain_time_caveat": "Repository preprocessing queries rain over [t, t+interval]. This is a retrospective case.",
        "rollout": "Original frame-by-frame growing-history rollout; future observations never passed to model.",
        "horizon_caveat": "Leads 5-12 extend beyond the four-future-frame training horizon.",
        "geo_bounds": list(args.geo_bounds),
        "origin": "upper",
        "geo_bounds_basis": "Repository preprocessing default, not georeferencing embedded in each TIFF.",
        "boundary_source": args.boundary_source,
        "rain_field": "rain_interpolated",
        "visualization": "Raw values; nearest rendering; no smoothing, GT matching, or intensity cap.",
        "rain_value_unit": "Dataset scale; no conversion to mm/h.",
        "prediction_rain_range": [float(prediction_rain.min()), float(prediction_rain.max())],
        "negative_prediction_fraction": float((prediction_rain < 0).mean()),
    }
    (args.output_dir / "case.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    save_case_figures(
        args.output_dir,
        history_rain,
        prediction_rain,
        truth_rain,
        past,
        future,
        boundary_path,
        tuple(args.geo_bounds),
        dpi=args.dpi,
    )
    print(f"Saved raw arrays, provenance, 12-frame maps and comparison: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
