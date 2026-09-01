#!/usr/bin/env python3
import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, ListedColormap

def _find_project_root(start_path: Path) -> Path:
    for parent in [start_path, *start_path.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"Could not find project root from {start_path}")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.visualization.color import bounds as rain_bounds
from src.utils.visualization.color import precipitation_colors

SICHUAN_BOUNDARY_URL = "https://geo.datav.aliyun.com/areas_v3/bound/510000_full.json"
DEFAULT_GEO_BOUNDS = (97.3, 108.4, 26.1, 34.25)
REGION_TEST_CONFIG = {
    "run_full_test": True,
    "max_batches": 1,
    "save_visualization": True,
    "max_visual_samples": 1,
    "sample_idx": 0,
    "output_dir": "/home/rainpred/RainPrediction/vis_show/ema8",
    "geo_bounds": DEFAULT_GEO_BOUNDS,
    "boundary_source": SICHUAN_BOUNDARY_URL,
    "future_start": 0,
    "future_steps": 12,
    "write_tex_table": True,
    "tex_table_path": None,
    "probability_top_n": 0,
    "match_pred_to_gt_for_visualization": False,
}


def _to_2d_numpy(data: np.ndarray | torch.Tensor) -> np.ndarray:
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()

    arr = np.asarray(data).squeeze()
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D image data after squeeze, got shape={arr.shape}")
    return np.nan_to_num(arr.astype(np.float32), nan=0.0)


def _format_time_phase(time_value: torch.Tensor | float | None) -> str | None:
    if time_value is None:
        return None

    value = float(time_value.item() if torch.is_tensor(time_value) else time_value)
    minutes = int(round(value * 24.0 * 60.0)) % (24 * 60)
    hour = minutes // 60
    minute = minutes % 60
    return f"{hour:02d}:{minute:02d}"


def _build_future_footer(
    future_times: torch.Tensor | None,
    *,
    sample_idx: int,
    frame_idx: int,
    image_label: str | None = None,
    future_timestamps: torch.Tensor | None = None,
) -> str:
    lead_label = f"Lead time: t+{frame_idx + 1}"
    prefix = image_label if image_label is not None else lead_label
    if future_timestamps is not None:
        timestamp = float(future_timestamps[sample_idx, frame_idx].item())
        valid_time = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        return f"{prefix} | {lead_label} | Valid time: {valid_time}"

    if future_times is None:
        return prefix if image_label is not None else lead_label

    valid_time = _format_time_phase(future_times[sample_idx, frame_idx])
    if valid_time is None:
        return prefix if image_label is not None else lead_label
    return f"{prefix} | {lead_label} | Valid time: {valid_time}"


def _robust_threshold_quantile(frame: torch.Tensor, q: float, *, threshold: float) -> torch.Tensor | None:
    core = frame[frame > threshold]
    if int(core.numel()) <= 0:
        return None
    return torch.quantile(core.float(), q)


def _match_pred_rain_to_gt_for_visualization(
    pred_rain: torch.Tensor,
    gt_rain: torch.Tensor,
    *,
    quantile: float = 0.95,
    core_threshold: float = 0.2,
    max_scale: float = 1.6,
) -> torch.Tensor:
    if pred_rain.shape != gt_rain.shape:
        raise ValueError(
            "pred_rain and gt_rain must have the same shape for visualization matching, "
            f"got pred={tuple(pred_rain.shape)}, gt={tuple(gt_rain.shape)}"
        )

    matched = pred_rain.detach().float().clone()
    gt_cpu = gt_rain.detach().float().cpu()
    matched = matched.cpu()
    b, _, t, _, _ = matched.shape

    for sample_idx in range(b):
        for frame_idx in range(t):
            pred_frame = matched[sample_idx, :, frame_idx]
            gt_frame = gt_cpu[sample_idx, :, frame_idx]
            pred_q = _robust_threshold_quantile(pred_frame, quantile, threshold=core_threshold)
            gt_q = _robust_threshold_quantile(gt_frame, quantile, threshold=core_threshold)
            if pred_q is None or gt_q is None:
                continue
            if float(pred_q.item()) <= 1.0e-8:
                continue

            scale = torch.clamp(gt_q / pred_q, min=1.0 / max_scale, max=max_scale)
            core_mask = pred_frame > core_threshold
            scaled_frame = torch.clamp(pred_frame * scale, min=0.0)
            matched[sample_idx, :, frame_idx] = torch.where(core_mask, scaled_frame, pred_frame)

    return matched


def _load_geojson(boundary_source: str | Path) -> dict:
    source = str(boundary_source)
    if source.startswith(("http://", "https://")):
        with urlopen(source, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))

    return json.loads(Path(source).read_text(encoding="utf-8"))


def _iter_polygon_rings(geometry: dict) -> list[np.ndarray]:
    geom_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    rings: list[np.ndarray] = []

    if geom_type == "Polygon":
        polygon_list = [coordinates]
    elif geom_type == "MultiPolygon":
        polygon_list = coordinates
    else:
        return rings

    for polygon in polygon_list:
        if not polygon:
            continue
        exterior = np.asarray(polygon[0], dtype=np.float32)
        if exterior.ndim == 2 and exterior.shape[1] >= 2:
            rings.append(exterior[:, :2])

    return rings


def _draw_geojson_boundary(ax: Axes, geojson: dict, *, color: str, linewidth: float) -> None:
    for feature in geojson.get("features", []):
        geometry = feature.get("geometry", {})
        for ring in _iter_polygon_rings(geometry):
            ax.plot(ring[:, 0], ring[:, 1], color=color, linewidth=linewidth)


def _rain_colormap() -> tuple[ListedColormap, BoundaryNorm]:
    cmap = ListedColormap(precipitation_colors)
    norm = BoundaryNorm(rain_bounds, ncolors=cmap.N)
    return cmap, norm


def _class_colormap(num_classes: int) -> tuple[ListedColormap, BoundaryNorm]:
    colors = precipitation_colors[:num_classes]
    if len(colors) < num_classes:
        colors = colors + ["black"] * (num_classes - len(colors))
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(num_classes + 1), ncolors=cmap.N)
    return cmap, norm


def save_rain_region_image(
    data: np.ndarray | torch.Tensor,
    out_path: str | Path,
    *,
    geo_bounds: tuple[float, float, float, float] = DEFAULT_GEO_BOUNDS,
    boundary_source: str | Path = SICHUAN_BOUNDARY_URL,
    class_map: bool = False,
    num_classes: int | None = None,
    title: str | None = None,
    footer_text: str | None = None,
    boundary_color: str = "red",
    boundary_linewidth: float = 0.7,
    boundary_margin_ratio: float = 0.03,
    alpha: float = 0.95,
    dpi: int = 180,
) -> Path:
    arr = _to_2d_numpy(data)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if class_map:
        inferred_classes = int(arr.max()) + 1 if arr.size > 0 else 1
        cmap, norm = _class_colormap(num_classes or inferred_classes)
    else:
        cmap, norm = _rain_colormap()

    lon_min, lon_max, lat_min, lat_max = geo_bounds
    lon_margin = (lon_max - lon_min) * float(boundary_margin_ratio)
    lat_margin = (lat_max - lat_min) * float(boundary_margin_ratio)
    geojson = _load_geojson(boundary_source)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.imshow(
        arr,
        cmap=cmap,
        norm=norm,
        extent=[lon_min, lon_max, lat_min, lat_max],
        origin="upper",
        alpha=alpha,
    )
    _draw_geojson_boundary(ax, geojson, color=boundary_color, linewidth=boundary_linewidth)

    ax.set_xlim(lon_min - lon_margin, lon_max + lon_margin)
    ax.set_ylim(lat_min - lat_margin, lat_max + lat_margin)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    if title:
        ax.set_title(title)
    if footer_text:
        fig.subplots_adjust(bottom=0.18)
        fig.text(0.5, 0.025, footer_text, ha="center", va="bottom", fontsize=10)

    pad_inches = 0.18 if footer_text else 0.06
    fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=pad_inches)
    plt.close(fig)
    return out


def _load_demo_hydra_cfg() -> DictConfig:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    config_dir = str(PROJECT_ROOT / "src" / "config" / "ts_rain_train")
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(config_name="rain_trainer_ts_next_frame")


def _load_cls_metric_cfg() -> DictConfig:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    config_dir = str(PROJECT_ROOT / "src" / "config" / "ts_rain_test")
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(config_name="rain_test_ts_swinnet_cls")


def _save_rain_region_sequence(
    rain: torch.Tensor,
    out_dir: Path,
    *,
    name: str,
    future_start: int,
    future_end: int,
    sample_idx: int,
    geo_bounds: tuple[float, float, float, float],
    boundary_source: str | Path,
    future_times: torch.Tensor | None = None,
    future_timestamps: torch.Tensor | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rain = rain[:, :, future_start:future_end].detach().float().cpu()
    if future_times is not None:
        future_times = future_times.detach().float().cpu()
    if future_timestamps is not None:
        future_timestamps = future_timestamps.detach().double().cpu()

    for row_idx in range(int(rain.shape[2])):
        frame_idx = future_start + row_idx
        image_label = f"{name}_future({frame_idx})"
        save_rain_region_image(
            rain[sample_idx, :, row_idx],
            out_dir / f"{name}_sample{sample_idx}_future{frame_idx}.jpg",
            geo_bounds=geo_bounds,
            boundary_source=boundary_source,
            class_map=False,
            title=None,
            footer_text=_build_future_footer(
                future_times,
                sample_idx=sample_idx,
                frame_idx=frame_idx,
                image_label=image_label,
                future_timestamps=future_timestamps,
            ),
        )


def _rain_to_batch_class_map(rain: torch.Tensor, *, num_classes: int, bounds: list[float]) -> torch.Tensor:
    from src.tests.test_swinnet_cls import RainPredictionTester

    class_maps: list[torch.Tensor] = []
    for sample_idx in range(int(rain.shape[0])):
        class_map = RainPredictionTester._to_class_map(rain[sample_idx : sample_idx + 1], num_classes, bounds)
        class_maps.append(class_map)
    return torch.stack(class_maps, dim=0).unsqueeze(1).float()


def _print_region_metrics_like_swinnet_cls(
    per_frame_metric_accs: list,
    *,
    n_strong: int,
    n_total: int,
) -> None:
    metric_cfg = _load_cls_metric_cfg()
    real_bounds = metric_cfg.test.get("bounds", [0, 0.01, 0.1, 0.2, 0.5, 10])
    print(f"Using bounds strictly from YAML: {real_bounds}")

    is_cumulative = metric_cfg.test.get("cumulative", True)

    title = "Cumulative (>=)" if is_cumulative else "Strict (==)"
    print(f"\n===== Per-frame Metrics: {title} =====")
    print(f"Strong-rain samples: {n_strong} / {n_total}")

    for t, metric_acc in enumerate(per_frame_metric_accs):
        metrics = metric_acc.compute()
        print(f"\n--- Future Frame t+{t + 1} ---")
        for th_key, vals in metrics.items():
            if is_cumulative:
                try:
                    th_val_internal = float(th_key.replace(">=", "").replace("mm", ""))
                    cls_id = int(th_val_internal + 0.5)
                    if cls_id < len(real_bounds):
                        readable_key = f"Rain >= {real_bounds[cls_id]} (Class {cls_id})"
                    else:
                        readable_key = f"Class >= {cls_id}"
                except Exception:
                    readable_key = th_key
            else:
                cls_id = int(th_key.split("_")[-1])
                if cls_id < len(real_bounds) - 1:
                    lower = real_bounds[cls_id]
                    upper = real_bounds[cls_id + 1]
                    readable_key = f"Rain in [{lower}, {upper}) (Strict Class {cls_id})"
                else:
                    readable_key = f"Class == {cls_id}"

            msg = " ".join([f"{k}={v.item():.4f}" for k, v in vals.items()])
            print(f"{readable_key}: {msg}")


def _make_metric_accumulator(*, device: torch.device, metric_cfg: DictConfig):
    from src.tests.test_swinnet_cls import StrictClassMetricsAccumulator
    from src.utils.metrics.compute_metrics_new import RainGlobalMetricsAccumulator

    real_bounds = metric_cfg.test.get("bounds", [0, 0.01, 0.1, 0.2, 0.5, 10])
    num_classes = len(real_bounds) - 1
    is_cumulative = metric_cfg.test.get("cumulative", True)
    tol_px = metric_cfg.get("val", {}).get("tolerance_px", 0)
    cumulative_thresholds = [-1.0] + [i + 0.5 for i in range(num_classes - 1)]

    if is_cumulative:
        return RainGlobalMetricsAccumulator(
            bounds=cumulative_thresholds,
            device=device,
            tolerance_px=tol_px,
        )
    return StrictClassMetricsAccumulator(num_classes, device)


def _print_metric_mode(metric_cfg: DictConfig) -> None:
    real_bounds = metric_cfg.test.get("bounds", [0, 0.01, 0.1, 0.2, 0.5, 10])
    is_cumulative = metric_cfg.test.get("cumulative", True)
    tol_px = metric_cfg.get("val", {}).get("tolerance_px", 0)
    mode_name = (
        f"Cumulative metrics (>=) | tolerance={tol_px}px"
        if is_cumulative
        else "Strict metrics (==)"
    )
    print(f"Using bounds strictly from YAML: {real_bounds}")
    print(f"Metric mode: {mode_name}")


def _update_class_metrics(
    per_frame_metric_accs: list | None,
    pred_rain: torch.Tensor,
    gt_rain: torch.Tensor,
    *,
    metric_cfg: DictConfig,
) -> list:
    real_bounds = metric_cfg.test.get("bounds", [0, 0.01, 0.1, 0.2, 0.5, 10])
    num_classes = len(real_bounds) - 1
    device = pred_rain.device
    pred_cls_idx = _rain_to_batch_class_map(pred_rain, num_classes=num_classes, bounds=real_bounds).to(device)
    gt_cls_idx = _rain_to_batch_class_map(gt_rain, num_classes=num_classes, bounds=real_bounds).to(device)

    t_future = int(pred_cls_idx.shape[2])
    if per_frame_metric_accs is None:
        per_frame_metric_accs = [_make_metric_accumulator(device=device, metric_cfg=metric_cfg) for _ in range(t_future)]

    for t in range(t_future):
        pred_t = pred_cls_idx[:, :, t : t + 1]
        gt_t = gt_cls_idx[:, :, t : t + 1]
        per_frame_metric_accs[t].update(pred_t, gt_t)
    return per_frame_metric_accs


def _regression_stats(pred: torch.Tensor, gt: torch.Tensor) -> dict[str, torch.Tensor]:
    diff = pred - gt
    pred_flat = pred.reshape(-1)
    gt_flat = gt.reshape(-1)
    pred_centered = pred_flat - pred_flat.mean()
    gt_centered = gt_flat - gt_flat.mean()
    corr_denom = torch.sqrt(pred_centered.square().sum() * gt_centered.square().sum())
    corr = torch.tensor(0.0, device=pred.device)
    if float(corr_denom.item()) > 1.0e-8:
        corr = (pred_centered * gt_centered).sum() / corr_denom

    return {
        "MAE": diff.abs().mean(),
        "MSE": diff.square().mean(),
        "RMSE": torch.sqrt(diff.square().mean()),
        "Bias": diff.mean(),
        "Corr": corr,
    }


def _update_regression_sums(
    per_frame_sums: list[dict[str, torch.Tensor]] | None,
    pred_rain: torch.Tensor,
    gt_rain: torch.Tensor,
) -> list[dict[str, torch.Tensor]]:
    t_future = int(pred_rain.shape[2])
    if per_frame_sums is None:
        per_frame_sums = []
        for _ in range(t_future):
            per_frame_sums.append(
                {
                    "abs_error": torch.tensor(0.0, device=pred_rain.device),
                    "sq_error": torch.tensor(0.0, device=pred_rain.device),
                    "error": torch.tensor(0.0, device=pred_rain.device),
                    "pred_sum": torch.tensor(0.0, device=pred_rain.device),
                    "gt_sum": torch.tensor(0.0, device=pred_rain.device),
                    "pred_sq_sum": torch.tensor(0.0, device=pred_rain.device),
                    "gt_sq_sum": torch.tensor(0.0, device=pred_rain.device),
                    "pred_gt_sum": torch.tensor(0.0, device=pred_rain.device),
                    "count": torch.tensor(0.0, device=pred_rain.device),
                }
            )

    for t in range(t_future):
        pred_t = pred_rain[:, :, t]
        gt_t = gt_rain[:, :, t]
        diff = pred_t - gt_t
        stats = per_frame_sums[t]
        stats["abs_error"] += diff.abs().sum()
        stats["sq_error"] += diff.square().sum()
        stats["error"] += diff.sum()
        stats["pred_sum"] += pred_t.sum()
        stats["gt_sum"] += gt_t.sum()
        stats["pred_sq_sum"] += pred_t.square().sum()
        stats["gt_sq_sum"] += gt_t.square().sum()
        stats["pred_gt_sum"] += (pred_t * gt_t).sum()
        stats["count"] += torch.tensor(float(pred_t.numel()), device=pred_rain.device)
    return per_frame_sums


def _print_regression_metrics(per_frame_sums: list[dict[str, torch.Tensor]]) -> None:
    print("\n===== Per-frame Regression Metrics =====")
    for t, stats in enumerate(per_frame_sums):
        count = torch.clamp(stats["count"], min=1.0)
        mae = stats["abs_error"] / count
        mse = stats["sq_error"] / count
        rmse = torch.sqrt(mse)
        bias = stats["error"] / count
        pred_mean = stats["pred_sum"] / count
        gt_mean = stats["gt_sum"] / count
        cov = stats["pred_gt_sum"] - count * pred_mean * gt_mean
        pred_var = stats["pred_sq_sum"] - count * pred_mean.square()
        gt_var = stats["gt_sq_sum"] - count * gt_mean.square()
        denom = torch.sqrt(torch.clamp(pred_var * gt_var, min=0.0))
        corr = torch.tensor(0.0, device=count.device)
        if float(denom.item()) > 1.0e-8:
            corr = cov / denom
        print(f"\n--- Future Frame t+{t + 1} ---")
        print(
            f"MAE={mae.item():.4f} "
            f"MSE={mse.item():.4f} "
            f"RMSE={rmse.item():.4f} "
            f"Bias={bias.item():.4f} "
            f"Corr={corr.item():.4f}"
        )


def run_inference_demo_region_plot(
    *,
    geo_bounds: tuple[float, float, float, float] = DEFAULT_GEO_BOUNDS,
    boundary_source: str | Path = SICHUAN_BOUNDARY_URL,
    output_dir: str | Path | None = None,
    sample_idx: int | None = None,
    run_full_test: bool = True,
    max_batches: int = 0,
    save_visualization: bool = True,
    max_visual_samples: int = 1,
    match_pred_to_gt_for_visualization: bool = True,
) -> Path:
    from src.tests.inference_demo import (
        DEMO_CONFIG,
        _analyze_input_reasonability,
        _build_demo_cfg_for_trainer,
        _load_ema_to_model,
        _prepare_analysis_modalities,
    )
    from src.trainer.rain_trainer_ts_next_frame import RainTSNextFrameTrainer

    demo_cfg = copy.deepcopy(DEMO_CONFIG)
    cfg = _load_demo_hydra_cfg()
    _build_demo_cfg_for_trainer(cfg, demo_cfg)

    trainer = RainTSNextFrameTrainer(cfg)
    _analyze_input_reasonability(
        trainer=trainer,
        context_frames=int(demo_cfg["context_frames"]),
        predict_frames=int(demo_cfg["predict_frames"]),
    )

    _load_ema_to_model(trainer, str(demo_cfg["ema_path"]))
    trainer.model.eval()
    trainer.global_step = 0

    vis_start = int(demo_cfg["vis_future_start"])
    vis_end = int(demo_cfg["vis_future_end"])
    target_sample_idx = int(demo_cfg.get("diff_sample_index", 0) if sample_idx is None else sample_idx)
    metric_cfg = _load_cls_metric_cfg()
    _print_metric_mode(metric_cfg)

    if output_dir is None:
        region_dir = (
            trainer.proj_dir
            / "val_viz"
            / f"step_{trainer.global_step:08d}"
            / f"inference_future_{vis_start}_{vis_end}_region"
        )
    else:
        region_dir = Path(output_dir)

    n_total = 0
    n_strong = 0
    saved_visual_samples = 0
    per_frame_metric_accs = None
    per_frame_regression_sums = None
    strong_threshold = metric_cfg.test.get("strong_threshold", 0)
    max_batches = int(max_batches)
    max_visual_samples = int(max_visual_samples)

    with torch.no_grad():
        data_iter = iter(trainer.val_dataloader)
        for batch_idx, batch in enumerate(data_iter):
            if max_batches > 0 and batch_idx >= max_batches:
                break
            if not run_full_test and batch_idx >= 1:
                break

            context, target, context_time, target_time = trainer._prepare_val_inference_batch(batch)
            target_timestamp = batch.get("time_future_timestamp")
            total_future_frames = int(target["rain"].shape[2])
            if vis_start < 0 or vis_start >= total_future_frames:
                raise ValueError(f"vis_future_start out of range: {vis_start}, available [0, {total_future_frames - 1}]")
            if vis_end <= vis_start or vis_end > total_future_frames:
                raise ValueError(f"vis_future_end out of range: {vis_end}, should satisfy {vis_start} < end <= {total_future_frames}")

            with trainer.accelerator.autocast():
                pred_target = trainer._rollout_predict(
                    context=context,
                    total_future_frames=total_future_frames,
                    context_time=context_time,
                    future_time=target_time,
                    future_modalities=target,
                )

            _, pred_modalities, target_modalities = _prepare_analysis_modalities(
                trainer=trainer,
                context=context,
                pred_target=pred_target,
                target=target,
            )

            pred_rain = pred_modalities["rain"][:, :, vis_start:vis_end].to(trainer.device)
            gt_rain = target_modalities["rain"][:, :, vis_start:vis_end].to(trainer.device)
            n_total += int(gt_rain.size(0))
            has_strong = (gt_rain >= strong_threshold).flatten(1).any(dim=1)
            idx = has_strong.nonzero(as_tuple=False).squeeze(-1)
            if idx.numel() == 0:
                continue

            pred_rain_strong = pred_rain[idx]
            gt_rain_strong = gt_rain[idx]
            n_strong += int(idx.numel())
            per_frame_regression_sums = _update_regression_sums(
                per_frame_regression_sums,
                pred_rain_strong,
                gt_rain_strong,
            )
            per_frame_metric_accs = _update_class_metrics(
                per_frame_metric_accs,
                pred_rain_strong,
                gt_rain_strong,
                metric_cfg=metric_cfg,
            )

            if save_visualization and saved_visual_samples < max_visual_samples:
                pred_rain_for_visualization = pred_modalities["rain"]
                if match_pred_to_gt_for_visualization:
                    pred_rain_for_visualization = _match_pred_rain_to_gt_for_visualization(
                        pred_modalities["rain"],
                        target_modalities["rain"],
                    )

                for local_i in range(int(pred_rain.shape[0])):
                    if saved_visual_samples >= max_visual_samples:
                        break
                    if not bool(has_strong[local_i].item()):
                        continue
                    _save_rain_region_sequence(
                        rain=pred_rain_for_visualization,
                        out_dir=region_dir,
                        name="pred_rain",
                        future_start=vis_start,
                        future_end=vis_end,
                        sample_idx=local_i,
                        geo_bounds=geo_bounds,
                        boundary_source=boundary_source,
                        future_times=target_time,
                        future_timestamps=target_timestamp,
                    )
                    _save_rain_region_sequence(
                        rain=target_modalities["rain"],
                        out_dir=region_dir,
                        name="gt_rain",
                        future_start=vis_start,
                        future_end=vis_end,
                        sample_idx=local_i,
                        geo_bounds=geo_bounds,
                        boundary_source=boundary_source,
                        future_times=target_time,
                        future_timestamps=target_timestamp,
                    )
                    saved_visual_samples += 1

    if n_strong == 0:
        print("\nNo strong rain samples found in test set!")
        return region_dir

    if per_frame_regression_sums is not None:
        _print_regression_metrics(per_frame_regression_sums)
    if per_frame_metric_accs is not None:
        _print_region_metrics_like_swinnet_cls(
            per_frame_metric_accs,
            n_strong=n_strong,
            n_total=n_total,
        )
    if save_visualization:
        print(f"Saved rain region prediction and gt images: {region_dir}")
        print(f"Visualization saved samples: {saved_visual_samples}")
    return region_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference_demo and plot predicted rain with Sichuan region boundary.")
    parser.add_argument("--input-npy", type=str, default="")
    parser.add_argument("--out-path", type=str, default="")
    parser.add_argument("--geo-bounds", type=float, nargs=4, default=REGION_TEST_CONFIG["geo_bounds"])
    parser.add_argument("--boundary-source", type=str, default=REGION_TEST_CONFIG["boundary_source"])
    parser.add_argument("--class-map", action="store_true")
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--footer-text", type=str, default=None)
    parser.add_argument("--from-inference-demo", action="store_true")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--sample-idx", type=int, default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--no-vis", action="store_true")
    parser.add_argument("--max-visual-samples", type=int, default=None)
    parser.add_argument("--no-match-pred-visual", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    geo_bounds = tuple(float(v) for v in args.geo_bounds)
    if args.from_inference_demo or not args.input_npy:
        output_dir = args.output_dir or REGION_TEST_CONFIG["output_dir"]
        sample_idx = REGION_TEST_CONFIG["sample_idx"] if args.sample_idx is None else args.sample_idx
        max_batches = REGION_TEST_CONFIG["max_batches"] if args.max_batches is None else args.max_batches
        max_visual_samples = (
            REGION_TEST_CONFIG["max_visual_samples"]
            if args.max_visual_samples is None
            else args.max_visual_samples
        )
        run_inference_demo_region_plot(
            geo_bounds=geo_bounds,
            boundary_source=args.boundary_source,
            output_dir=output_dir,
            sample_idx=sample_idx,
            run_full_test=bool(REGION_TEST_CONFIG["run_full_test"]) and not bool(args.quick),
            max_batches=int(max_batches),
            save_visualization=bool(REGION_TEST_CONFIG["save_visualization"]) and not bool(args.no_vis),
            max_visual_samples=int(max_visual_samples),
            match_pred_to_gt_for_visualization=bool(REGION_TEST_CONFIG["match_pred_to_gt_for_visualization"])
            and not bool(args.no_match_pred_visual),
        )
        return

    if not args.out_path:
        raise ValueError("--out-path is required when --input-npy is set.")

    data = np.load(args.input_npy)
    save_rain_region_image(
        data,
        args.out_path,
        geo_bounds=geo_bounds,
        boundary_source=args.boundary_source,
        class_map=bool(args.class_map),
        num_classes=args.num_classes,
        title=args.title,
        footer_text=args.footer_text,
    )


if __name__ == "__main__":
    main()
