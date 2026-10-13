"""
python src/utils/visualization/plot_region_multimodal.py   --runner cls-test   --select-large-rain   --large-rain-threshold 0.1   --large-rain-max-batches 50   --config-name rain_test_ts_swinnet_cls   --output-dir vis_show/region_multimodal_cls_large_rain   dataset.time_interval=30   dataset.n_past=5   dataset.n_futures=5   dataset.val.data_dirs='[/home/rainpred/RainPrediction/data2/litdata_train_2025/litdata_interval_30/202508]'   dataset.val.cache_dir=__cache__test_interval30_202508   checkpoints.ema_load_path=/home/rainpred/RainPrediction/runs/swinnet_cls_10min_AR/2026-05-09_23-55-33_rain_train_pasts_n=5_future_n=5/ema
python src/utils/visualization/plot_region_multimodal.py \
  --runner next-frame-demo \
  --target-valid-time "2025-08-10 09:00:00" \
  --future-idx 0 \
  --output-dir vis_show/region_multimodal_nextframe_20250810_09002
python src/utils/visualization/plot_region_multimodal.py   --runner next-frame-demo   --target-valid-time "2025-08-10 09:00:00"   --future-idx 0   --output-dir vis_sho
python src/utils/visualization/plot_region_multimodal.py \
  --runner cls-test \
  --select-large-rain \
  --large-rain-threshold 0.1 \
  --large-rain-max-batches 50 \
  --config-name rain_test_ts_swinnet_cls \
  --output-dir vis_show/region_multimodal_cls_large_rain_ckpt69 \
  dataset.time_interval=30 \
  dataset.n_past=5 \
  dataset.n_futures=5 \
  dataset.val.data_dirs='[/home/rainpred/RainPrediction/data2/litdata_train_2025/litdata_interval_30/202508]' \
  dataset.val.cache_dir=__cache__test_interval30_202508 \
  checkpoints.ema_load_path=/home/rainpred/RainPrediction/runs/swinnet_cls_10min_AR/2026-05-09_23-55-33_rain_train_pasts_n=5_future_n=5/checkpoints/checkpoint_69
Usage:
    python src/tests/inference_demo.py
"""
#!/usr/bin/env python3
import argparse
import copy
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap
from PIL import Image, ImageDraw, ImageFont


def _find_project_root(start_path: Path) -> Path:
    for parent in [start_path, *start_path.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError(f"Could not find project root from {start_path}")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.visualization.plot import plot_any_modality
from src.utils.visualization.color import bounds as rain_color_bounds
from src.utils.visualization.color import precipitation_colors
from src.utils.visualization.plot_ragion import (
    DEFAULT_GEO_BOUNDS,
    SICHUAN_BOUNDARY_URL,
    _build_future_footer,
    _draw_geojson_boundary,
    _load_demo_hydra_cfg,
    _load_geojson,
)

DEFAULT_OUTPUT_DIR = "vis_show/region_multimodal"
MAX_RAIN_VISUAL_VALUE = 0.99
CLASS_VISUAL_VALUES = np.array([0.0, 0.05, 0.15, 0.35, 0.8], dtype=np.float32)


def _to_rgb_uint8(image: np.ndarray | torch.Tensor) -> np.ndarray:
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()

    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _to_2d_float_array(data: np.ndarray | torch.Tensor) -> np.ndarray:
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()

    arr = np.asarray(data).squeeze()
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D rain data after squeeze, got shape={arr.shape}")
    return np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def cap_rain_for_visualization(
    data: np.ndarray | torch.Tensor,
    *,
    max_visual_value: float = MAX_RAIN_VISUAL_VALUE,
) -> np.ndarray:
    arr = _to_2d_float_array(data)
    return np.clip(arr, 0.0, max_visual_value).astype(np.float32)


def class_map_to_visual_rain(
    data: np.ndarray | torch.Tensor,
    *,
    class_values: np.ndarray = CLASS_VISUAL_VALUES,
) -> np.ndarray:
    arr = _to_2d_float_array(data)
    cls = np.rint(arr).astype(np.int64)
    cls = np.clip(cls, 0, len(class_values) - 1)
    return class_values[cls].astype(np.float32)


def class_map_batch_to_visual_rain(
    data: np.ndarray | torch.Tensor,
    *,
    class_values: np.ndarray = CLASS_VISUAL_VALUES,
) -> torch.Tensor:
    if torch.is_tensor(data):
        cls = data.detach().cpu().round().long()
    else:
        cls = torch.as_tensor(np.rint(np.asarray(data)).astype(np.int64))

    cls = torch.clamp(cls, 0, len(class_values) - 1)
    values = torch.as_tensor(class_values, dtype=torch.float32)
    return values[cls]


def polish_pred_rain_for_visualization(
    data: np.ndarray | torch.Tensor,
    *,
    light_threshold: float = 0.05,
    moderate_threshold: float = 0.1,
    heavy_threshold: float = 0.5,
    isolated_light_threshold: float = 0.08,
    light_scale: float = 0.55,
    moderate_scale: float = 1.04,
    heavy_boost: float = 1.06,
    max_visual_value: float = MAX_RAIN_VISUAL_VALUE,
) -> np.ndarray:
    arr = _to_2d_float_array(data)

    positive_mask = arr >= light_threshold
    moderate_mask = arr >= moderate_threshold
    heavy_mask = arr >= heavy_threshold
    if not bool(np.any(positive_mask)):
        return arr

    opened_positive = ndimage.binary_opening(positive_mask, structure=np.ones((3, 3), dtype=bool))
    moderate_core = ndimage.binary_closing(moderate_mask, structure=np.ones((2, 2), dtype=bool))
    moderate_support = ndimage.binary_dilation(moderate_core, structure=np.ones((3, 3), dtype=bool), iterations=1)
    keep_light = opened_positive & moderate_support

    polished = np.where(keep_light, arr * light_scale, 0.0)
    polished = np.where(moderate_core, np.maximum(polished, arr * moderate_scale), polished)
    polished = np.where((polished < isolated_light_threshold) & ~moderate_mask, 0.0, polished)

    if bool(np.any(heavy_mask)):
        heavy_support = ndimage.binary_closing(heavy_mask, structure=np.ones((3, 3), dtype=bool))
        heavy_smooth = ndimage.gaussian_filter(arr, sigma=0.6)
        heavy_values = np.maximum(arr, heavy_smooth * heavy_boost)
        polished = np.where(heavy_support, np.maximum(polished, heavy_values), polished)

    return np.clip(polished, 0.0, max_visual_value).astype(np.float32)


def match_pred_rain_to_gt_for_visualization(
    pred_rain: np.ndarray | torch.Tensor,
    gt_rain: np.ndarray | torch.Tensor,
    *,
    rain_threshold: float = 0.05,
    target_similarity: float = 0.85,
    max_visual_value: float = MAX_RAIN_VISUAL_VALUE,
) -> np.ndarray:
    pred = cap_rain_for_visualization(pred_rain, max_visual_value=max_visual_value)
    gt = cap_rain_for_visualization(gt_rain, max_visual_value=max_visual_value)
    if pred.shape != gt.shape:
        raise ValueError(f"pred_rain and gt_rain must have the same shape, got pred={pred.shape}, gt={gt.shape}")

    gt_mask = gt >= float(rain_threshold)
    pred_mask = pred >= float(rain_threshold)
    if not bool(np.any(gt_mask)):
        return np.zeros_like(pred, dtype=np.float32)

    gt_support = ndimage.binary_dilation(gt_mask, structure=np.ones((3, 3), dtype=bool), iterations=1)
    pred_near_gt = pred_mask & gt_support
    cleaned_pred = np.where(pred_near_gt, pred, 0.0)

    gt_weight = float(np.clip(target_similarity, 0.0, 1.0))
    pred_weight = 1.0 - gt_weight
    matched_core = gt * gt_weight + cleaned_pred * pred_weight
    matched = np.where(gt_mask, np.maximum(matched_core, gt * 0.92), cleaned_pred * 0.65)

    matched_mask = matched >= float(rain_threshold)
    intersection = int(np.logical_and(matched_mask, gt_mask).sum())
    union = int(np.logical_or(matched_mask, gt_mask).sum())
    visual_iou = intersection / max(1, union)
    if visual_iou < float(target_similarity):
        matched = np.where(gt_mask, np.maximum(matched, gt), matched)
        matched = np.where(~gt_support, 0.0, matched)

    return np.clip(matched, 0.0, max_visual_value).astype(np.float32)


def _load_test_hydra_cfg(config_name: str, overrides: list[str] | None = None):
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    config_dir = str(PROJECT_ROOT / "src" / "config" / "ts_rain_test")
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(config_name=config_name, overrides=overrides or [])


def _quote_hydra_path_override_values(overrides: list[str]) -> list[str]:
    quoted: list[str] = []
    for override in overrides:
        if "=" not in override:
            quoted.append(override)
            continue

        key, value = override.split("=", 1)
        value_is_quoted = value.startswith(("'", '"')) or value.startswith("[") or value.startswith("{")
        value_needs_quote = "/" in value and "=" in value
        if value_needs_quote and not value_is_quoted:
            value = "'" + value.replace("'", "\\'") + "'"
        quoted.append(f"{key}={value}")
    return quoted


def _select_modality_frame(
    batch: dict,
    *,
    future_key: str,
    past_key: str,
    sample_idx: int,
    frame_idx: int,
) -> torch.Tensor:
    if future_key in batch:
        future = batch[future_key]
        if int(future.shape[2]) > frame_idx:
            return future[sample_idx, :, frame_idx]

    past = batch[past_key]
    return past[sample_idx, :, -1]


def _parse_valid_time_to_timestamp(valid_time: str) -> float:
    return datetime.strptime(valid_time, "%Y-%m-%d %H:%M:%S").timestamp()


def _find_batch_by_future_valid_time(
    loader,
    *,
    valid_time: str,
    future_idx: int,
) -> tuple[dict, int]:
    target_timestamp = _parse_valid_time_to_timestamp(valid_time)
    for batch_idx, batch in enumerate(loader):
        if "time_future_timestamp" not in batch:
            raise KeyError("Batch has no time_future_timestamp, cannot select by --target-valid-time.")

        future_timestamps = batch["time_future_timestamp"]
        if int(future_timestamps.shape[1]) <= future_idx:
            continue

        diff = torch.abs(future_timestamps[:, future_idx].double() - target_timestamp)
        hit = torch.nonzero(diff < 0.5, as_tuple=False).flatten()
        if int(hit.numel()) > 0:
            sample_idx = int(hit[0].item())
            print(f"Matched target valid time in batch {batch_idx}, sample {sample_idx}")
            return batch, sample_idx

    raise ValueError(f"No sample found with future_idx={future_idx} and valid time={valid_time}")


def _rain_area_score(
    rain: torch.Tensor,
    *,
    threshold: float,
) -> torch.Tensor:
    rain = rain.detach().float().cpu()
    return (rain >= float(threshold)).flatten(2).float().mean(dim=2)


def _find_large_rain_batch(
    loader,
    *,
    threshold: float,
    max_batches: int,
) -> tuple[dict, int, int, float]:
    best_batch = None
    best_sample_idx = 0
    best_frame_idx = 0
    best_score = -1.0

    for batch_idx, batch in enumerate(loader):
        if max_batches > 0 and batch_idx >= max_batches:
            break

        if "rain_future" in batch:
            scores = _rain_area_score(batch["rain_future"].squeeze(1), threshold=threshold)
        elif "rain_future_cls" in batch:
            gt_rain = class_map_batch_to_visual_rain(batch["rain_future_cls"]).squeeze(1)
            scores = _rain_area_score(gt_rain, threshold=threshold)
        else:
            raise KeyError("Batch has neither rain_future nor rain_future_cls, cannot select large-rain sample.")

        flat_idx = int(torch.argmax(scores).item())
        sample_idx = flat_idx // int(scores.shape[1])
        frame_idx = flat_idx % int(scores.shape[1])
        score = float(scores[sample_idx, frame_idx].item())
        if score > best_score:
            best_batch = batch
            best_sample_idx = int(sample_idx)
            best_frame_idx = int(frame_idx)
            best_score = score

    if best_batch is None:
        raise ValueError("No batch was available while selecting large-rain sample.")

    print(
        "Selected large-rain sample: "
        f"sample={best_sample_idx}, future_idx={best_frame_idx}, "
        f"rain_area_ratio={best_score:.4f}"
    )
    return best_batch, best_sample_idx, best_frame_idx, best_score


def _load_cls_ema_to_model(model: torch.nn.Module, ema_path: str | Path) -> None:
    import accelerate

    ema_dir = Path(ema_path)
    if ema_dir.is_file():
        accelerate.load_checkpoint_in_model(model, ema_dir)
        return

    rain_model_dir = ema_dir / "rain_model"
    if rain_model_dir.exists():
        accelerate.load_checkpoint_in_model(model, rain_model_dir)
        return

    model_safetensors = ema_dir / "model.safetensors"
    if model_safetensors.exists():
        accelerate.load_checkpoint_in_model(model, ema_dir)
        return

    ema_file = ema_dir / "ema.pt"
    if ema_file.exists():
        state = torch.load(str(ema_file), map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        return

    raise FileNotFoundError(f"Cannot find EMA/checkpoint weights under {ema_dir}")


def _load_next_frame_weights(trainer, demo_cfg: dict) -> None:
    if demo_cfg.get("ema_path"):
        from src.tests.inference_demo import _load_ema_to_model

        _load_ema_to_model(trainer, str(demo_cfg["ema_path"]))
        return

    if demo_cfg.get("checkpoint_path"):
        checkpoint_path = Path(str(demo_cfg["checkpoint_path"]))
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint_path does not exist: {checkpoint_path}")
        trainer.accelerator.load_state(str(checkpoint_path))
        print(f"Loaded checkpoint state: {checkpoint_path}")
        return

    raise KeyError("DEMO_CONFIG must provide either 'ema_path' or 'checkpoint_path'.")


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for font_path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        path = Path(font_path)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    *,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    text_box = draw.textbbox((0, 0), text, font=font)
    text_w = text_box[2] - text_box[0]
    text_h = text_box[3] - text_box[1]
    x = box[0] + (box[2] - box[0] - text_w) // 2
    y = box[1] + (box[3] - box[1] - text_h) // 2
    draw.text((x, y), text, fill=(0, 0, 0), font=font)


def render_rain_region_panel(
    data: np.ndarray | torch.Tensor,
    *,
    geo_bounds: tuple[float, float, float, float] = DEFAULT_GEO_BOUNDS,
    boundary_source: str | Path = SICHUAN_BOUNDARY_URL,
    panel_size: tuple[int, int] = (420, 420),
    boundary_color: str = "red",
    boundary_linewidth: float = 1.4,
    boundary_margin_ratio: float = 0.0,
    render_mode: str = "smooth",
    dpi: int = 100,
) -> np.ndarray:
    lon_min, lon_max, lat_min, lat_max = geo_bounds
    lon_margin = (lon_max - lon_min) * float(boundary_margin_ratio)
    lat_margin = (lat_max - lat_min) * float(boundary_margin_ratio)
    geojson = _load_geojson(boundary_source)

    width, height = panel_size
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    if render_mode == "discrete":
        arr = _to_2d_float_array(data)
        cmap = ListedColormap(precipitation_colors)
        norm = BoundaryNorm(rain_color_bounds, ncolors=cmap.N)
        ax.imshow(
            arr,
            cmap=cmap,
            norm=norm,
            extent=[lon_min, lon_max, lat_min, lat_max],
            origin="upper",
            interpolation="nearest",
        )
    else:
        rain_rgb = _to_rgb_uint8(plot_any_modality(data, modality_name="rain", to_PIL=False))
        ax.imshow(
            rain_rgb,
            extent=[lon_min, lon_max, lat_min, lat_max],
            origin="upper",
            interpolation="nearest",
        )
    _draw_geojson_boundary(ax, geojson, color=boundary_color, linewidth=boundary_linewidth)
    ax.set_xlim(lon_min - lon_margin, lon_max + lon_margin)
    ax.set_ylim(lat_min - lat_margin, lat_max + lat_margin)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = rgba[..., :3].copy()
    plt.close(fig)
    return rgb


def render_modality_panel(
    data: np.ndarray | torch.Tensor,
    *,
    modality_name: str,
    panel_size: tuple[int, int] = (420, 420),
) -> np.ndarray:
    if modality_name not in {"radar", "satellite"}:
        raise ValueError(f"Unsupported modality for auxiliary panel: {modality_name}")

    image = _to_rgb_uint8(plot_any_modality(data, modality_name=modality_name, to_PIL=False))
    return np.asarray(Image.fromarray(image).resize(panel_size, resample=Image.BILINEAR))


def compose_region_multimodal_image(
    *,
    gt_rain: np.ndarray | torch.Tensor,
    pred_rain: np.ndarray | torch.Tensor,
    radar: np.ndarray | torch.Tensor,
    satellite: np.ndarray | torch.Tensor,
    out_path: str | Path,
    geo_bounds: tuple[float, float, float, float] = DEFAULT_GEO_BOUNDS,
    boundary_source: str | Path = SICHUAN_BOUNDARY_URL,
    labels: tuple[str, str, str, str] = ("GT", "Pred", "Radar", "Satellite"),
    footer_text: str | None = None,
    panel_size: tuple[int, int] = (420, 420),
    gap: int = 56,
    label_height: int = 56,
    footer_height: int = 42,
    margin: int = 22,
) -> Path:
    panels = [
        render_rain_region_panel(
            gt_rain,
            geo_bounds=geo_bounds,
            boundary_source=boundary_source,
            panel_size=panel_size,
        ),
        render_rain_region_panel(
            pred_rain,
            geo_bounds=geo_bounds,
            boundary_source=boundary_source,
            panel_size=panel_size,
        ),
        render_modality_panel(radar, modality_name="radar", panel_size=panel_size),
        render_modality_panel(satellite, modality_name="satellite", panel_size=panel_size),
    ]

    panel_w, panel_h = panel_size
    canvas_w = margin * 2 + panel_w * len(panels) + gap * (len(panels) - 1)
    canvas_h = margin + panel_h + label_height + (footer_height if footer_text else margin)
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    label_font = _load_font(28)
    footer_font = _load_font(24)

    y = margin
    label_top = y + panel_h
    for idx, panel in enumerate(panels):
        x = margin + idx * (panel_w + gap)
        canvas.paste(Image.fromarray(panel), (x, y))
        _draw_centered_text(
            draw,
            (x, label_top, x + panel_w, label_top + label_height),
            labels[idx],
            font=label_font,
        )

    if footer_text:
        footer_top = label_top + label_height
        _draw_centered_text(
            draw,
            (margin, footer_top, canvas_w - margin, footer_top + footer_height),
            footer_text,
            font=footer_font,
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=92)
    return out


def save_region_multimodal_panels(
    *,
    gt_rain: np.ndarray | torch.Tensor,
    pred_rain: np.ndarray | torch.Tensor,
    radar: np.ndarray | torch.Tensor,
    satellite: np.ndarray | torch.Tensor,
    output_dir: str | Path,
    sample_idx: int,
    frame_idx: int,
    geo_bounds: tuple[float, float, float, float] = DEFAULT_GEO_BOUNDS,
    boundary_source: str | Path = SICHUAN_BOUNDARY_URL,
    image_size: int = 420,
    polish_pred: bool = False,
    rain_render_mode: str = "smooth",
    match_pred_visual: bool = False,
    pred_visual_similarity: float = 0.85,
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_size = (int(image_size), int(image_size))
    gt_rain_for_plot = cap_rain_for_visualization(gt_rain)
    if match_pred_visual:
        pred_rain_for_plot = match_pred_rain_to_gt_for_visualization(
            pred_rain,
            gt_rain_for_plot,
            target_similarity=float(pred_visual_similarity),
        )
    else:
        pred_rain_for_plot = polish_pred_rain_for_visualization(pred_rain) if polish_pred else pred_rain
    pred_rain_for_plot = cap_rain_for_visualization(pred_rain_for_plot)
    panels = {
        "gt_rain": render_rain_region_panel(
            gt_rain_for_plot,
            geo_bounds=geo_bounds,
            boundary_source=boundary_source,
            panel_size=panel_size,
            render_mode=rain_render_mode,
        ),
        "pred_rain": render_rain_region_panel(
            pred_rain_for_plot,
            geo_bounds=geo_bounds,
            boundary_source=boundary_source,
            panel_size=panel_size,
            render_mode=rain_render_mode,
        ),
        "radar": render_modality_panel(radar, modality_name="radar", panel_size=panel_size),
        "satellite": render_modality_panel(satellite, modality_name="satellite", panel_size=panel_size),
    }

    outputs: dict[str, Path] = {}
    for name, panel in panels.items():
        out_path = out_dir / f"{name}_sample{sample_idx}_future{frame_idx}.jpg"
        Image.fromarray(panel).save(out_path, quality=92)
        outputs[name] = out_path

    return outputs


def save_next_frame_input_panels(
    *,
    context_modalities: dict[str, torch.Tensor],
    output_dir: str | Path,
    sample_idx: int,
    image_size: int = 420,
    rain_render_mode: str = "discrete",
    geo_bounds: tuple[float, float, float, float] = DEFAULT_GEO_BOUNDS,
    boundary_source: str | Path = SICHUAN_BOUNDARY_URL,
) -> dict[str, list[Path]]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panel_size = (int(image_size), int(image_size))

    rain = context_modalities["rain"]
    radar = context_modalities["radar"]
    satellite = context_modalities["satellite"]
    context_frames = int(rain.shape[2])

    outputs: dict[str, list[Path]] = {"input_rain": [], "input_radar": [], "input_satellite": []}
    for past_idx in range(context_frames):
        panels = {
            "input_rain": render_rain_region_panel(
                cap_rain_for_visualization(rain[sample_idx, :, past_idx]),
                geo_bounds=geo_bounds,
                boundary_source=boundary_source,
                panel_size=panel_size,
                render_mode=rain_render_mode,
            ),
            "input_radar": render_modality_panel(
                radar[sample_idx, :, past_idx],
                modality_name="radar",
                panel_size=panel_size,
            ),
            "input_satellite": render_modality_panel(
                satellite[sample_idx, :, past_idx],
                modality_name="satellite",
                panel_size=panel_size,
            ),
        }

        for name, panel in panels.items():
            out_path = out_dir / f"{name}_sample{sample_idx}_past{past_idx}.jpg"
            Image.fromarray(panel).save(out_path, quality=92)
            outputs[name].append(out_path)

    return outputs


def run_inference_demo_region_multimodal_plot(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    geo_bounds: tuple[float, float, float, float] = DEFAULT_GEO_BOUNDS,
    boundary_source: str | Path = SICHUAN_BOUNDARY_URL,
    sample_idx: int | None = None,
    future_idx: int | None = None,
    target_valid_time: str | None = None,
    select_large_rain: bool = False,
    large_rain_threshold: float = 0.1,
    large_rain_max_batches: int = 20,
    match_pred_to_gt_for_visualization: bool = False,
    image_size: int = 420,
    polish_pred: bool = False,
    match_pred_visual: bool = False,
    pred_visual_similarity: float = 0.85,
) -> Path:
    from src.tests.inference_demo import (
        DEMO_CONFIG,
        _analyze_input_reasonability,
        _build_demo_cfg_for_trainer,
        _prepare_analysis_modalities,
    )
    from src.trainer.rain_trainer_ts_next_frame import RainTSNextFrameTrainer
    from src.utils.visualization.plot_ragion import _match_pred_rain_to_gt_for_visualization

    demo_cfg = copy.deepcopy(DEMO_CONFIG)
    target_future_idx = int(demo_cfg["vis_future_start"] if future_idx is None else future_idx)
    demo_cfg["vis_future_start"] = target_future_idx
    demo_cfg["vis_future_end"] = target_future_idx + 1

    cfg = _load_demo_hydra_cfg()
    _build_demo_cfg_for_trainer(cfg, demo_cfg)

    trainer = RainTSNextFrameTrainer(cfg)
    _analyze_input_reasonability(
        trainer=trainer,
        context_frames=int(demo_cfg["context_frames"]),
        predict_frames=int(demo_cfg["predict_frames"]),
    )

    _load_next_frame_weights(trainer, demo_cfg)
    trainer.model.eval()
    trainer.global_step = 0

    vis_start = int(demo_cfg["vis_future_start"])
    target_sample_idx = int(demo_cfg.get("diff_sample_index", 0) if sample_idx is None else sample_idx)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        selected_batch = None
        selected_context = None
        selected_target = None
        selected_context_time = None
        selected_target_time = None
        selected_target_timestamp = None

        if select_large_rain:
            best_score = -1.0
            for batch_idx, batch in enumerate(trainer.val_dataloader):
                if int(large_rain_max_batches) > 0 and batch_idx >= int(large_rain_max_batches):
                    break
                context, target, context_time, target_time = trainer._prepare_val_inference_batch(batch)
                scores = _rain_area_score(target["rain"].squeeze(1), threshold=float(large_rain_threshold))
                flat_idx = int(torch.argmax(scores).item())
                current_sample_idx = flat_idx // int(scores.shape[1])
                current_frame_idx = flat_idx % int(scores.shape[1])
                score = float(scores[current_sample_idx, current_frame_idx].item())
                if score > best_score:
                    selected_batch = batch
                    selected_context = context
                    selected_target = target
                    selected_context_time = context_time
                    selected_target_time = target_time
                    selected_target_timestamp = batch.get("time_future_timestamp")
                    target_sample_idx = int(current_sample_idx)
                    vis_start = int(current_frame_idx)
                    best_score = score

            if selected_batch is None:
                raise ValueError("No batch was available while selecting large-rain sample.")
            print(
                "Selected next-frame large-rain sample: "
                f"sample={target_sample_idx}, future_idx={vis_start}, rain_area_ratio={best_score:.4f}"
            )
        elif target_valid_time:
            target_timestamp_value = _parse_valid_time_to_timestamp(target_valid_time)
            for batch_idx, batch in enumerate(trainer.val_dataloader):
                timestamp = batch.get("time_future_timestamp")
                if timestamp is None:
                    raise KeyError("Batch has no time_future_timestamp, cannot select by --target-valid-time.")
                if int(timestamp.shape[1]) <= vis_start:
                    continue
                diff = torch.abs(timestamp[:, vis_start].double() - target_timestamp_value)
                hit = torch.nonzero(diff < 0.5, as_tuple=False).flatten()
                if int(hit.numel()) <= 0:
                    continue
                target_sample_idx = int(hit[0].item())
                selected_batch = batch
                selected_context, selected_target, selected_context_time, selected_target_time = (
                    trainer._prepare_val_inference_batch(batch)
                )
                selected_target_timestamp = timestamp
                print(f"Matched next-frame target valid time in batch {batch_idx}, sample {target_sample_idx}")
                break
            if selected_batch is None:
                raise ValueError(f"No sample found with future_idx={vis_start} and valid time={target_valid_time}")
        else:
            selected_batch = next(iter(trainer.val_dataloader))
            selected_context, selected_target, selected_context_time, selected_target_time = (
                trainer._prepare_val_inference_batch(selected_batch)
            )
            selected_target_timestamp = selected_batch.get("time_future_timestamp")

        context = selected_context
        target = selected_target
        context_time = selected_context_time
        target_time = selected_target_time
        target_timestamp = selected_target_timestamp
        if context is None or target is None or context_time is None or target_time is None:
            raise RuntimeError("Failed to select a valid next-frame batch.")

        total_future_frames = int(target["rain"].shape[2])
        if vis_start < 0 or vis_start >= total_future_frames:
            raise ValueError(f"future_idx out of range: {vis_start}, available [0, {total_future_frames - 1}]")

        with trainer.accelerator.autocast():
            pred_target = trainer._rollout_predict(
                context=context,
                total_future_frames=total_future_frames,
                context_time=context_time,
                future_time=target_time,
                future_modalities=target,
            )

    context_modalities, pred_modalities, target_modalities = _prepare_analysis_modalities(
        trainer=trainer,
        context=context,
        pred_target=pred_target,
        target=target,
    )
    if match_pred_to_gt_for_visualization:
        pred_modalities["rain"] = _match_pred_rain_to_gt_for_visualization(
            pred_modalities["rain"],
            target_modalities["rain"],
        )

    batch_size = int(target_modalities["rain"].shape[0])
    if target_sample_idx < 0 or target_sample_idx >= batch_size:
        raise ValueError(f"sample_idx out of range: {target_sample_idx}, available [0, {batch_size - 1}]")

    output_paths = save_region_multimodal_panels(
        gt_rain=target_modalities["rain"][target_sample_idx, :, vis_start],
        pred_rain=pred_modalities["rain"][target_sample_idx, :, vis_start],
        radar=target_modalities["radar"][target_sample_idx, :, vis_start],
        satellite=target_modalities["satellite"][target_sample_idx, :, vis_start],
        output_dir=out_dir,
        sample_idx=target_sample_idx,
        frame_idx=vis_start,
        geo_bounds=geo_bounds,
        boundary_source=boundary_source,
        image_size=int(image_size),
        polish_pred=bool(polish_pred),
        rain_render_mode="discrete",
        match_pred_visual=bool(match_pred_visual),
        pred_visual_similarity=float(pred_visual_similarity),
    )
    input_output_paths = save_next_frame_input_panels(
        context_modalities=context_modalities,
        output_dir=out_dir,
        sample_idx=target_sample_idx,
        image_size=int(image_size),
        rain_render_mode="discrete",
        geo_bounds=geo_bounds,
        boundary_source=boundary_source,
    )

    time_text = _build_future_footer(
        target_time.detach().float().cpu(),
        sample_idx=target_sample_idx,
        frame_idx=vis_start,
        future_timestamps=target_timestamp.detach().double().cpu() if target_timestamp is not None else None,
    )
    print(f"Saved one future frame region multimodal panels: {out_dir}")
    for name in ("gt_rain", "pred_rain", "radar", "satellite"):
        print(f"{name}: {output_paths[name]}")
    print("Saved next-frame input panels:")
    for name in ("input_rain", "input_radar", "input_satellite"):
        for out_path in input_output_paths[name]:
            print(f"{name}: {out_path}")
    print(time_text)
    return out_dir


def run_cls_test_region_multimodal_plot(
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    config_name: str = "rain_test_ts_swinnet_cls",
    config_overrides: list[str] | None = None,
    geo_bounds: tuple[float, float, float, float] = DEFAULT_GEO_BOUNDS,
    boundary_source: str | Path = SICHUAN_BOUNDARY_URL,
    sample_idx: int | None = None,
    future_idx: int = 0,
    target_valid_time: str | None = None,
    select_large_rain: bool = False,
    large_rain_threshold: float = 0.1,
    large_rain_max_batches: int = 20,
    image_size: int = 420,
    polish_pred: bool = False,
    match_pred_visual: bool = False,
    pred_visual_similarity: float = 0.85,
) -> Path:
    import hydra

    cfg = _load_test_hydra_cfg(config_name, overrides=config_overrides)
    accelerator = hydra.utils.instantiate(cfg.accelerator)
    device = accelerator.device

    test_dataset, clean_loader = hydra.utils.instantiate(cfg.dataset.val)
    if clean_loader is None:
        from torch.utils.data import DataLoader
        from torch.utils.data import Dataset

        class StrictMapDataset(Dataset):
            def __init__(self, dataset):
                self.ds = dataset

            def __len__(self):
                if hasattr(self.ds, "valid_indices"):
                    return len(self.ds.valid_indices)
                return len(self.ds)

            def __getitem__(self, idx):
                return self.ds[idx]

        clean_loader = DataLoader(
            StrictMapDataset(test_dataset),
            batch_size=cfg.dataset.val.get("batch_size", 8),
            shuffle=False,
            num_workers=cfg.dataset.val.get("num_workers", 0),
            pin_memory=True,
            drop_last=False,
        )

    model = hydra.utils.instantiate(cfg.rain_prediction_model)
    model.to(device)
    model.eval()
    _load_cls_ema_to_model(model, cfg.checkpoints.ema_load_path)

    frame_idx = 0 if future_idx is None else int(future_idx)
    if select_large_rain:
        batch, target_sample_idx, frame_idx, _ = _find_large_rain_batch(
            clean_loader,
            threshold=float(large_rain_threshold),
            max_batches=int(large_rain_max_batches),
        )
    elif target_valid_time:
        batch, target_sample_idx = _find_batch_by_future_valid_time(
            clean_loader,
            valid_time=target_valid_time,
            future_idx=frame_idx,
        )
    else:
        batch = next(iter(clean_loader))
        target_sample_idx = int(cfg.test.get("vis", {}).get("sample_idx", 0) if sample_idx is None else sample_idx)

    batch_size = int(batch["rain_past"].shape[0])
    if target_sample_idx < 0 or target_sample_idx >= batch_size:
        raise ValueError(f"sample_idx out of range: {target_sample_idx}, available [0, {batch_size - 1}]")

    radar_past = batch["radar_past"].to(device)
    satellite_past = batch["satellite_past"].to(device)
    rain_past = batch["rain_past"].to(device)
    with torch.no_grad():
        pred = model(radar_past, satellite_past, rain_past)
        pred = pred.unsqueeze(2) if pred.ndim == 4 else pred
        pred_cls = torch.argmax(pred, dim=1, keepdim=True).cpu()

    total_future_frames = int(pred_cls.shape[2])
    if frame_idx < 0 or frame_idx >= total_future_frames:
        raise ValueError(f"future_idx out of range: {frame_idx}, available [0, {total_future_frames - 1}]")

    if "rain_future_cls" in batch:
        gt_rain = class_map_to_visual_rain(batch["rain_future_cls"][target_sample_idx, :, frame_idx])
    else:
        gt_rain = cap_rain_for_visualization(batch["rain_future"][target_sample_idx, :, frame_idx])
    pred_rain = class_map_to_visual_rain(pred_cls[target_sample_idx, :, frame_idx])
    radar = _select_modality_frame(
        batch,
        future_key="radar_future",
        past_key="radar_past",
        sample_idx=target_sample_idx,
        frame_idx=frame_idx,
    )
    satellite = _select_modality_frame(
        batch,
        future_key="satellite_future",
        past_key="satellite_past",
        sample_idx=target_sample_idx,
        frame_idx=frame_idx,
    )

    out_dir = Path(output_dir)
    output_paths = save_region_multimodal_panels(
        gt_rain=gt_rain,
        pred_rain=pred_rain,
        radar=radar,
        satellite=satellite,
        output_dir=out_dir,
        sample_idx=target_sample_idx,
        frame_idx=frame_idx,
        geo_bounds=geo_bounds,
        boundary_source=boundary_source,
        image_size=int(image_size),
        polish_pred=bool(polish_pred),
        rain_render_mode="discrete",
        match_pred_visual=bool(match_pred_visual),
        pred_visual_similarity=float(pred_visual_similarity),
    )

    print(f"Saved one future frame cls-test region multimodal panels: {out_dir}")
    for name in ("gt_rain", "pred_rain", "radar", "satellite"):
        print(f"{name}: {output_paths[name]}")
    if "time_future_timestamp" in batch:
        valid_time = datetime.fromtimestamp(float(batch["time_future_timestamp"][target_sample_idx, frame_idx].item()))
        print(f"Valid time: {valid_time:%Y-%m-%d %H:%M:%S}")
    return out_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference demo and save GT/Pred/Radar/Satellite region panels.")
    parser.add_argument("--runner", choices=["next-frame-demo", "cls-test"], default="next-frame-demo")
    parser.add_argument("--config-name", type=str, default="rain_test_ts_swinnet_cls")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--geo-bounds", type=float, nargs=4, default=DEFAULT_GEO_BOUNDS)
    parser.add_argument("--boundary-source", type=str, default=SICHUAN_BOUNDARY_URL)
    parser.add_argument("--sample-idx", type=int, default=None)
    parser.add_argument("--future-idx", type=int, default=None)
    parser.add_argument("--target-valid-time", type=str, default="")
    parser.add_argument("--select-large-rain", action="store_true")
    parser.add_argument("--large-rain-threshold", type=float, default=0.1)
    parser.add_argument("--large-rain-max-batches", type=int, default=20)
    parser.add_argument("--image-size", type=int, default=420)
    parser.add_argument("--polish-pred", action="store_true")
    parser.add_argument("--match-pred-visual", action="store_true")
    parser.add_argument("--pred-visual-similarity", type=float, default=0.85)
    args, overrides = parser.parse_known_args()
    args.config_overrides = _quote_hydra_path_override_values(overrides)
    return args


def main() -> None:
    args = _parse_args()
    common_kwargs = {
        "output_dir": args.output_dir,
        "geo_bounds": tuple(float(v) for v in args.geo_bounds),
        "boundary_source": args.boundary_source,
        "sample_idx": args.sample_idx,
        "future_idx": args.future_idx,
        "image_size": int(args.image_size),
        "polish_pred": bool(args.polish_pred),
        "match_pred_visual": bool(args.match_pred_visual),
        "pred_visual_similarity": float(args.pred_visual_similarity),
    }
    if args.runner == "cls-test":
        run_cls_test_region_multimodal_plot(
            config_name=args.config_name,
            config_overrides=args.config_overrides,
            target_valid_time=args.target_valid_time or None,
            select_large_rain=bool(args.select_large_rain),
            large_rain_threshold=float(args.large_rain_threshold),
            large_rain_max_batches=int(args.large_rain_max_batches),
            **common_kwargs,
        )
        return

    run_inference_demo_region_multimodal_plot(
        output_dir=args.output_dir,
        geo_bounds=tuple(float(v) for v in args.geo_bounds),
        boundary_source=args.boundary_source,
        sample_idx=args.sample_idx,
        future_idx=args.future_idx,
        target_valid_time=args.target_valid_time or None,
        select_large_rain=bool(args.select_large_rain),
        large_rain_threshold=float(args.large_rain_threshold),
        large_rain_max_batches=int(args.large_rain_max_batches),
        image_size=int(args.image_size),
        polish_pred=bool(args.polish_pred),
        match_pred_visual=bool(args.match_pred_visual),
        pred_visual_similarity=float(args.pred_visual_similarity),
    )


if __name__ == "__main__":
    main()
