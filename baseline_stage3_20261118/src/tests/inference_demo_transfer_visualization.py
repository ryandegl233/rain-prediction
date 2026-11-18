from matplotlib import colormaps
import matplotlib.colors as mcolors
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from src.tests.inference_demo import (
    _build_reference_sequence,
    _compute_diff_spatial_metrics,
    _denormalize_rain_if_needed,
    _plot_frame,
    _safe_ratio,
    _to_uint8_rgb,
)
from src.trainer.rain_trainer_ts_next_frame_transfer import RainTSNextFrameTransferTrainer


def _plot_diff_frame(diff_frame: torch.Tensor, *, vmax: float) -> np.ndarray:
    diff_np = diff_frame.detach().cpu().squeeze().numpy()
    vmax = max(float(vmax), 1.0e-6)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    image = colormaps["RdBu_r"](norm(diff_np))[..., :3]
    return _to_uint8_rgb(image * 255.0)


def save_conditioned_timeline(
    trainer: RainTSNextFrameTransferTrainer,
    context: dict[str, torch.Tensor],
    predicted_rain: torch.Tensor,
    target: dict[str, torch.Tensor],
    *,
    future_start: int,
    future_end: int,
    sample_idx: int,
    output_prefix: str,
) -> None:
    viz_dir = trainer.proj_dir / "val_viz" / f"step_{trainer.global_step:08d}"
    viz_dir.mkdir(parents=True, exist_ok=True)

    future_condition = {
        "radar": target["radar"][:, :, future_start:future_end],
        "satellite": target["satellite"][:, :, future_start:future_end],
        "rain": predicted_rain[:, :, future_start:future_end],
    }
    future_target = {
        name: value[:, :, future_start:future_end]
        for name, value in target.items()
    }
    history_frames = int(context["rain"].shape[2])
    future_frames = future_end - future_start
    labels = [f"history {index + 1}" for index in range(history_frames)]
    labels.extend(f"cond/pred t+{future_start + index + 1}" for index in range(future_frames))
    labels.extend(f"gt t+{future_start + index + 1}" for index in range(future_frames))

    first_tile = _plot_frame(context["radar"][sample_idx, :, 0], modality_name="radar")
    tile_h, tile_w = int(first_tile.shape[0]), int(first_tile.shape[1])
    left_label_w = 152
    top_header_h = 26
    cell_gap = 6
    row_gap = 8
    row_specs = (
        ("radar (GT cond)", "radar"),
        ("satellite (GT cond)", "satellite"),
        ("rain (prediction)", "rain"),
    )
    canvas_w = left_label_w + len(labels) * tile_w + max(0, len(labels) - 1) * cell_gap
    canvas_h = top_header_h + len(row_specs) * tile_h + max(0, len(row_specs) - 1) * row_gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for col_idx, label in enumerate(labels):
        x = left_label_w + col_idx * (tile_w + cell_gap)
        draw.text((x + 4, 6), label, fill=(0, 0, 0), font=font)

    for separator in (history_frames, history_frames + future_frames):
        x = left_label_w + separator * tile_w + max(separator - 1, 0) * cell_gap + (cell_gap // 2)
        draw.line((x, 0, x, canvas_h), fill=(160, 160, 160), width=2)

    for row_idx, (row_label, modality_name) in enumerate(row_specs):
        y = top_header_h + row_idx * (tile_h + row_gap)
        draw.text((8, y + 8), row_label, fill=(0, 0, 0), font=font)
        frames = (
            context[modality_name][sample_idx].unbind(dim=1)
            + future_condition[modality_name][sample_idx].unbind(dim=1)
            + future_target[modality_name][sample_idx].unbind(dim=1)
        )
        for col_idx, frame in enumerate(frames):
            x = left_label_w + col_idx * (tile_w + cell_gap)
            image = _plot_frame(frame, modality_name=modality_name)
            canvas.paste(Image.fromarray(image), (x, y))

    output_path = viz_dir / f"{output_prefix}_sample{sample_idx}_timeline.jpg"
    canvas.save(output_path, quality=85)
    print(f"Saved conditioned multimodal timeline: {output_path}")


def save_conditioning_diff_analysis(
    trainer: RainTSNextFrameTransferTrainer,
    context_modality: torch.Tensor,
    condition_modality: torch.Tensor,
    *,
    modality_name: str,
    future_start: int,
    future_end: int,
    sample_idx: int,
    output_prefix: str,
    diff_vmax: float | None,
) -> None:
    viz_dir = trainer.proj_dir / "val_viz" / f"step_{trainer.global_step:08d}"
    viz_dir.mkdir(parents=True, exist_ok=True)

    reference = _build_reference_sequence(context_modality, condition_modality)
    reference = reference[:, :, future_start:future_end]
    condition = condition_modality[:, :, future_start:future_end]
    condition_diff = condition - reference
    if diff_vmax is None:
        diff_vmax_value = max(float(condition_diff.abs().amax().item()), 1.0e-3)
    else:
        diff_vmax_value = max(float(diff_vmax), 1.0e-6)

    first_tile = _plot_frame(reference[sample_idx, :, 0], modality_name=modality_name)
    tile_h, tile_w = int(first_tile.shape[0]), int(first_tile.shape[1])
    left_label_w = 112
    top_header_h = 26
    cell_gap = 6
    row_gap = 8
    headers = [f"{modality_name}_t", "GT_condition_t+1", "condition_diff"]
    n_rows = int(condition.shape[2])
    canvas_w = left_label_w + len(headers) * tile_w + (len(headers) - 1) * cell_gap
    canvas_h = top_header_h + n_rows * tile_h + max(0, n_rows - 1) * row_gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for col_idx, header in enumerate(headers):
        x = left_label_w + col_idx * (tile_w + cell_gap)
        draw.text((x + 4, 6), header, fill=(0, 0, 0), font=font)

    stats_lines: list[str] = []
    for row_idx in range(n_rows):
        frame_idx = future_start + row_idx
        y = top_header_h + row_idx * (tile_h + row_gap)
        draw.text((8, y + 8), f"future[{frame_idx}]", fill=(0, 0, 0), font=font)
        diff_frame = condition_diff[sample_idx, :, row_idx]
        row_images = (
            _plot_frame(reference[sample_idx, :, row_idx], modality_name=modality_name),
            _plot_frame(condition[sample_idx, :, row_idx], modality_name=modality_name),
            _plot_diff_frame(diff_frame, vmax=diff_vmax_value),
        )
        for col_idx, image in enumerate(row_images):
            x = left_label_w + col_idx * (tile_w + cell_gap)
            canvas.paste(Image.fromarray(image), (x, y))
        stats_lines.append(
            " | ".join(
                [
                    f"future[{frame_idx}]",
                    f"condition_diff_mean={float(diff_frame.mean().item()):.6f}",
                    f"condition_diff_abs_mean={float(diff_frame.abs().mean().item()):.6f}",
                    f"condition_diff_abs_max={float(diff_frame.abs().max().item()):.6f}",
                ]
            )
        )

    image_path = (
        viz_dir
        / f"{output_prefix}_{modality_name}_sample{sample_idx}_frames_{future_start}_{future_end}_diff_grid.png"
    )
    stats_path = (
        viz_dir
        / f"{output_prefix}_{modality_name}_sample{sample_idx}_frames_{future_start}_{future_end}_diff_stats.txt"
    )
    canvas.save(image_path)
    stats_path.write_text(
        "\n".join(
            [
                "analysis=known_future_condition",
                f"modality={modality_name}",
                f"sample_idx={sample_idx}",
                f"future_range=[{future_start}, {future_end})",
                f"diff_vmax={diff_vmax_value:.6f}",
                "",
                *stats_lines,
            ]
        ),
        encoding="utf-8",
    )
    print(f"Saved {modality_name} conditioning diff image: {image_path}")
    print(f"Saved {modality_name} conditioning diff stats: {stats_path}")


def save_modality_diff_analysis(
    trainer: RainTSNextFrameTransferTrainer,
    context_modality: torch.Tensor,
    pred_modality: torch.Tensor,
    target_modality: torch.Tensor,
    *,
    modality_name: str,
    future_start: int,
    future_end: int,
    sample_idx: int,
    output_prefix: str,
    diff_vmax: float | None,
) -> None:
    viz_dir = trainer.proj_dir / "val_viz" / f"step_{trainer.global_step:08d}"
    viz_dir.mkdir(parents=True, exist_ok=True)

    font = ImageFont.load_default()
    headers = [f"{modality_name}_t", "pred_t+1", "gt_t+1", "pred_diff", "gt_diff"]

    reference = _build_reference_sequence(context_modality, target_modality)
    reference = reference[:, :, future_start:future_end]
    prediction = pred_modality[:, :, future_start:future_end]
    target = target_modality[:, :, future_start:future_end]
    pred_diff = prediction - reference
    gt_diff = target - reference

    if diff_vmax is None:
        auto_vmax = torch.stack([pred_diff.abs().amax(), gt_diff.abs().amax()]).amax().item()
        diff_vmax_value = max(float(auto_vmax), 1.0e-3)
    else:
        diff_vmax_value = max(float(diff_vmax), 1.0e-6)

    first_tile = _plot_frame(reference[sample_idx, :, 0], modality_name=modality_name)
    tile_h, tile_w = int(first_tile.shape[0]), int(first_tile.shape[1])
    left_label_w = 112
    top_header_h = 26
    cell_gap = 6
    row_gap = 8
    n_rows = int(prediction.shape[2])
    n_cols = len(headers)
    canvas_w = left_label_w + n_cols * tile_w + max(0, n_cols - 1) * cell_gap
    canvas_h = top_header_h + n_rows * tile_h + max(0, n_rows - 1) * row_gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for col_idx, header in enumerate(headers):
        x = left_label_w + col_idx * (tile_w + cell_gap)
        draw.text((x + 4, 6), header, fill=(0, 0, 0), font=font)

    stats_lines: list[str] = []
    for row_idx in range(n_rows):
        frame_idx = future_start + row_idx
        y = top_header_h + row_idx * (tile_h + row_gap)
        draw.text((8, y + 8), f"future[{frame_idx}]", fill=(0, 0, 0), font=font)

        row_images = [
            _plot_frame(reference[sample_idx, :, row_idx], modality_name=modality_name),
            _plot_frame(prediction[sample_idx, :, row_idx], modality_name=modality_name),
            _plot_frame(target[sample_idx, :, row_idx], modality_name=modality_name),
            _plot_diff_frame(pred_diff[sample_idx, :, row_idx], vmax=diff_vmax_value),
            _plot_diff_frame(gt_diff[sample_idx, :, row_idx], vmax=diff_vmax_value),
        ]
        for col_idx, image in enumerate(row_images):
            x = left_label_w + col_idx * (tile_w + cell_gap)
            canvas.paste(Image.fromarray(image), (x, y))

        pred_diff_frame = pred_diff[sample_idx, :, row_idx]
        gt_diff_frame = gt_diff[sample_idx, :, row_idx]
        pred_abs_mean = float(pred_diff_frame.abs().mean().item())
        gt_abs_mean = float(gt_diff_frame.abs().mean().item())
        pred_abs_max = float(pred_diff_frame.abs().max().item())
        gt_abs_max = float(gt_diff_frame.abs().max().item())
        spatial_metrics = _compute_diff_spatial_metrics(pred_diff_frame, gt_diff_frame)
        stats_lines.append(
            " | ".join(
                [
                    f"future[{frame_idx}]",
                    f"pred_abs_mean={pred_abs_mean:.6f}",
                    f"gt_abs_mean={gt_abs_mean:.6f}",
                    f"mean_ratio={_safe_ratio(pred_abs_mean, gt_abs_mean):.6f}",
                    f"pred_abs_max={pred_abs_max:.6f}",
                    f"gt_abs_max={gt_abs_max:.6f}",
                    f"max_ratio={_safe_ratio(pred_abs_max, gt_abs_max):.6f}",
                    f"diff_corr={spatial_metrics['diff_corr']:.6f}",
                    f"abs_diff_corr={spatial_metrics['abs_diff_corr']:.6f}",
                    f"change_iou={spatial_metrics['change_iou']:.6f}",
                    f"signed_change_iou={spatial_metrics['signed_change_iou']:.6f}",
                    f"active_thr={spatial_metrics['active_threshold']:.6f}",
                    f"sign_thr={spatial_metrics['sign_threshold']:.6f}",
                ]
            )
        )

    image_path = (
        viz_dir
        / f"{output_prefix}_{modality_name}_sample{sample_idx}_frames_{future_start}_{future_end}_diff_grid.png"
    )
    stats_path = (
        viz_dir
        / f"{output_prefix}_{modality_name}_sample{sample_idx}_frames_{future_start}_{future_end}_diff_stats.txt"
    )
    canvas.save(image_path)
    stats_path.write_text(
        "\n".join(
            [
                f"modality={modality_name}",
                f"sample_idx={sample_idx}",
                f"future_range=[{future_start}, {future_end})",
                f"diff_vmax={diff_vmax_value:.6f}",
                "",
                *stats_lines,
            ]
        ),
        encoding="utf-8",
    )
    print(f"Saved {modality_name} diff analysis image: {image_path}")
    print(f"Saved {modality_name} diff analysis stats: {stats_path}")


def save_rain_residual_diagnostic_analysis(
    trainer: RainTSNextFrameTransferTrainer,
    residual_diag: dict[str, torch.Tensor],
    target_rain: torch.Tensor,
    radar_ref: torch.Tensor,
    radar_target: torch.Tensor,
    *,
    future_start: int,
    future_end: int,
    sample_idx: int,
    output_prefix: str,
    delta_vmax: float | None,
    radar_diff_vmax: float | None,
) -> None:
    viz_dir = trainer.proj_dir / "val_viz" / f"step_{trainer.global_step:08d}"
    viz_dir.mkdir(parents=True, exist_ok=True)

    rain_ref = residual_diag["rain_ref"][:, :, future_start:future_end].detach().float().cpu()
    pred_delta = residual_diag["pred_delta"][:, :, future_start:future_end].detach().float().cpu()
    gt_delta = residual_diag["gt_delta"][:, :, future_start:future_end].detach().float().cpu()
    final_rain = residual_diag["final_rain"][:, :, future_start:future_end].detach().float().cpu()
    gt_rain = target_rain[:, :, future_start:future_end].detach().float().cpu()
    radar_diff = (radar_target - radar_ref)[:, :, future_start:future_end].detach().float().cpu()

    rain_ref = _denormalize_rain_if_needed(trainer, rain_ref)
    final_rain = _denormalize_rain_if_needed(trainer, final_rain)
    gt_rain = _denormalize_rain_if_needed(trainer, gt_rain)
    if getattr(trainer, "modality_zero_centering", False):
        scale = float(getattr(trainer, "rain_norm_std", 1.0) or 1.0)
        pred_delta = pred_delta * scale
        gt_delta = gt_delta * scale

    delta_error = pred_delta - gt_delta
    if delta_vmax is None:
        auto_vmax = torch.stack([pred_delta.abs().amax(), gt_delta.abs().amax(), delta_error.abs().amax()]).amax()
        delta_vmax_value = max(float(auto_vmax.item()), 1.0e-3)
    else:
        delta_vmax_value = max(float(delta_vmax), 1.0e-6)
    if radar_diff_vmax is None:
        radar_diff_vmax_value = max(float(radar_diff.abs().amax().item()), 1.0e-3)
    else:
        radar_diff_vmax_value = max(float(radar_diff_vmax), 1.0e-6)

    headers = ["rain_ref", "pred_delta", "gt_delta", "final_pred", "gt_rain", "delta_error", "radar_diff"]
    font = ImageFont.load_default()
    first_tile = _plot_frame(rain_ref[sample_idx, :, 0], modality_name="rain")
    tile_h, tile_w = int(first_tile.shape[0]), int(first_tile.shape[1])
    left_label_w = 112
    top_header_h = 26
    cell_gap = 6
    row_gap = 8
    n_rows = int(pred_delta.shape[2])
    n_cols = len(headers)
    canvas_w = left_label_w + n_cols * tile_w + max(0, n_cols - 1) * cell_gap
    canvas_h = top_header_h + n_rows * tile_h + max(0, n_rows - 1) * row_gap
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for col_idx, header in enumerate(headers):
        x = left_label_w + col_idx * (tile_w + cell_gap)
        draw.text((x + 4, 6), header, fill=(0, 0, 0), font=font)

    stats_lines: list[str] = []
    for row_idx in range(n_rows):
        frame_idx = future_start + row_idx
        y = top_header_h + row_idx * (tile_h + row_gap)
        draw.text((8, y + 8), f"future[{frame_idx}]", fill=(0, 0, 0), font=font)

        row_images = [
            _plot_frame(rain_ref[sample_idx, :, row_idx], modality_name="rain"),
            _plot_diff_frame(pred_delta[sample_idx, :, row_idx], vmax=delta_vmax_value),
            _plot_diff_frame(gt_delta[sample_idx, :, row_idx], vmax=delta_vmax_value),
            _plot_frame(final_rain[sample_idx, :, row_idx], modality_name="rain"),
            _plot_frame(gt_rain[sample_idx, :, row_idx], modality_name="rain"),
            _plot_diff_frame(delta_error[sample_idx, :, row_idx], vmax=delta_vmax_value),
            _plot_diff_frame(radar_diff[sample_idx, :, row_idx], vmax=radar_diff_vmax_value),
        ]
        for col_idx, image in enumerate(row_images):
            x = left_label_w + col_idx * (tile_w + cell_gap)
            canvas.paste(Image.fromarray(image), (x, y))

        pred_delta_frame = pred_delta[sample_idx, :, row_idx]
        gt_delta_frame = gt_delta[sample_idx, :, row_idx]
        delta_error_frame = delta_error[sample_idx, :, row_idx]
        spatial_metrics = _compute_diff_spatial_metrics(pred_delta_frame, gt_delta_frame)
        stats_lines.append(
            " | ".join(
                [
                    f"future[{frame_idx}]",
                    f"pred_delta_abs_mean={float(pred_delta_frame.abs().mean().item()):.6f}",
                    f"gt_delta_abs_mean={float(gt_delta_frame.abs().mean().item()):.6f}",
                    f"mean_ratio={_safe_ratio(float(pred_delta_frame.abs().mean().item()), float(gt_delta_frame.abs().mean().item())):.6f}",
                    f"delta_error_abs_mean={float(delta_error_frame.abs().mean().item()):.6f}",
                    f"pred_delta_abs_max={float(pred_delta_frame.abs().max().item()):.6f}",
                    f"gt_delta_abs_max={float(gt_delta_frame.abs().max().item()):.6f}",
                    f"diff_corr={spatial_metrics['diff_corr']:.6f}",
                    f"abs_diff_corr={spatial_metrics['abs_diff_corr']:.6f}",
                    f"change_iou={spatial_metrics['change_iou']:.6f}",
                    f"signed_change_iou={spatial_metrics['signed_change_iou']:.6f}",
                    f"active_thr={spatial_metrics['active_threshold']:.6f}",
                    f"sign_thr={spatial_metrics['sign_threshold']:.6f}",
                ]
            )
        )

    image_path = (
        viz_dir
        / f"{output_prefix}_rain_residual_sample{sample_idx}_frames_{future_start}_{future_end}_grid.png"
    )
    stats_path = (
        viz_dir
        / f"{output_prefix}_rain_residual_sample{sample_idx}_frames_{future_start}_{future_end}_stats.txt"
    )
    canvas.save(image_path)
    stats_path.write_text(
        "\n".join(
            [
                "diagnostic=rain_residual",
                f"sample_idx={sample_idx}",
                f"future_range=[{future_start}, {future_end})",
                f"delta_vmax={delta_vmax_value:.6f}",
                f"radar_diff_vmax={radar_diff_vmax_value:.6f}",
                "",
                *stats_lines,
            ]
        ),
        encoding="utf-8",
    )
    print(f"Saved rain residual diagnostic image: {image_path}")
    print(f"Saved rain residual diagnostic stats: {stats_path}")
