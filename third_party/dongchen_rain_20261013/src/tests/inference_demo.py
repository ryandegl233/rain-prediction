"""
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

import sys
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
import torch
from omegaconf import DictConfig
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.trainer.rain_trainer_ts_next_frame import RainTSNextFrameTrainer
from src.utils.visualization.plot import plot_any_modality


def _slice_modalities_by_time(
    x: dict[str, torch.Tensor],
    start_idx: int,
    end_idx: int,
) -> dict[str, torch.Tensor]:
    sliced: dict[str, torch.Tensor] = {}
    for key, value in x.items():
        sliced[key] = value[:, :, start_idx:end_idx, :, :]
    return sliced


def _build_demo_cfg_for_trainer(cfg: DictConfig, demo_cfg: dict) -> None:
    context_frames = int(demo_cfg["context_frames"])
    predict_frames = int(demo_cfg["predict_frames"])

    cfg.train.debug = True
    cfg.train.proj_dir = str(demo_cfg["output_dir"])
    cfg.train.log.log_with_time = False

    cfg.dataset.n_past = context_frames
    cfg.dataset.n_futures = predict_frames

    cfg.dataset.train_inp_dirs = demo_cfg["data_dirs"]
    cfg.dataset.val_inp_dirs = demo_cfg["data_dirs"]
    cfg.dataset.train.inp_dirs = "${dataset.train_inp_dirs}"
    cfg.dataset.val.inp_dirs = "${dataset.val_inp_dirs}"

    cfg.dataset.train.n_past = context_frames
    cfg.dataset.train.n_futures = predict_frames
    cfg.dataset.val.n_past = context_frames
    cfg.dataset.val.n_futures = predict_frames

    cfg.dataset.augmentation.train_enabled = False
    cfg.dataset.augmentation.val_enabled = False
    cfg.dataset.rain_ratio_filter.enabled = False
    cfg.dataset.train.aug_enabled = False
    cfg.dataset.val.aug_enabled = False
    cfg.dataset.train.rain_ratio_filter_enabled = False
    cfg.dataset.val.rain_ratio_filter_enabled = False

    cfg.dataset.val.batch_size = int(demo_cfg.get("val_batch_size", cfg.dataset.val.batch_size))
    cfg.dataset.val.num_workers = int(demo_cfg.get("val_num_workers", cfg.dataset.val.num_workers))
    cfg.dataset.val.persistent_workers = bool(demo_cfg.get("val_persistent_workers", cfg.dataset.val.persistent_workers))
    cfg.dataset.val.pin_memory = bool(demo_cfg.get("val_pin_memory", cfg.dataset.val.pin_memory))
    if "val_prefetch_factor" in demo_cfg:
        cfg.dataset.val.prefetch_factor = int(demo_cfg["val_prefetch_factor"])


def _denormalize_rain_if_needed(trainer: RainTSNextFrameTrainer, rain: torch.Tensor) -> torch.Tensor:
    if not getattr(trainer, "modality_zero_centering", False):
        return rain
    return trainer._denormalize_rain_for_metrics(rain)


def _prepare_analysis_modalities(
    trainer: RainTSNextFrameTrainer,
    context: torch.Tensor,
    pred_target: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    context_split = trainer._split_modalities(context)
    context_modalities = {k: v.detach().float().cpu() for k, v in context_split.items()}
    pred_modalities = {k: v.detach().float().cpu() for k, v in pred_target.items()}
    target_modalities = {k: v.detach().float().cpu() for k, v in target.items()}

    context_modalities["rain"] = _denormalize_rain_if_needed(trainer, context_modalities["rain"])
    pred_modalities["rain"] = _denormalize_rain_if_needed(trainer, pred_modalities["rain"])
    target_modalities["rain"] = _denormalize_rain_if_needed(trainer, target_modalities["rain"])
    return context_modalities, pred_modalities, target_modalities


def _analyze_input_reasonability(trainer: RainTSNextFrameTrainer, context_frames: int, predict_frames: int) -> None:
    frame_patch_size = int(getattr(trainer.model, "frame_patch_size", 1))
    max_frames = int(getattr(trainer.model, "max_frames", context_frames + predict_frames))
    total = context_frames + predict_frames

    issues: list[str] = []
    suggestions: list[str] = []

    if context_frames <= 0:
        issues.append("context_frames 必须 > 0")
    if predict_frames <= 0:
        issues.append("predict_frames 必须 > 0")
    if total > max_frames:
        issues.append(f"context_frames + predict_frames = {total} 超过 model.max_frames = {max_frames}")

    if context_frames % frame_patch_size != 0:
        suggestions.append(
            f"建议 context_frames 是 frame_patch_size({frame_patch_size}) 的整数倍，避免 rollout seed 切块不整齐"
        )
    if context_frames < frame_patch_size:
        suggestions.append(f"建议 context_frames >= frame_patch_size({frame_patch_size})，否则可能触发 rollout 约束")

    if issues:
        detail = "；".join(issues)
        raise ValueError(f"输入帧配置不合理：{detail}")

    print("[Frame Check] 输入帧配置可运行")
    print(f"[Frame Check] context_frames={context_frames}, predict_frames={predict_frames}, total={total}")
    print(f"[Frame Check] model.max_frames={max_frames}, frame_patch_size={frame_patch_size}")
    if suggestions:
        print("[Frame Check] 合理性建议：")
        for item in suggestions:
            print(f"  - {item}")


def _load_ema_to_model(trainer: RainTSNextFrameTrainer, ema_dir: str) -> None:
    ema_file = Path(ema_dir) / "ema.pt"
    if not ema_file.exists():
        raise FileNotFoundError(f"ema.pt not found in {ema_dir}")

    if trainer.ema_model is None:
        raise ValueError("trainer.ema_model is None. Please enable EMA in config before loading ema.pt.")

    ema_state_dict = torch.load(str(ema_file), map_location=trainer.device, weights_only=False)
    trainer.ema_model.load_state_dict(ema_state_dict)

    if hasattr(trainer.ema_model, "copy_params_from_ema_to_model"):
        trainer.ema_model.copy_params_from_ema_to_model()
    elif hasattr(trainer.ema_model, "copy_to"):
        unwrapped = trainer.accelerator.unwrap_model(trainer.model)
        trainer.ema_model.copy_to(unwrapped.parameters())
    else:
        raise AttributeError("EMA object has no supported method to copy EMA weights into model.")

    print(f"Loaded EMA checkpoint and copied to model: {ema_file}")


def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    return np.clip(img, 0, 255).astype(np.uint8)


def _plot_frame(frame: torch.Tensor, *, modality_name: str) -> np.ndarray:
    img = plot_any_modality(frame.detach().cpu(), modality_name=modality_name, to_PIL=False)
    return _to_uint8_rgb(img)


def _plot_diff_frame(diff_frame: torch.Tensor, *, vmax: float) -> np.ndarray:
    diff_np = diff_frame.detach().cpu().squeeze().numpy()
    vmax = max(float(vmax), 1.0e-6)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    img = cm.get_cmap("RdBu_r")(norm(diff_np))[..., :3]
    return _to_uint8_rgb(img * 255.0)


def _build_reference_sequence(context_modality: torch.Tensor, target_modality: torch.Tensor) -> torch.Tensor:
    history_last = context_modality[:, :, -1:]
    if int(target_modality.shape[2]) <= 1:
        return history_last
    return torch.cat([history_last, target_modality[:, :, :-1]], dim=2)


def _rain_residual_enabled(trainer: RainTSNextFrameTrainer) -> bool:
    loss_cfg = trainer.train_cfg.get("loss", {})
    residual_cfg = loss_cfg.get("rain_residual", {})
    return bool(residual_cfg.get("enabled", False))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(denominator) <= 1.0e-8:
        return 0.0
    return numerator / denominator


def _flatten_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1).to(dtype=torch.float32)


def _pearson_corr(x: torch.Tensor, y: torch.Tensor) -> float:
    x_flat = _flatten_tensor(x)
    y_flat = _flatten_tensor(y)
    x_centered = x_flat - x_flat.mean()
    y_centered = y_flat - y_flat.mean()
    denom = torch.sqrt((x_centered.square().sum()) * (y_centered.square().sum())).item()
    if denom <= 1.0e-8:
        return 0.0
    return float((x_centered * y_centered).sum().item() / denom)


def _binary_iou(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    a = mask_a.to(dtype=torch.bool)
    b = mask_b.to(dtype=torch.bool)
    union = torch.logical_or(a, b).sum().item()
    if union <= 0:
        return 1.0
    intersection = torch.logical_and(a, b).sum().item()
    return float(intersection / union)


def _signed_change_iou(pred_diff: torch.Tensor, gt_diff: torch.Tensor, threshold: float) -> float:
    pred_pos = pred_diff >= threshold
    gt_pos = gt_diff >= threshold
    pred_neg = pred_diff <= -threshold
    gt_neg = gt_diff <= -threshold
    pos_iou = _binary_iou(pred_pos, gt_pos)
    neg_iou = _binary_iou(pred_neg, gt_neg)
    return 0.5 * (pos_iou + neg_iou)


def _compute_diff_spatial_metrics(pred_diff: torch.Tensor, gt_diff: torch.Tensor) -> dict[str, float]:
    gt_abs = gt_diff.abs()
    gt_abs_mean = float(gt_abs.mean().item())
    gt_abs_max = float(gt_abs.max().item())
    active_threshold = max(gt_abs_mean, 0.1 * gt_abs_max, 1.0e-6)
    sign_threshold = max(0.5 * gt_abs_mean, 0.05 * gt_abs_max, 1.0e-6)

    pred_active = pred_diff.abs() >= active_threshold
    gt_active = gt_abs >= active_threshold
    return {
        "diff_corr": _pearson_corr(pred_diff, gt_diff),
        "abs_diff_corr": _pearson_corr(pred_diff.abs(), gt_abs),
        "change_iou": _binary_iou(pred_active, gt_active),
        "signed_change_iou": _signed_change_iou(pred_diff, gt_diff, threshold=sign_threshold),
        "active_threshold": active_threshold,
        "sign_threshold": sign_threshold,
    }


def _build_diagnostic_rollout_seed(
    trainer: RainTSNextFrameTrainer,
    context: torch.Tensor,
    seed_frames: int,
    detach_history: bool,
    context_time: torch.Tensor | None = None,
    future_time: torch.Tensor | None = None,
    future_modality_forcing: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    frame_patch_size = trainer._resolve_frame_patch_size()
    seed_list: list[torch.Tensor] = [context[:, :, -frame_patch_size:, :, :]]
    context_cur = context
    context_time_cur = context_time
    generated = frame_patch_size

    while generated < seed_frames:
        forward_kwargs = {
            "context_x": context_cur,
            "target_x": seed_list[-1],
            "predict_frames": frame_patch_size,
            "strict_target_isolation": bool(trainer.train_cfg.strict_target_isolation),
            "return_modality_dict": True,
        }
        if torch.is_tensor(context_time_cur):
            forward_kwargs["context_time"] = context_time_cur
            forward_kwargs["target_time"] = context_time_cur[:, -frame_patch_size:]
        pred_one = trainer.model.forward_ar(**forward_kwargs)
        if _rain_residual_enabled(trainer):
            seed_modalities = trainer._split_modalities(seed_list[-1])
            pred_one = dict(pred_one)
            pred_one["rain"] = trainer._apply_rain_residual_output(pred_one["rain"], seed_modalities["rain"])

        pred_one_tensor = trainer._merge_modalities(pred_one["radar"], pred_one["satellite"], pred_one["rain"])
        pred_hist = pred_one_tensor.detach() if detach_history else pred_one_tensor
        if future_modality_forcing is not None:
            step_start = generated - frame_patch_size
            step_end = generated
            pred_modalities = trainer._split_modalities(pred_hist)
            pred_hist = trainer._merge_modalities(
                future_modality_forcing["radar"][:, :, step_start:step_end],
                future_modality_forcing["satellite"][:, :, step_start:step_end],
                pred_modalities["rain"],
            )

        seed_list.append(pred_hist)
        context_cur = torch.cat([context_cur, pred_hist], dim=2)
        if torch.is_tensor(context_time_cur):
            if future_time is None:
                raise ValueError("future_time should be provided when context_time is provided.")
            next_time = future_time[:, generated - frame_patch_size : generated]
            context_time_cur = torch.cat([context_time_cur, next_time], dim=1)
        generated += frame_patch_size

    return torch.cat(seed_list, dim=2)


def _rollout_with_residual_diagnostics(
    trainer: RainTSNextFrameTrainer,
    context: torch.Tensor,
    target: dict[str, torch.Tensor],
    context_time: torch.Tensor | None = None,
    future_time: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    mode, rollout_block_size, detach_history, use_gt_future_modalities = trainer._resolve_rollout_mode()
    total_future_frames = int(target["rain"].shape[2])
    frame_patch_size = trainer._resolve_frame_patch_size()

    remaining = total_future_frames
    produced = 0
    context_cur = context
    context_time_cur = context_time

    rain_ref_list: list[torch.Tensor] = []
    pred_delta_list: list[torch.Tensor] = []
    gt_delta_list: list[torch.Tensor] = []
    final_rain_list: list[torch.Tensor] = []

    while remaining > 0:
        if mode == "frame":
            chunk = frame_patch_size
        else:
            raw_chunk = min(rollout_block_size, remaining)
            chunk = (raw_chunk // frame_patch_size) * frame_patch_size
            if chunk <= 0:
                chunk = frame_patch_size
        if chunk > remaining:
            chunk = remaining

        chunk_future_time = None if future_time is None else future_time[:, produced : produced + chunk]
        chunk_future_modalities = {
            "radar": target["radar"][:, :, produced : produced + chunk],
            "satellite": target["satellite"][:, :, produced : produced + chunk],
        }
        seed_block = _build_diagnostic_rollout_seed(
            trainer=trainer,
            context=context_cur,
            seed_frames=chunk,
            detach_history=detach_history,
            context_time=context_time_cur,
            future_time=chunk_future_time,
            future_modality_forcing=chunk_future_modalities if use_gt_future_modalities else None,
        )

        forward_kwargs = {
            "context_x": context_cur,
            "target_x": seed_block,
            "predict_frames": chunk,
            "strict_target_isolation": bool(trainer.train_cfg.strict_target_isolation),
            "return_modality_dict": True,
        }
        if torch.is_tensor(context_time_cur):
            if chunk_future_time is None:
                raise ValueError("chunk_future_time should be provided when context_time_cur is provided.")
            target_seed_time = torch.cat(
                [context_time_cur[:, -frame_patch_size:], chunk_future_time[:, :-frame_patch_size]],
                dim=1,
            )
            forward_kwargs["context_time"] = context_time_cur
            forward_kwargs["target_time"] = target_seed_time
        pred_block_raw = trainer.model.forward_ar(**forward_kwargs)

        seed_modalities = trainer._split_modalities(seed_block)
        rain_ref = seed_modalities["rain"]
        if _rain_residual_enabled(trainer):
            pred_delta = pred_block_raw["rain"]
            final_rain = trainer._apply_rain_residual_output(pred_delta, rain_ref)
        else:
            final_rain = pred_block_raw["rain"]
            pred_delta = final_rain - rain_ref

        target_rain = target["rain"][:, :, produced : produced + chunk]
        gt_delta = target_rain - rain_ref

        rain_ref_list.append(rain_ref)
        pred_delta_list.append(pred_delta)
        gt_delta_list.append(gt_delta)
        final_rain_list.append(final_rain)

        pred_hist = trainer._merge_modalities(pred_block_raw["radar"], pred_block_raw["satellite"], final_rain)
        pred_hist = pred_hist.detach() if detach_history else pred_hist
        if use_gt_future_modalities:
            pred_modalities = trainer._split_modalities(pred_hist)
            pred_hist = trainer._merge_modalities(
                chunk_future_modalities["radar"],
                chunk_future_modalities["satellite"],
                pred_modalities["rain"],
            )
        context_cur = torch.cat([context_cur, pred_hist], dim=2)
        if torch.is_tensor(context_time_cur):
            if chunk_future_time is None:
                raise ValueError("chunk_future_time should be provided when context_time_cur is provided.")
            context_time_cur = torch.cat([context_time_cur, chunk_future_time], dim=1)

        produced += chunk
        remaining -= chunk

    return {
        "rain_ref": torch.cat(rain_ref_list, dim=2),
        "pred_delta": torch.cat(pred_delta_list, dim=2),
        "gt_delta": torch.cat(gt_delta_list, dim=2),
        "final_rain": torch.cat(final_rain_list, dim=2),
    }


def _save_modality_diff_analysis(
    trainer: RainTSNextFrameTrainer,
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

    ref_modality = _build_reference_sequence(context_modality, target_modality)
    ref_modality = ref_modality[:, :, future_start:future_end]
    pred_modality = pred_modality[:, :, future_start:future_end]
    target_modality = target_modality[:, :, future_start:future_end]
    pred_diff = pred_modality - ref_modality
    gt_diff = target_modality - ref_modality

    if diff_vmax is None:
        auto_vmax = torch.stack([pred_diff.abs().amax(), gt_diff.abs().amax()]).amax().item()
        diff_vmax_value = max(float(auto_vmax), 1.0e-3)
    else:
        diff_vmax_value = max(float(diff_vmax), 1.0e-6)

    first_tile = _plot_frame(ref_modality[sample_idx, :, 0], modality_name=modality_name)
    tile_h, tile_w = int(first_tile.shape[0]), int(first_tile.shape[1])
    left_label_w = 112
    top_header_h = 26
    cell_gap = 6
    row_gap = 8
    n_rows = int(pred_modality.shape[2])
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
            _plot_frame(ref_modality[sample_idx, :, row_idx], modality_name=modality_name),
            _plot_frame(pred_modality[sample_idx, :, row_idx], modality_name=modality_name),
            _plot_frame(target_modality[sample_idx, :, row_idx], modality_name=modality_name),
            _plot_diff_frame(pred_diff[sample_idx, :, row_idx], vmax=diff_vmax_value),
            _plot_diff_frame(gt_diff[sample_idx, :, row_idx], vmax=diff_vmax_value),
        ]
        for col_idx, img in enumerate(row_images):
            x = left_label_w + col_idx * (tile_w + cell_gap)
            canvas.paste(Image.fromarray(img), (x, y))

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

    image_path = viz_dir / f"{output_prefix}_{modality_name}_sample{sample_idx}_frames_{future_start}_{future_end}_diff_grid.png"
    stats_path = viz_dir / f"{output_prefix}_{modality_name}_sample{sample_idx}_frames_{future_start}_{future_end}_diff_stats.txt"
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


def _save_rain_residual_diagnostic_analysis(
    trainer: RainTSNextFrameTrainer,
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
        auto_delta_vmax = torch.stack([pred_delta.abs().amax(), gt_delta.abs().amax(), delta_error.abs().amax()]).amax()
        delta_vmax_value = max(float(auto_delta_vmax.item()), 1.0e-3)
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
        for col_idx, img in enumerate(row_images):
            x = left_label_w + col_idx * (tile_w + cell_gap)
            canvas.paste(Image.fromarray(img), (x, y))

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

    image_path = viz_dir / f"{output_prefix}_rain_residual_sample{sample_idx}_frames_{future_start}_{future_end}_grid.png"
    stats_path = viz_dir / f"{output_prefix}_rain_residual_sample{sample_idx}_frames_{future_start}_{future_end}_stats.txt"
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


def run_demo(cfg: DictConfig, demo_cfg: dict) -> None:
    _build_demo_cfg_for_trainer(cfg, demo_cfg)

    print("Instantiating trainer...")
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
    diff_sample_idx = int(demo_cfg.get("diff_sample_index", 0))
    diff_vmax_rain = demo_cfg.get("rain_diff_vmax", None)
    diff_vmax_radar = demo_cfg.get("radar_diff_vmax", None)
    save_residual_diagnostic = bool(demo_cfg.get("save_residual_diagnostic", True))
    residual_delta_vmax = demo_cfg.get("residual_delta_vmax", None)

    with torch.no_grad():
        batch = next(iter(trainer.val_dataloader))
        context, target, context_time, target_time = trainer._prepare_val_inference_batch(batch)

        total_future_frames = int(target["rain"].shape[2])
        print(f"Run inference: context={int(context.shape[2])}, future={total_future_frames}")

        if vis_start < 0 or vis_start >= total_future_frames:
            raise ValueError(
                f"vis_future_start 越界: {vis_start}, 可用范围 [0, {total_future_frames - 1}]"
            )
        if vis_end <= vis_start or vis_end > total_future_frames:
            raise ValueError(
                f"vis_future_end 越界: {vis_end}, 必须满足 {vis_start} < vis_future_end <= {total_future_frames}"
            )

        with trainer.accelerator.autocast():
            pred_target = trainer._rollout_predict(
                context=context,
                total_future_frames=total_future_frames,
                context_time=context_time,
                future_time=target_time,
                future_modalities=target,
            )
            residual_diag = None
            if save_residual_diagnostic:
                residual_diag = _rollout_with_residual_diagnostics(
                    trainer=trainer,
                    context=context,
                    target=target,
                    context_time=context_time,
                    future_time=target_time,
                )

    context_modalities, pred_modalities, target_modalities = _prepare_analysis_modalities(
        trainer=trainer,
        context=context,
        pred_target=pred_target,
        target=target,
    )

    batch_size = int(pred_modalities["rain"].shape[0])
    if batch_size <= 0:
        raise ValueError("推理 batch 为空，无法做差分分析。")
    if diff_sample_idx < 0 or diff_sample_idx >= batch_size:
        raise ValueError(f"diff_sample_index 越界: {diff_sample_idx}, 可用范围 [0, {batch_size - 1}]")

    pred_for_vis = _slice_modalities_by_time(pred_target, start_idx=vis_start, end_idx=vis_end)
    target_for_vis = _slice_modalities_by_time(target, start_idx=vis_start, end_idx=vis_end)

    trainer._save_val_visualizations(
        context=context,
        pred_target=pred_for_vis,
        target=target_for_vis,
        output_prefix=f"inference_future_{vis_start}_{vis_end}",
    )

    _save_modality_diff_analysis(
        trainer=trainer,
        context_modality=context_modalities["rain"],
        pred_modality=pred_modalities["rain"],
        target_modality=target_modalities["rain"],
        modality_name="rain",
        future_start=vis_start,
        future_end=vis_end,
        sample_idx=diff_sample_idx,
        output_prefix="inference",
        diff_vmax=diff_vmax_rain,
    )
    _save_modality_diff_analysis(
        trainer=trainer,
        context_modality=context_modalities["radar"],
        pred_modality=pred_modalities["radar"],
        target_modality=target_modalities["radar"],
        modality_name="radar",
        future_start=vis_start,
        future_end=vis_end,
        sample_idx=diff_sample_idx,
        output_prefix="inference",
        diff_vmax=diff_vmax_radar,
    )
    if residual_diag is not None:
        radar_ref = _build_reference_sequence(context_modalities["radar"], target_modalities["radar"])
        _save_rain_residual_diagnostic_analysis(
            trainer=trainer,
            residual_diag=residual_diag,
            target_rain=target_modalities["rain"],
            radar_ref=radar_ref,
            radar_target=target_modalities["radar"],
            future_start=vis_start,
            future_end=vis_end,
            sample_idx=diff_sample_idx,
            output_prefix="inference",
            delta_vmax=residual_delta_vmax,
            radar_diff_vmax=diff_vmax_radar,
        )

    print("Visualization done: 三行按 radar/satellite/rain 排列，左侧 past，中间 pred，右侧 future gt。")
    print("Diff analysis done: 已输出 rain 和 radar 的逐帧差分图、幅值比值统计、以及空间位置一致性指标。")
    if save_residual_diagnostic:
        print("Residual diagnostic done: 已输出 rain_ref、pred_delta、gt_delta、final_pred、gt_rain、delta_error、radar_diff。")
    print(f"Visualized future range: [{vis_start}, {vis_end})")
    print(f"Saved to: {trainer.proj_dir}/val_viz/step_00000000/")


DEMO_CONFIG: dict = {
    "data_dirs": [
        "data2/litdata_train_2025/litdata_interval_30/202508",
    ],
    "context_frames": 4,
    "predict_frames": 12,
    "val_batch_size": 1,
    "val_num_workers": 1,
    "val_persistent_workers": False,
    "val_pin_memory": False,
    #"ema_path": "/home/rainpred/RainPrediction/runs/time_series_next_frame/2026-05-21_00-16-03_stage1_next_frame_block/ema",
    #"ema_path": "/home/rainpred/RainPrediction/runs/time_series_next_frame_temdiff_correct_roll/2026-07-15/13-54-05_stage1_next_frame_block/ema",
    #"ema_path": "/home/rainpred/RainPrediction/runs/next_frame_temdiff_correct_roll_residual/2026-07-28/21-19-29_stage1_next_frame_block/ema",
    #"ema_path": "/home/rainpred/RainPrediction/runs/next_frame_temdiff_correct_roll_residual/2026-07-30/02-24-18_stage1_next_frame_block/ema",
    #"checkpoint_path": "/home/rainpred/RainPrediction/runs/next_frame_temdiff_correct_roll_residual_roll4/2026-07-31/10-24-50_stage1_next_frame_block/checkpoints/checkpoint_4",
    #"ema_path": "/home/rainpred/RainPrediction/runs/next_frame_temdiff_correct_roll_residual_roll4/2026-07-31/10-24-50_stage1_next_frame_block/ema",
    #"ema_path": "/home/rainpred/RainPrediction/runs/next_frame_temdiff_correct_roll_residual_roll4/2026-08-01/01-10-09_stage1_next_frame_block/ema",
    #"ema_path": "/home/rainpred/RainPrediction/runs/next_frame_temdiff_correct_roll_residual_roll4/2026-08-02/02-56-25_stage1_next_frame_block/ema",
    #"ema_path": "/home/rainpred/RainPrediction/runs/next_frame_final/2026-08-17/14-41-50_stage1_next_frame_block/ema",
    "ema_path": "/home/rainpred/RainPrediction/runs/next_frame_delta_filter/2026-08-21/20-58-56_stage1_next_frame_delta_filter/ema",

    #"output_dir": "vis/roll+residual_main+see_all+frame4+sequenceloss(true)4",
    "output_dir": "vis/roll3",
    "vis_future_start": 0,
    "vis_future_end": 12,
    "diff_sample_index": 0,
    "rain_diff_vmax": None,
    "radar_diff_vmax": None,
    "save_residual_diagnostic": True,
    "residual_delta_vmax": None,
}


if __name__ == "__main__":
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    config_dir = str(Path(__file__).resolve().parents[1] / "config" / "ts_rain_train")
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="rain_trainer_ts_next_frame")

    run_demo(cfg, DEMO_CONFIG)
