"""Compare raw and light-rain-suppressed autoregressive transfer inference.

Usage:
    python src/tests/inference_demo_transfer_light_rain_suppression.py \
        --checkpoint-path runs/next_frame_final/2026-10-12/20-17-52_stage1_next_frame_block/checkpoints/step_00070000 \
        --predict-frames 12 \
        --target-valid-time "2025-08-04 23:00:00" \
        --history-window-frames 7 \
        --clear-threshold 0.02 \
        --light-upper 0.1 \
        --light-scale 0.9
"""

import argparse
import json
import math
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from omegaconf import DictConfig, OmegaConf

from src.tests.inference_demo import _analyze_input_reasonability, _build_demo_cfg_for_trainer
from src.tests.inference_demo_transfer import (
    CheckpointFiles,
    _compare_frame_stats,
    _rollout_predict_sliding_window,
    _save_prediction_visuals,
    load_checkpoint_weights,
    resolve_checkpoint_files,
    select_inference_window,
)
from src.trainer.rain_trainer_ts_next_frame_transfer import RainTSNextFrameTransferTrainer


def suppress_light_rain(
    rain: torch.Tensor,
    *,
    clear_threshold: float,
    light_upper: float,
    light_scale: float,
) -> torch.Tensor:
    if not all(math.isfinite(value) for value in (clear_threshold, light_upper, light_scale)):
        raise ValueError("Light-rain suppression settings must be finite")
    if clear_threshold < 0 or light_upper <= clear_threshold:
        raise ValueError("Expected 0 <= clear_threshold < light_upper")
    if not 0 <= light_scale <= 1:
        raise ValueError("light_scale must be in [0, 1]")

    clear_mask = rain < clear_threshold
    light_mask = (rain >= clear_threshold) & (rain < light_upper)
    suppressed = torch.where(light_mask, rain * light_scale, rain)
    return torch.where(clear_mask, torch.zeros_like(suppressed), suppressed)


def rollout_predict_with_light_rain_suppression(
    trainer: RainTSNextFrameTransferTrainer,
    context: torch.Tensor,
    target: dict[str, torch.Tensor],
    *,
    context_time: torch.Tensor | None,
    future_time: torch.Tensor | None,
    history_window_frames: int,
    clear_threshold: float,
    light_upper: float,
    light_scale: float,
) -> tuple[dict[str, torch.Tensor], list[int]]:
    model = trainer.accelerator.unwrap_model(trainer.model)
    frame_patch_size = int(model.frame_patch_size)
    total_future_frames = int(target["rain"].shape[2])
    if history_window_frames <= 0 or history_window_frames % frame_patch_size != 0:
        raise ValueError(
            "history_window_frames must be positive and divisible by frame_patch_size, "
            f"got history_window_frames={history_window_frames}, frame_patch_size={frame_patch_size}"
        )
    if history_window_frames + frame_patch_size > int(model.max_frames):
        raise ValueError(
            "Sliding-window model input exceeds max_frames: "
            f"history={history_window_frames}, seed={frame_patch_size}, max_frames={int(model.max_frames)}"
        )
    if total_future_frames <= 0 or total_future_frames % frame_patch_size != 0:
        raise ValueError(
            "Future frames must be positive and divisible by frame_patch_size, "
            f"got future_frames={total_future_frames}, frame_patch_size={frame_patch_size}"
        )
    if context_time is not None or future_time is not None:
        if context_time is None or future_time is None:
            raise ValueError("context_time and future_time must both be provided for sliding rollout")
        if int(context_time.shape[1]) != int(context.shape[2]):
            raise ValueError("context_time length must match context frames")
        if int(future_time.shape[1]) != total_future_frames:
            raise ValueError("future_time length must match future frames")
    if not trainer._resolve_rain_residual_cfg()[0]:
        raise ValueError("Light-rain suppression demo requires rain residual prediction")

    context_cur = context[:, :, -history_window_frames:]
    context_time_cur = None if context_time is None else context_time[:, -history_window_frames:]
    rain_predictions: list[torch.Tensor] = []
    effective_rain_deltas: list[torch.Tensor] = []
    model_rain_deltas: list[torch.Tensor] = []
    model_input_frames: list[int] = []

    for start in range(0, total_future_frames, frame_patch_size):
        end = start + frame_patch_size
        seed = context_cur[:, :, -frame_patch_size:]
        rain_seed = seed[:, -trainer.rain_c :]
        model_input_frames.append(int(context_cur.shape[2]) + frame_patch_size)
        model_rain_delta = trainer.model.forward_ar(
            context_x=context_cur,
            target_x=seed,
            predict_frames=frame_patch_size,
            strict_target_isolation=bool(trainer.train_cfg.strict_target_isolation),
            return_modality_dict=False,
            context_time=context_time_cur,
            target_time=None if context_time_cur is None else context_time_cur[:, -frame_patch_size:],
        )
        if not torch.is_tensor(model_rain_delta):
            raise TypeError(f"Expected rain tensor from rain-only decoder, got {type(model_rain_delta).__name__}")

        rain = trainer._apply_rain_residual_output(model_rain_delta, rain_seed)
        rain = suppress_light_rain(
            rain,
            clear_threshold=clear_threshold,
            light_upper=light_upper,
            light_scale=light_scale,
        )
        rain_predictions.append(rain)
        effective_rain_deltas.append(rain - rain_seed)
        model_rain_deltas.append(model_rain_delta)

        if end >= total_future_frames:
            continue

        history = trainer._merge_modalities(
            target["radar"][:, :, start:end],
            target["satellite"][:, :, start:end],
            rain.detach(),
        )
        context_cur = torch.cat([context_cur, history], dim=2)[:, :, -history_window_frames:]
        if context_time_cur is not None:
            if future_time is None:
                raise ValueError("future_time is required when context_time is provided")
            context_time_cur = torch.cat([context_time_cur, future_time[:, start:end]], dim=1)
            context_time_cur = context_time_cur[:, -history_window_frames:]

    return {
        "rain": torch.cat(rain_predictions, dim=2),
        "rain_delta": torch.cat(effective_rain_deltas, dim=2),
        "model_rain_delta": torch.cat(model_rain_deltas, dim=2),
    }, model_input_frames


def _binary_frame_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    threshold: float,
) -> list[dict[str, float]]:
    pred_mask = prediction >= threshold
    target_mask = target >= threshold
    reduce_dims = (0, 1, 3, 4)
    pred_area = pred_mask.float().mean(dim=reduce_dims)
    target_area = target_mask.float().mean(dim=reduce_dims)
    hits = (pred_mask & target_mask).sum(dim=reduce_dims).float()
    false_alarms = (pred_mask & ~target_mask).sum(dim=reduce_dims).float()
    misses = (~pred_mask & target_mask).sum(dim=reduce_dims).float()
    far = false_alarms / (hits + false_alarms).clamp_min(1.0)
    csi = hits / (hits + false_alarms + misses).clamp_min(1.0)

    return [
        {
            "area": float(pred_area[index].item()),
            "target_area": float(target_area[index].item()),
            "far": float(far[index].item()),
            "csi": float(csi[index].item()),
        }
        for index in range(int(prediction.shape[2]))
    ]


def _write_suppression_comparison(
    output_dir: Path,
    output_prefix: str,
    trainer: RainTSNextFrameTransferTrainer,
    raw_prediction: torch.Tensor,
    suppressed_prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    clear_threshold: float,
    light_upper: float,
    light_scale: float,
    temporal_change_analysis: dict[str, object],
) -> tuple[Path, Path]:
    raw_rain = trainer._denormalize_rain_for_metrics(raw_prediction.detach().float()).cpu()
    suppressed_rain = trainer._denormalize_rain_for_metrics(suppressed_prediction.detach().float()).cpu()
    target_rain = trainer._denormalize_rain_for_metrics(target.detach().float()).cpu()
    thresholds = (0.01, 0.05, 0.1)
    threshold_metrics: dict[str, dict[str, list[dict[str, float]]]] = {}
    for threshold in thresholds:
        threshold_metrics[f"{threshold:.2f}"] = {
            "raw": _binary_frame_metrics(raw_rain, target_rain, threshold),
            "suppressed": _binary_frame_metrics(suppressed_rain, target_rain, threshold),
        }

    reduce_dims = (0, 1, 3, 4)
    raw_mean = raw_rain.mean(dim=reduce_dims)
    suppressed_mean = suppressed_rain.mean(dim=reduce_dims)
    target_mean = target_rain.mean(dim=reduce_dims)
    report = {
        "suppression": {
            "clear_threshold": clear_threshold,
            "light_upper": light_upper,
            "light_scale": light_scale,
            "applied_before_history_update": True,
        },
        "frame_mean": [
            {
                "future_index": index,
                "raw": float(raw_mean[index].item()),
                "suppressed": float(suppressed_mean[index].item()),
                "target": float(target_mean[index].item()),
            }
            for index in range(int(raw_rain.shape[2]))
        ],
        "threshold_metrics": threshold_metrics,
        "temporal_change": temporal_change_analysis,
    }
    json_path = output_dir / f"{output_prefix}_comparison.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")

    text_lines = [
        "reference=raw_sliding_window",
        "candidate=light_rain_suppressed_sliding_window",
        f"clear_threshold={clear_threshold:.6f}",
        f"light_upper={light_upper:.6f}",
        f"light_scale={light_scale:.6f}",
        "applied_before_history_update=True",
    ]
    for threshold_name, variants in threshold_metrics.items():
        text_lines.extend(
            [
                "",
                f"[threshold={threshold_name}]",
                "future | raw_area | suppressed_area | target_area | raw_far | "
                "suppressed_far | raw_csi | suppressed_csi",
            ]
        )
        for index, (raw_values, suppressed_values) in enumerate(zip(variants["raw"], variants["suppressed"])):
            text_lines.append(
                f"{index} | {raw_values['area']:.6f} | {suppressed_values['area']:.6f} | "
                f"{raw_values['target_area']:.6f} | {raw_values['far']:.6f} | "
                f"{suppressed_values['far']:.6f} | {raw_values['csi']:.6f} | "
                f"{suppressed_values['csi']:.6f}"
            )
    text_path = output_dir / f"{output_prefix}_comparison.txt"
    text_path.write_text("\n".join(text_lines), encoding="utf-8")
    print(f"Saved suppression comparison: {json_path}")
    print(f"Saved suppression summary: {text_path}")
    return json_path, text_path


def run_demo(cfg: DictConfig, demo_cfg: dict[str, object], files: CheckpointFiles) -> None:
    _build_demo_cfg_for_trainer(cfg, demo_cfg)
    cfg.train.resume_path = None
    cfg.ema.beta = 0

    trainer = RainTSNextFrameTransferTrainer(cfg)
    if bool(getattr(trainer, "modality_zero_centering", False)):
        raise ValueError("This demo expects physical rain values with modality_zero_centering=False")
    history_window_frames = int(demo_cfg["history_window_frames"])
    frame_patch_size = trainer._resolve_frame_patch_size()
    _analyze_input_reasonability(trainer, context_frames=history_window_frames, predict_frames=frame_patch_size)
    source, weight_file = load_checkpoint_weights(trainer, files, str(demo_cfg["weight_source"]))
    print(f"Loaded {source} weights: {weight_file}")

    trainer.model.eval()
    dataset_index_value = demo_cfg.get("dataset_index")
    target_valid_time_value = demo_cfg.get("target_valid_time")
    batch, sample_selection = select_inference_window(
        trainer.val_dataset,
        dataset_index=None if dataset_index_value is None else int(dataset_index_value),
        target_valid_time=None if target_valid_time_value is None else str(target_valid_time_value),
    )
    target_time_token = (
        str(sample_selection["target_valid_time"])
        .replace("-", "")
        .replace(":", "")
        .replace(" ", "T")
    )
    sample_tag = f"dataset{int(sample_selection['dataset_index']):06d}_target{target_time_token}"
    clear_threshold = float(demo_cfg["clear_threshold"])
    light_upper = float(demo_cfg["light_upper"])
    light_scale = float(demo_cfg["light_scale"])

    with torch.no_grad(), trainer.accelerator.autocast():
        context, target, context_time, future_time = trainer._prepare_val_inference_batch(batch)
        raw_prediction, raw_model_input_frames = _rollout_predict_sliding_window(
            trainer,
            context,
            target,
            context_time=context_time,
            future_time=future_time,
            history_window_frames=history_window_frames,
        )
        suppressed_prediction, suppressed_model_input_frames = rollout_predict_with_light_rain_suppression(
            trainer,
            context,
            target,
            context_time=context_time,
            future_time=future_time,
            history_window_frames=history_window_frames,
            clear_threshold=clear_threshold,
            light_upper=light_upper,
            light_scale=light_scale,
        )

    if raw_model_input_frames != suppressed_model_input_frames:
        raise RuntimeError("Raw and suppressed rollouts used different model input lengths")
    total_future_frames = int(raw_prediction["rain"].shape[2])
    vis_start = int(demo_cfg["vis_future_start"])
    vis_end = int(demo_cfg["vis_future_end"])
    sample_idx = int(demo_cfg["sample_index"])
    if not 0 <= vis_start < vis_end <= total_future_frames:
        raise ValueError(f"Visualization range [{vis_start}, {vis_end}) exceeds {total_future_frames} future frames")
    if not 0 <= sample_idx < int(raw_prediction["rain"].shape[0]):
        raise ValueError(f"sample_index={sample_idx} is outside the inference batch")

    raw_prefix = f"inference_sliding_w{history_window_frames}_raw_{sample_tag}"
    suppressed_prefix = f"inference_sliding_w{history_window_frames}_light_suppressed_{sample_tag}"
    for prefix, prediction in ((raw_prefix, raw_prediction), (suppressed_prefix, suppressed_prediction)):
        _save_prediction_visuals(
            trainer=trainer,
            context=context,
            target=target,
            prediction=prediction,
            future_start=vis_start,
            future_end=vis_end,
            sample_idx=sample_idx,
            output_prefix=prefix,
            rain_diff_vmax=None,
            radar_diff_vmax=None,
            residual_delta_vmax=None,
            save_residual_diagnostic=bool(demo_cfg["save_residual_diagnostic"]),
            save_radar_conditioning=False,
        )

    output_dir = trainer.proj_dir / "val_viz" / f"step_{trainer.global_step:08d}"
    raw_rain_stats = output_dir / f"{raw_prefix}_rain_sample{sample_idx}_frames_{vis_start}_{vis_end}_diff_stats.txt"
    suppressed_rain_stats = (
        output_dir / f"{suppressed_prefix}_rain_sample{sample_idx}_frames_{vis_start}_{vis_end}_diff_stats.txt"
    )
    temporal_change_analysis = _compare_frame_stats(raw_rain_stats, suppressed_rain_stats)
    comparison_prefix = f"inference_sliding_w{history_window_frames}_light_suppression_{sample_tag}"
    _write_suppression_comparison(
        output_dir,
        comparison_prefix,
        trainer,
        raw_prediction["rain"],
        suppressed_prediction["rain"],
        target["rain"],
        clear_threshold=clear_threshold,
        light_upper=light_upper,
        light_scale=light_scale,
        temporal_change_analysis=temporal_change_analysis,
    )

    manifest = {
        **sample_selection,
        "sample_tag": sample_tag,
        "checkpoint_config": str(files.config),
        "checkpoint_file": str(weight_file),
        "checkpoint_step": int(trainer.global_step),
        "weight_source": source,
        "context_frames": int(demo_cfg["context_frames"]),
        "predict_frames": int(demo_cfg["predict_frames"]),
        "history_window_frames": history_window_frames,
        "model_input_frames": raw_model_input_frames,
        "clear_threshold": clear_threshold,
        "light_upper": light_upper,
        "light_scale": light_scale,
        "suppression_applied_before_history_update": True,
    }
    manifest_path = output_dir / f"{comparison_prefix}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Saved suppression manifest: {manifest_path}")
    print(f"Light-rain suppression inference complete: {output_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare raw and light-rain-suppressed sliding inference")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--weight-source", choices=("auto", "ema", "model"), default="auto")
    parser.add_argument("--data-dir", action="append", dest="data_dirs")
    parser.add_argument("--output-dir", default="vis/inference_demo_transfer_light_rain_suppression")
    parser.add_argument("--context-frames", type=int, default=None)
    parser.add_argument("--predict-frames", type=int, default=None)
    parser.add_argument("--history-window-frames", type=int, default=7)
    parser.add_argument("--vis-future-start", type=int, default=0)
    parser.add_argument("--vis-future-end", type=int, default=None)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--clear-threshold", type=float, default=0.02)
    parser.add_argument("--light-upper", type=float, default=0.1)
    parser.add_argument("--light-scale", type=float, default=0.7)
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument("--dataset-index", type=int, default=None)
    selection_group.add_argument("--target-valid-time", default=None)
    parser.add_argument("--no-residual-diagnostic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    files = resolve_checkpoint_files(args.checkpoint_path)
    cfg = OmegaConf.load(files.config)
    context_frames = int(cfg.dataset.n_past) if args.context_frames is None else args.context_frames
    predict_frames = int(cfg.dataset.n_futures) if args.predict_frames is None else args.predict_frames
    data_dirs = list(cfg.dataset.val_inp_dirs) if args.data_dirs is None else args.data_dirs
    vis_end = predict_frames if args.vis_future_end is None else args.vis_future_end
    demo_cfg: dict[str, object] = {
        "data_dirs": data_dirs,
        "context_frames": context_frames,
        "predict_frames": predict_frames,
        "val_batch_size": 1,
        "val_num_workers": 1,
        "val_persistent_workers": False,
        "val_pin_memory": False,
        "output_dir": args.output_dir,
        "vis_future_start": args.vis_future_start,
        "vis_future_end": vis_end,
        "sample_index": args.sample_index,
        "dataset_index": args.dataset_index,
        "target_valid_time": args.target_valid_time,
        "weight_source": args.weight_source,
        "history_window_frames": args.history_window_frames,
        "clear_threshold": args.clear_threshold,
        "light_upper": args.light_upper,
        "light_scale": args.light_scale,
        "save_residual_diagnostic": not args.no_residual_diagnostic,
    }
    run_demo(cfg, demo_cfg, files)


if __name__ == "__main__":
    main()
