"""
Inference demo for RainTSNextFrameTrainer.

Usage:
    python src/tests/inference_demo.py
"""

import sys
from pathlib import Path

import torch
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.trainer.rain_trainer_ts_next_frame import RainTSNextFrameTrainer


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

    pred_for_vis = _slice_modalities_by_time(pred_target, start_idx=vis_start, end_idx=vis_end)
    target_for_vis = _slice_modalities_by_time(target, start_idx=vis_start, end_idx=vis_end)

    trainer._save_val_visualizations(
        context=context,
        pred_target=pred_for_vis,
        target=target_for_vis,
        output_prefix=f"inference_future_{vis_start}_{vis_end}",
    )

    print(
        "Visualization done: 三行按 radar/satellite/rain 排列，左侧 past，中间 pred，右侧 future gt。"
    )
    print(f"Visualized future range: [{vis_start}, {vis_end})")
    print(f"Saved to: {trainer.proj_dir}/val_viz/step_00000000/")


# ===== User-editable interface (edit here before main) =====
DEMO_CONFIG: dict = {
    "data_dirs": [
        "data2/litdata_train_2025/litdata_interval_30/202508",
    ],
    "context_frames": 4,
    "predict_frames": 12,
    "ema_path": "/home/rainpred/RainPrediction/runs/time_series_next_frame/2026-05-21_00-16-03_stage1_next_frame_block/ema",
    "output_dir": "vis/mse_with_radar_hugeweight_t2",
    # visualize [vis_future_start, vis_future_end), all requested future frames can be shown by setting 0 -> predict_frames
    "vis_future_start": 0,
    "vis_future_end": 12,
}


if __name__ == "__main__":
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    config_dir = str(Path(__file__).resolve().parents[1] / "config" / "ts_rain_train")
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="rain_trainer_ts_next_frame")

    run_demo(cfg, DEMO_CONFIG)
