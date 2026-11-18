"""Fast A/B verification for solving light-rain residual accumulation via
un-detached rain_anchor + clamp_output=True during rollout training.

All outputs are saved strictly inside `src/light_rain_residual_check/outputs/`.
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.tests.inference_demo import _build_demo_cfg_for_trainer
from src.tests.inference_demo_transfer import (
    load_checkpoint_weights,
    resolve_checkpoint_files,
    select_inference_window,
)
from src.tests.inference_demo_transfer_light_rain_suppression import (
    _binary_frame_metrics,
    rollout_predict_with_light_rain_suppression,
)
from src.trainer.rain_trainer_ts_next_frame_transfer import RainTSNextFrameTransferTrainer


def rollout_sliding_window_eval(
    trainer: RainTSNextFrameTransferTrainer,
    context: torch.Tensor,
    target: dict[str, torch.Tensor],
    *,
    context_time: torch.Tensor | None,
    future_time: torch.Tensor | None,
    history_window_frames: int,
    clamp_output: bool,
) -> dict[str, torch.Tensor]:
    """12-frame sliding-window rollout without any ad-hoc light-rain suppression."""
    model = trainer.accelerator.unwrap_model(trainer.model)
    frame_patch_size = int(model.frame_patch_size)
    total_future_frames = int(target["rain"].shape[2])

    context_cur = context[:, :, -history_window_frames:]
    context_time_cur = None if context_time is None else context_time[:, -history_window_frames:]
    rain_predictions: list[torch.Tensor] = []
    rain_deltas: list[torch.Tensor] = []

    for start in range(0, total_future_frames, frame_patch_size):
        end = start + frame_patch_size
        seed = context_cur[:, :, -frame_patch_size:]
        rain_seed = seed[:, -trainer.rain_c :]
        rain_delta = trainer.model.forward_ar(
            context_x=context_cur,
            target_x=seed,
            predict_frames=frame_patch_size,
            strict_target_isolation=bool(trainer.train_cfg.strict_target_isolation),
            return_modality_dict=False,
            context_time=context_time_cur,
            target_time=None if context_time_cur is None else context_time_cur[:, -frame_patch_size:],
        )
        rain = rain_seed + rain_delta
        if clamp_output:
            rain = rain.clamp_min(0.0)

        rain_predictions.append(rain)
        rain_deltas.append(rain_delta)

        if end >= total_future_frames:
            continue

        history = trainer._merge_modalities(
            target["radar"][:, :, start:end],
            target["satellite"][:, :, start:end],
            rain.detach(),
        )
        context_cur = torch.cat([context_cur, history], dim=2)[:, :, -history_window_frames:]
        if context_time_cur is not None and future_time is not None:
            context_time_cur = torch.cat([context_time_cur, future_time[:, start:end]], dim=1)
            context_time_cur = context_time_cur[:, -history_window_frames:]

    return {
        "rain": torch.cat(rain_predictions, dim=2),
        "rain_delta": torch.cat(rain_deltas, dim=2),
    }


def train_rollout_step(
    trainer: RainTSNextFrameTransferTrainer,
    context: torch.Tensor,
    target: dict[str, torch.Tensor],
    *,
    context_time: torch.Tensor | None,
    future_time: torch.Tensor | None,
    rollout_frames: int,
    history_window_frames: int,
    detach_rain_anchor: bool,
    clamp_output: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Single training step comparing detached vs. un-detached rain_anchor."""
    model = trainer.accelerator.unwrap_model(trainer.model)
    frame_patch_size = int(model.frame_patch_size)

    # 1. Teacher-forced next-frame loss on first 4 future frames (keeps single-step accuracy anchored)
    tf_frames = min(4, rollout_frames)
    full_radar = torch.cat([context[:, : trainer.radar_c], target["radar"][:, :, :tf_frames]], dim=2)
    full_sat = torch.cat(
        [
            context[:, trainer.radar_c : trainer.radar_c + trainer.satellite_c],
            target["satellite"][:, :, :tf_frames],
        ],
        dim=2,
    )
    full_rain = torch.cat([context[:, -trainer.rain_c :], target["rain"][:, :, :tf_frames]], dim=2)
    full_tensor = trainer._merge_modalities(full_radar, full_sat, full_rain)
    full_time = torch.cat([context_time, future_time[:, :tf_frames]], dim=1) if context_time is not None else None

    seq_context = full_tensor[:, :, :frame_patch_size]
    seed_tensor = full_tensor[:, :, :-frame_patch_size]
    gt_rain_seq = full_rain[:, :, frame_patch_size:]
    ref_rain_seq = full_rain[:, :, :-frame_patch_size]

    with trainer.accelerator.autocast():
        tf_delta = trainer.model.forward_ar(
            context_x=seq_context,
            target_x=seed_tensor,
            predict_frames=int(seed_tensor.shape[2]),
            strict_target_isolation=False,
            return_modality_dict=False,
            context_time=None if full_time is None else full_time[:, :frame_patch_size],
            target_time=None if full_time is None else full_time[:, :-frame_patch_size],
        )
    tf_pred_rain = ref_rain_seq + tf_delta
    if clamp_output:
        tf_pred_rain = tf_pred_rain.clamp_min(0.0)
    loss_tf = F.mse_loss(tf_pred_rain, gt_rain_seq)

    # 2. Multi-step autoregressive rollout branch
    context_cur = context[:, :, -history_window_frames:]
    context_time_cur = None if context_time is None else context_time[:, -history_window_frames:]
    rain_anchor = context_cur[:, -trainer.rain_c :, -frame_patch_size:]

    pred_rain_steps: list[torch.Tensor] = []
    pred_delta_steps: list[torch.Tensor] = []

    for start in range(0, rollout_frames, frame_patch_size):
        end = start + frame_patch_size
        seed = context_cur[:, :, -frame_patch_size:]
        with trainer.accelerator.autocast():
            raw_rain_delta = trainer.model.forward_ar(
                context_x=context_cur,
                target_x=seed,
                predict_frames=frame_patch_size,
                strict_target_isolation=bool(trainer.train_cfg.strict_target_isolation),
                return_modality_dict=False,
                context_time=context_time_cur,
                target_time=None if context_time_cur is None else context_time_cur[:, -frame_patch_size:],
            )
        step_rain = rain_anchor + raw_rain_delta
        if clamp_output:
            step_rain = step_rain.clamp_min(0.0)

        pred_rain_steps.append(step_rain)
        pred_delta_steps.append(raw_rain_delta)

        if end >= rollout_frames:
            continue

        # Key difference:
        # - context_cur ALWAYS uses step_rain.detach() (zero extra encoder BPTT memory)
        # - rain_anchor uses step_rain.detach() in Control, and un-detached step_rain in Method 1!
        rain_anchor = step_rain.detach() if detach_rain_anchor else step_rain
        history_for_encoder = trainer._merge_modalities(
            target["radar"][:, :, start:end],
            target["satellite"][:, :, start:end],
            step_rain.detach(),
        )
        context_cur = torch.cat([context_cur, history_for_encoder], dim=2)[:, :, -history_window_frames:]
        if context_time_cur is not None and future_time is not None:
            context_time_cur = torch.cat([context_time_cur, future_time[:, start:end]], dim=1)
            context_time_cur = context_time_cur[:, -history_window_frames:]

    rollout_pred = torch.cat(pred_rain_steps, dim=2)
    rollout_gt = target["rain"][:, :, :rollout_frames]
    loss_rollout = F.mse_loss(rollout_pred, rollout_gt)

    total_loss = loss_tf + 0.5 * loss_rollout
    all_deltas = torch.cat(pred_delta_steps, dim=2).detach()
    dry_mask = rollout_gt < 0.01
    dry_delta_mean = float(all_deltas[dry_mask].mean().item()) if dry_mask.any() else 0.0
    return total_loss, {
        "loss_tf": float(loss_tf.item()),
        "loss_rollout": float(loss_rollout.item()),
        "dry_delta_mean": dry_delta_mean,
    }


def summarize_12frame_metrics(
    pred_rain: torch.Tensor,
    pred_delta: torch.Tensor,
    target_rain: torch.Tensor,
) -> dict[str, object]:
    thresholds = (0.01, 0.05, 0.10)
    per_thr: dict[str, dict[str, object]] = {}
    for thr in thresholds:
        frames = _binary_frame_metrics(pred_rain, target_rain, thr)
        mean_csi = sum(f["csi"] for f in frames) / len(frames)
        late_csi = sum(f["csi"] for f in frames[6:]) / len(frames[6:])
        late_area = sum(f["area"] for f in frames[6:]) / len(frames[6:])
        late_far = sum(f["far"] for f in frames[6:]) / len(frames[6:])
        per_thr[f"{thr:.2f}"] = {
            "mean_csi_12f": mean_csi,
            "late_csi_f6_11": late_csi,
            "frame0_area": frames[0]["area"],
            "frame11_area": frames[-1]["area"],
            "late_area_f6_11": late_area,
            "late_far_f6_11": late_far,
            "frames": frames,
        }

    dry_mask = target_rain < 0.01
    dry_delta_mean = float(pred_delta[dry_mask].mean().item()) if dry_mask.any() else 0.0
    dry_pos_delta_ratio = float((pred_delta[dry_mask] > 0).float().mean().item()) if dry_mask.any() else 0.0
    return {
        "dry_delta_mean": dry_delta_mean,
        "dry_pos_delta_ratio": dry_pos_delta_ratio,
        "rmse_12f": float(torch.sqrt(F.mse_loss(pred_rain, target_rain)).item()),
        "thresholds": per_thr,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast A/B check for un-detached rain_anchor")
    parser.add_argument(
        "--checkpoint-path",
        default="/home/dongchenwang/RainPrediction/runs/next_frame_final/2026-10-12/20-17-52_stage1_next_frame_block/checkpoints/step_00099000",
    )
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--lr", type=float, default=5.0e-5)
    parser.add_argument("--rollout-frames", type=int, default=6)
    parser.add_argument("--output-dir", default="src/light_rain_residual_check/outputs")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    files = resolve_checkpoint_files(args.checkpoint_path)
    cfg = OmegaConf.load(files.config)
    demo_cfg = {
        "data_dirs": list(cfg.dataset.val_inp_dirs),
        "context_frames": 4,
        "predict_frames": 12,
        "val_batch_size": 1,
        "val_num_workers": 0,
        "val_persistent_workers": False,
        "val_pin_memory": False,
        "output_dir": str(out_dir / "trainer_work"),
    }
    _build_demo_cfg_for_trainer(cfg, demo_cfg)
    cfg.train.resume_path = None
    cfg.ema.beta = 0

    trainer = RainTSNextFrameTransferTrainer(cfg)
    source, weight_file = load_checkpoint_weights(trainer, files, "ema")
    print(f"Loaded {source} weights from {weight_file}")

    unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
    init_rain_decoder_state = copy.deepcopy(unwrapped_model.rain_decoder.state_dict())

    # 1. Load the exact benchmark evaluation sample (dataset_index=45, 2025-08-04 23:00:00)
    eval_batch_raw, eval_sel = select_inference_window(
        trainer.val_dataset,
        dataset_index=None,
        target_valid_time="2025-08-04 23:00:00",
    )
    eval_context, eval_target, eval_ctime, eval_ftime = trainer._prepare_val_inference_batch(eval_batch_raw)
    target_rain_cpu = eval_target["rain"].detach().float().cpu()

    # 2. Load 16 disjoint training windows (from indices >= 200, weeks away from index 45)
    total_len = len(trainer.val_dataset.times_pairs)
    train_indices = [200 + i * ((total_len - 250) // 16) for i in range(16)]
    print(f"Caching {len(train_indices)} disjoint windows for fast A/B check: {train_indices}")
    cached_train_batches = []
    for idx in train_indices:
        b_raw, _ = select_inference_window(trainer.val_dataset, dataset_index=idx, target_valid_time=None)
        cached_train_batches.append(trainer._prepare_val_inference_batch(b_raw))

    results: dict[str, object] = {}

    # Evaluate Variant 0: Raw Initial Checkpoint
    trainer.model.eval()
    with torch.no_grad(), trainer.accelerator.autocast():
        raw_out = rollout_sliding_window_eval(
            trainer,
            eval_context,
            eval_target,
            context_time=eval_ctime,
            future_time=eval_ftime,
            history_window_frames=7,
            clamp_output=False,
        )
        sup_out, _ = rollout_predict_with_light_rain_suppression(
            trainer,
            eval_context,
            eval_target,
            context_time=eval_ctime,
            future_time=eval_ftime,
            history_window_frames=7,
            clear_threshold=0.02,
            light_upper=0.10,
            light_scale=0.90,
        )
    results["0_init_raw"] = summarize_12frame_metrics(
        raw_out["rain"].detach().float().cpu(),
        raw_out["rain_delta"].detach().float().cpu(),
        target_rain_cpu,
    )
    results["1_init_post_suppressed"] = summarize_12frame_metrics(
        sup_out["rain"].detach().float().cpu(),
        sup_out["model_rain_delta"].detach().float().cpu(),
        target_rain_cpu,
    )

    # Run A/B training variants (updating rain_decoder with identical seed, lr, and batch order)
    variants = [
        ("2_control_detached_noclamp", True, False),
        ("3_method1_undetached_noclamp", False, False),
        ("4_method1_undetached_clamp", False, True),
    ]

    for name, detach_anchor, clamp_out in variants:
        torch.manual_seed(2025)
        unwrapped_model.rain_decoder.load_state_dict(copy.deepcopy(init_rain_decoder_state))
        for p in unwrapped_model.parameters():
            p.requires_grad_(False)
        for p in unwrapped_model.rain_decoder.parameters():
            p.requires_grad_(True)

        optimizer = torch.optim.AdamW(unwrapped_model.rain_decoder.parameters(), lr=args.lr, weight_decay=0.0)
        unwrapped_model.eval()
        unwrapped_model.rain_decoder.train()

        for step in range(args.steps):
            c_ctx, c_tgt, c_ctime, c_ftime = cached_train_batches[step % len(cached_train_batches)]
            optimizer.zero_grad(set_to_none=True)
            loss, step_info = train_rollout_step(
                trainer,
                c_ctx,
                c_tgt,
                context_time=c_ctime,
                future_time=c_ftime,
                rollout_frames=args.rollout_frames,
                history_window_frames=7,
                detach_rain_anchor=detach_anchor,
                clamp_output=clamp_out,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unwrapped_model.rain_decoder.parameters(), 1.0)
            optimizer.step()
            if (step + 1) % 40 == 0:
                print(
                    f"[{name}] step {step+1:3d}/{args.steps} | "
                    f"loss_rollout={step_info['loss_rollout']:.6f} | "
                    f"dry_delta_mean={step_info['dry_delta_mean']:+.6f}"
                )

        unwrapped_model.eval()
        with torch.no_grad(), trainer.accelerator.autocast():
            eval_out = rollout_sliding_window_eval(
                trainer,
                eval_context,
                eval_target,
                context_time=eval_ctime,
                future_time=eval_ftime,
                history_window_frames=7,
                clamp_output=clamp_out,
            )
        results[name] = summarize_12frame_metrics(
            eval_out["rain"].detach().float().cpu(),
            eval_out["rain_delta"].detach().float().cpu(),
            target_rain_cpu,
        )

    report_path = out_dir / "ab_verification_report.json"
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "=" * 105)
    print(
        f"{'Variant (12-frame Raw Rollout unless noted)':32s} | "
        f"{'DryDelta':>9s} | {'Area@0.01 (f0->f11)':>20s} | "
        f"{'CSI@0.01 (12f/late)':>19s} | {'CSI@0.05':>8s} | {'CSI@0.10':>8s}"
    )
    print("-" * 105)
    for k, v in results.items():
        t01 = v["thresholds"]["0.01"]
        t05 = v["thresholds"]["0.05"]
        t10 = v["thresholds"]["0.10"]
        print(
            f"{k:32s} | {v['dry_delta_mean']:+9.5f} | "
            f"{t01['frame0_area']*100:5.2f}% -> {t01['frame11_area']*100:5.2f}%     | "
            f"{t01['mean_csi_12f']:7.4f} / {t01['late_csi_f6_11']:7.4f}   | "
            f"{t05['mean_csi_12f']:8.4f} | {t10['mean_csi_12f']:8.4f}"
        )
    print("=" * 105)
    print(f"Saved detailed JSON report to: {report_path}")


if __name__ == "__main__":
    main()
