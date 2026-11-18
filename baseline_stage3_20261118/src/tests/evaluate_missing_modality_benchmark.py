import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from omegaconf import DictConfig, OmegaConf
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure

from src.residual_diffusion_stage.metrics import DeterministicMetricAccumulator
from src.tests.inference_demo_transfer import load_checkpoint_weights, resolve_checkpoint_files
from src.trainer.rain_trainer_ts_next_frame_transfer import RainTSNextFrameTransferTrainer

MODALITY_COMBINATIONS: dict[str, tuple[bool, bool, bool]] = {
    "rad_sat_rain": (True, True, True),
    "rad_sat": (True, True, False),
    "rad_rain": (True, False, True),
    "sat_rain": (False, True, True),
    "rad": (True, False, False),
    "sat": (False, True, False),
    "rain": (False, False, True),
}


def apply_historical_modality_mask(
    context: torch.Tensor,
    available: tuple[bool, bool, bool],
    *,
    radar_channels: int,
    satellite_channels: int,
    rain_channels: int,
    mask_mode: str = "zero",
    radar_mask_token: torch.Tensor | None = None,
    satellite_mask_token: torch.Tensor | None = None,
    rain_mask_token: torch.Tensor | None = None,
    anchor_first_frame: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if context.ndim != 5:
        raise ValueError(f"context must have shape [B,C,T,H,W], got {tuple(context.shape)}")
    if mask_mode not in {"zero", "mask_token"}:
        raise ValueError(f"mask_mode must be 'zero' or 'mask_token', got {mask_mode}")

    total_channels = radar_channels + satellite_channels + rain_channels
    if int(context.shape[1]) != total_channels:
        raise ValueError(f"Expected {total_channels} channels, got {int(context.shape[1])}")

    b, _, t, h, w = context.shape
    orig_radar = context[:, :radar_channels]
    orig_satellite = context[:, radar_channels : radar_channels + satellite_channels]
    orig_rain = context[:, radar_channels + satellite_channels :]

    radar = orig_radar.clone()
    satellite = orig_satellite.clone()
    rain = orig_rain.clone()
    start_t = 1 if (anchor_first_frame and t > 1) else 0
    mask_t = t - start_t

    if not available[0] and mask_t > 0:
        if mask_mode == "mask_token" and radar_mask_token is not None:
            radar[:, :, start_t:] = radar_mask_token.to(device=context.device, dtype=context.dtype).expand(
                b, -1, mask_t, h, w
            )
        else:
            radar[:, :, start_t:] = 0.0
    if not available[1] and mask_t > 0:
        if mask_mode == "mask_token" and satellite_mask_token is not None:
            satellite[:, :, start_t:] = satellite_mask_token.to(device=context.device, dtype=context.dtype).expand(
                b, -1, mask_t, h, w
            )
        else:
            satellite[:, :, start_t:] = 0.0
    if not available[2] and mask_t > 0:
        if mask_mode == "mask_token" and rain_mask_token is not None:
            rain[:, :, start_t:] = rain_mask_token.to(device=context.device, dtype=context.dtype).expand(
                b, -1, mask_t, h, w
            )
        else:
            rain[:, :, start_t:] = 0.0

    if available[2]:
        last_valid_rain = orig_rain[:, :, -1:]
    elif anchor_first_frame and t > 0:
        last_valid_rain = orig_rain[:, :, :1]
    else:
        last_valid_rain = torch.zeros_like(orig_rain[:, :, -1:])

    masked_context = torch.cat([radar, satellite, rain], dim=1)
    return masked_context, last_valid_rain


def rollout_with_modality_combination(
    trainer: RainTSNextFrameTransferTrainer,
    context: torch.Tensor,
    target: dict[str, torch.Tensor],
    *,
    available: tuple[bool, bool, bool],
    context_time: torch.Tensor | None = None,
    future_time: torch.Tensor | None = None,
    use_gt_future_modalities: bool = False,
    mask_mode: str = "zero",
    anchor_first_frame: bool = False,
) -> dict[str, torch.Tensor]:
    model = trainer.accelerator.unwrap_model(trainer.model)
    frame_patch_size = trainer._resolve_frame_patch_size()
    total_future_frames = int(target["rain"].shape[2])
    if total_future_frames <= 0 or total_future_frames % frame_patch_size != 0:
        raise ValueError(
            f"total_future_frames={total_future_frames} must be positive and divisible by {frame_patch_size}"
        )

    masked_context, initial_rain_anchor = apply_historical_modality_mask(
        context,
        available,
        radar_channels=trainer.radar_c,
        satellite_channels=trainer.satellite_c,
        rain_channels=trainer.rain_c,
        mask_mode=mask_mode,
        radar_mask_token=getattr(model, "radar_mask_token", None),
        satellite_mask_token=getattr(model, "satellite_mask_token", None),
        rain_mask_token=getattr(model, "rain_mask_token", None),
        anchor_first_frame=anchor_first_frame,
    )

    context_cur = masked_context
    context_time_cur = context_time
    rain_anchor = initial_rain_anchor
    residual_enabled = trainer._resolve_rain_residual_cfg()[0]
    pred_radar: list[torch.Tensor] = []
    pred_satellite: list[torch.Tensor] = []
    pred_rain: list[torch.Tensor] = []

    for start in range(0, total_future_frames, frame_patch_size):
        end = start + frame_patch_size
        seed_block = context_cur[:, :, -frame_patch_size:]
        forward_kwargs: dict[str, object] = {
            "context_x": context_cur,
            "target_x": seed_block,
            "predict_frames": frame_patch_size,
            "strict_target_isolation": bool(trainer.train_cfg.strict_target_isolation),
            "return_modality_dict": True,
        }
        if torch.is_tensor(context_time_cur):
            forward_kwargs["context_time"] = context_time_cur
            forward_kwargs["target_time"] = context_time_cur[:, -frame_patch_size:]

        with trainer.accelerator.autocast():
            pred_block = trainer.model.forward_ar(**forward_kwargs)
        if not isinstance(pred_block, dict):
            raise TypeError("Expected modality dict from forward_ar")

        step_rain = pred_block["rain"]
        if residual_enabled:
            step_rain = trainer._apply_rain_residual_output(step_rain, rain_anchor)
        pred_radar.append(pred_block["radar"])
        pred_satellite.append(pred_block["satellite"])
        pred_rain.append(step_rain)

        next_radar = target["radar"][:, :, start:end] if use_gt_future_modalities else pred_block["radar"].detach()
        next_satellite = (
            target["satellite"][:, :, start:end] if use_gt_future_modalities else pred_block["satellite"].detach()
        )
        next_rain = step_rain.detach()
        rain_anchor = next_rain[:, :, -frame_patch_size:]
        next_hist = trainer._merge_modalities(next_radar, next_satellite, next_rain)
        context_cur = torch.cat([context_cur, next_hist], dim=2)
        if torch.is_tensor(context_time_cur) and future_time is not None:
            context_time_cur = torch.cat([context_time_cur, future_time[:, start:end]], dim=1)

    return {
        "radar": torch.cat(pred_radar, dim=2),
        "satellite": torch.cat(pred_satellite, dim=2),
        "rain": torch.cat(pred_rain, dim=2),
    }


def compute_auxiliary_psnr_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
) -> tuple[float, float, int]:
    b, c, t, h, w = pred.shape
    pred_bt = pred.detach().float().permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    target_bt = target.detach().float().permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    psnr_val = float(
        peak_signal_noise_ratio(pred_bt, target_bt, data_range=data_range, reduction="elementwise_mean").item()
    )
    ssim_val = float(
        structural_similarity_index_measure(
            pred_bt, target_bt, data_range=data_range, reduction="elementwise_mean"
        ).item()
    )
    return psnr_val, ssim_val, int(pred_bt.shape[0])


def evaluate_combinations_on_batches(
    trainer: RainTSNextFrameTransferTrainer,
    batches: list[dict[str, torch.Tensor]],
    *,
    combination_names: list[str],
    thresholds: list[float],
    use_gt_future_modalities: bool,
    mask_mode: str,
    include_persistence: bool = True,
    anchor_first_frame: bool = False,
) -> dict[str, dict[str, object]]:
    if not batches:
        raise ValueError("batches must not be empty")

    sample_target_frames = int(batches[0]["rain_future"].shape[1 if batches[0]["rain_future"].ndim == 4 else 2])
    results: dict[str, dict[str, object]] = {}

    for combo_name in combination_names:
        available = MODALITY_COMBINATIONS[combo_name]
        accumulator = DeterministicMetricAccumulator(
            thresholds=thresholds,
            lead_times=sample_target_frames,
            device=trainer.device,
        )
        radar_psnr_sum = 0.0
        radar_ssim_sum = 0.0
        sat_psnr_sum = 0.0
        sat_ssim_sum = 0.0
        frame_count = 0

        for batch in batches:
            context, target, context_time, target_time = trainer._prepare_val_inference_batch(batch)
            pred = rollout_with_modality_combination(
                trainer,
                context,
                target,
                available=available,
                context_time=context_time,
                future_time=target_time,
                use_gt_future_modalities=use_gt_future_modalities,
                mask_mode=mask_mode,
                anchor_first_frame=anchor_first_frame,
            )
            rain_pred = trainer._denormalize_rain_for_metrics(pred["rain"].detach())
            rain_target = trainer._denormalize_rain_for_metrics(target["rain"].detach())
            accumulator.update(rain_pred, rain_target)

            r_psnr, r_ssim, count = compute_auxiliary_psnr_ssim(pred["radar"], target["radar"])
            s_psnr, s_ssim, _ = compute_auxiliary_psnr_ssim(pred["satellite"], target["satellite"])
            radar_psnr_sum += r_psnr * count
            radar_ssim_sum += r_ssim * count
            sat_psnr_sum += s_psnr * count
            sat_ssim_sum += s_ssim * count
            frame_count += count

        combo_metrics = accumulator.result()
        combo_metrics["auxiliary"] = {
            "radar_psnr": radar_psnr_sum / max(frame_count, 1),
            "radar_ssim": radar_ssim_sum / max(frame_count, 1),
            "satellite_psnr": sat_psnr_sum / max(frame_count, 1),
            "satellite_ssim": sat_ssim_sum / max(frame_count, 1),
        }
        combo_metrics["available_modalities"] = {
            "radar": available[0],
            "satellite": available[1],
            "rain": available[2],
        }
        results[combo_name] = combo_metrics

    if include_persistence:
        persistence_acc = DeterministicMetricAccumulator(
            thresholds=thresholds,
            lead_times=sample_target_frames,
            device=trainer.device,
        )
        for batch in batches:
            context, target, _, _ = trainer._prepare_val_inference_batch(batch)
            last_rain = trainer._split_modalities(context)["rain"][:, :, -1:]
            pers_pred = last_rain.expand(-1, -1, int(target["rain"].shape[2]), -1, -1)
            rain_pred = trainer._denormalize_rain_for_metrics(pers_pred.detach())
            rain_target = trainer._denormalize_rain_for_metrics(target["rain"].detach())
            persistence_acc.update(rain_pred, rain_target)
        results["persistence"] = persistence_acc.result()

    return results


def _normalize_data_path(raw_path: str, repo_root: Path) -> str:
    s = str(raw_path)
    if s.startswith("/Data2/ZihanCao/RainDataset2025/"):
        rel = s[len("/Data2/ZihanCao/RainDataset2025/") :]
        return str(repo_root / "data2" / rel)
    p = Path(s)
    if not p.is_absolute():
        return str(repo_root / p)
    return s


def _prepare_eval_cfg(
    cfg: DictConfig,
    *,
    output_dir: str,
    batch_size: int | None,
    num_workers: int | None,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    resolved_val_dirs = [_normalize_data_path(str(d), repo_root) for d in cfg.dataset.val_inp_dirs]
    cfg.train.debug = True
    cfg.train.proj_dir = str(repo_root / output_dir) if not Path(output_dir).is_absolute() else output_dir
    cfg.train.log.log_with_time = False
    cfg.train.resume_path = None
    cfg.ema.beta = 0
    cfg.dataset.train_inp_dirs = resolved_val_dirs
    cfg.dataset.val_inp_dirs = resolved_val_dirs
    cfg.dataset.train.inp_dirs = resolved_val_dirs
    cfg.dataset.val.inp_dirs = resolved_val_dirs
    cfg.dataset.train.aug_enabled = False
    cfg.dataset.train.num_workers = 0
    cfg.dataset.train.persistent_workers = False
    cfg.dataset.train.pin_memory = False
    if batch_size is not None:
        cfg.dataset.val.batch_size = batch_size
    if num_workers is not None:
        cfg.dataset.val.num_workers = num_workers
    cfg.dataset.val.persistent_workers = False
    cfg.dataset.val.pin_memory = False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate complete and missing-modality benchmark combinations")
    parser.add_argument(
        "--checkpoint-path",
        default="/home/dongchenwang/RainPrediction/runs/next_frame_final/2026-10-12/20-17-52_stage1_next_frame_block",
    )
    parser.add_argument("--weight-source", choices=("auto", "ema", "model"), default="model")
    parser.add_argument("--max-batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--mask-mode", choices=("zero", "mask_token"), default="zero")
    parser.add_argument("--use-gt-future-modalities", action="store_true")
    parser.add_argument("--anchor-first-frame", action="store_true")
    parser.add_argument("--output-dir", default="vis/missing_modality_eval")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    files = resolve_checkpoint_files(args.checkpoint_path)
    cfg = OmegaConf.load(files.config)
    _prepare_eval_cfg(
        cfg,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    trainer = RainTSNextFrameTransferTrainer(cfg)
    selected_source, weight_file = load_checkpoint_weights(trainer, files, args.weight_source)
    trainer.model.eval()

    batches: list[dict[str, torch.Tensor]] = []
    iterator = iter(trainer.val_dataloader)
    for _ in range(args.max_batches):
        try:
            batches.append(next(iterator))
        except StopIteration:
            break

    thresholds = [float(v) for v in cfg.val.get("csi_thresholds", [0.1, 0.3, 0.5])]
    combo_names = list(MODALITY_COMBINATIONS.keys())
    with torch.no_grad():
        summary = evaluate_combinations_on_batches(
            trainer,
            batches,
            combination_names=combo_names,
            thresholds=thresholds,
            use_gt_future_modalities=bool(args.use_gt_future_modalities),
            mask_mode=args.mask_mode,
            include_persistence=True,
            anchor_first_frame=bool(args.anchor_first_frame),
        )

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = Path(args.output_dir)
    if not out_dir.is_absolute():
        out_dir = (repo_root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    mode_tag = "conditional_gt_future" if args.use_gt_future_modalities else "historical_only"
    anchor_tag = "_anchor1" if args.anchor_first_frame else ""
    out_file = out_dir / f"eval_{selected_source}_{mode_tag}_{args.mask_mode}{anchor_tag}_b{len(batches)}.json"
    payload = {
        "checkpoint": str(weight_file),
        "weight_source": selected_source,
        "global_step": int(trainer.global_step),
        "num_batches": len(batches),
        "num_sequences": sum(int(b["rain_future"].shape[0]) for b in batches),
        "use_gt_future_modalities": bool(args.use_gt_future_modalities),
        "mask_mode": args.mask_mode,
        "anchor_first_frame": bool(args.anchor_first_frame),
        "results": summary,
    }
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        f"Evaluated {payload['num_sequences']} sequences ({len(batches)} batches) | "
        f"source={selected_source} | mode={mode_tag} | mask={args.mask_mode}"
    )
    header = (
        f"{'Combination':<14} | {'CSI@0.1':>8} {'POD@0.1':>8} {'FAR@0.1':>8} {'HSS@0.1':>8} | "
        f"{'CSI@0.3':>8} | {'CSI@0.5':>8} | {'RadPSNR':>8} {'SatPSNR':>8}"
    )
    print(header)
    print("-" * len(header))
    for name, res in summary.items():
        overall = res["overall"]
        t01 = overall["threshold@0.1"]
        t03 = overall["threshold@0.3"]
        t05 = overall["threshold@0.5"]
        aux = res.get("auxiliary", {})
        rad_psnr = f"{aux['radar_psnr']:8.4f}" if "radar_psnr" in aux else "     N/A"
        sat_psnr = f"{aux['satellite_psnr']:8.4f}" if "satellite_psnr" in aux else "     N/A"
        print(
            f"{name:<14} | {t01['csi']:8.4f} {t01['pod']:8.4f} {t01['far']:8.4f} {t01['hss']:8.4f} | "
            f"{t03['csi']:8.4f} | {t05['csi']:8.4f} | {rad_psnr} {sat_psnr}"
        )
    print(f"Saved JSON report to: {out_file}")


if __name__ == "__main__":
    main()
