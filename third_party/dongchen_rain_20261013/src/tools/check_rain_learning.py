import argparse
import json
from pathlib import Path

import hydra
import numpy as np
import torch
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf

from src.dataset.rain_ts_litdata import RainTimeSeriesDataset
from src.tools.diagnose_rain_transitions import region_metrics
from src.trainer.rain_trainer_ts_next_frame import RainTSNextFrameTrainer


class DiagnosticTrainer(RainTSNextFrameTrainer):
    def __init__(self, cfg: DictConfig, checkpoint: Path) -> None:
        self.cfg = cfg
        self.train_cfg = cfg.train
        self.val_cfg = cfg.val
        self.dataset_cfg = cfg.dataset
        self.accelerator = Accelerator(mixed_precision="bf16")
        self.device = self.accelerator.device
        self.model = hydra.utils.instantiate(cfg.rain_prediction_model)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not bool(payload["initted"]):
            raise ValueError("EMA is not initialized.")
        self.global_step = int(payload["step"])
        state = {key.removeprefix("ema_model."): value for key, value in payload.items() if key.startswith("ema_model.")}
        if not state:
            raise ValueError("EMA model weights are absent.")
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self.radar_c = self.model.radar_out_channels
        self.satellite_c = self.model.satellite_out_channels
        self.rain_c = self.model.rain_out_channels
        if cfg.dataset.get("modality_zero_centering", False):
            raise ValueError("This audit expects uncentered rain units.")


def forecast_metrics(prediction: torch.Tensor, target: torch.Tensor, anchor: torch.Tensor) -> dict:
    prediction = prediction.float()
    target = target.float()
    anchor = anchor.float()
    previous_pred = torch.cat([anchor, prediction[:, :, :-1]], dim=2)
    previous_gt = torch.cat([anchor, target[:, :, :-1]], dim=2)
    delta_pred = prediction - previous_pred
    delta_gt = target - previous_gt
    reduce_dims = (0, 1, 3, 4)
    target_delta_energy = delta_gt.square().mean(dim=reduce_dims)
    return {
        "mse_by_lead": (prediction - target).square().mean(dim=reduce_dims).tolist(),
        "persistence_mse_by_lead": (anchor - target).square().mean(dim=reduce_dims).tolist(),
        "delta_mse_by_lead": (delta_pred - delta_gt).square().mean(dim=reduce_dims).tolist(),
        "predicted_delta_energy_by_lead": delta_pred.square().mean(dim=reduce_dims).tolist(),
        "target_delta_energy_by_lead": target_delta_energy.tolist(),
        "delta_cross_by_lead": (delta_pred * delta_gt).mean(dim=reduce_dims).tolist(),
        "mean_by_lead": prediction.mean(dim=reduce_dims).tolist(),
        "target_mean_by_lead": target.mean(dim=reduce_dims).tolist(),
    }


def gradient_vector(loss: torch.Tensor, parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([
        grad.detach().float().reshape(-1).cpu() if grad is not None else torch.zeros(parameter.numel())
        for grad, parameter in zip(grads, parameters, strict=True)
    ])


def sequence_mean(values: torch.Tensor, aux: dict) -> torch.Tensor:
    context_frames = int(aux["sequence_context_frames"])
    context_weight = float(aux["sequence_context_weight"])
    future_weight = float(aux["sequence_future_weight"])
    return (
        values[:, :, :context_frames].mean() * context_weight
        + values[:, :, context_frames:].mean() * future_weight
    ) / (context_weight + future_weight)


def gradient_checks(trainer: DiagnosticTrainer, batch: dict[str, torch.Tensor]) -> dict:
    shared = [parameter for name, parameter in trainer.model.named_parameters() if "_decoder." not in name]
    context, seed, target, aux = trainer._build_next_pred_batch(batch, apply_missing_modality=False)
    with trainer.accelerator.autocast():
        pred = trainer.model.forward_ar(
            context_x=context, target_x=seed, predict_frames=int(aux["target_frames"]),
            strict_target_isolation=bool(trainer.train_cfg.strict_target_isolation),
            return_modality_dict=True, context_time=aux["context_time"], target_time=aux["target_seed_time"],
        )
        physical = dict(pred)
        reference = aux["rain_residual_ref"].to(pred["rain"].dtype)
        physical["rain"] = trainer._apply_rain_residual_output(pred["rain"], reference)
        components = {
            name: float(trainer.train_cfg.loss_weights[name]) * sequence_mean(
                (physical[name].float() - target[name]).square(), aux
            ) for name in ("radar", "satellite", "rain")
        }
        current_total, logs = trainer._next_prediction_loss(pred, target, aux)
        components["temporal_diff"] = current_total - sum(components.values())
        components["historical_residual_delta"] = float(trainer.train_cfg.loss.rain_residual.get("delta_weight", 1.0)) * sequence_mean(
            (pred["rain"].float() - (target["rain"] - reference)).square(), aux
        )
        if not torch.allclose(components["temporal_diff"].detach(), logs["loss/temporal_diff"], atol=1e-6, rtol=1e-4):
            raise ValueError("Temporal loss decomposition does not match original loss logs.")
        vectors = {name: gradient_vector(loss, shared) for name, loss in components.items()}
        rain_total = vectors["rain"] + vectors["temporal_diff"] + vectors["historical_residual_delta"]
        result = {
            "teacher_forced_weighted_losses": {name: float(value.detach()) for name, value in components.items()},
            "shared_gradient_norm": {name: float(value.norm()) for name, value in vectors.items()},
            "rain_combined_shared_gradient_norm": float(rain_total.norm()),
            "shared_radar_rain_cosine": float(torch.nn.functional.cosine_similarity(vectors["radar"], rain_total, dim=0)),
            "rain_mse_vs_temporal_cosine": float(torch.nn.functional.cosine_similarity(vectors["rain"], vectors["temporal_diff"], dim=0)),
        }
    del pred, physical, current_total, logs, components, vectors, rain_total, context, seed, target, aux

    context, target, context_time, future_time = trainer._prepare_val_inference_batch(batch)
    frames = int(trainer.train_cfg.next_pred.rollout_branch.rollout_frames)
    target = {name: value[:, :, :frames] for name, value in target.items()}
    future_time = future_time[:, :frames]
    with trainer.accelerator.autocast():
        pred = trainer._rollout_predict_with_settings(
            context=context, total_future_frames=frames, mode="frame", rollout_block_size=1,
            detach_history=True, context_time=context_time, future_time=future_time,
            future_modalities=target, use_gt_future_modalities=True, return_rain_delta=True,
        )
        previous_gt = torch.cat([context[:, -trainer.rain_c:, -1:], target["rain"][:, :, :-1]], dim=2)
        delta_gt = target["rain"] - previous_gt
        rain = pred["rain"].float()
        raw = pred["rain_delta"].float()
        weight = float(trainer.train_cfg.next_pred.rollout_branch.weight)
        delta_weight = float(trainer.train_cfg.next_pred.rollout_branch.delta_loss.weight)
        absolute_loss = (rain - target["rain"]).square().mean()
        inner_total, logs = trainer._next_prediction_loss(
            pred, target,
            {"rain_residual_already_applied": 1, "temporal_diff_anchor": trainer._split_modalities(context[:, :, -1:])},
            loss_weight_override={"radar": 0.0, "satellite": 0.0, "rain": 1.0},
        )
        components = {
            "absolute": weight * absolute_loss,
            "temporal_diff": weight * (inner_total - absolute_loss),
            "raw_delta": weight * delta_weight * (raw - delta_gt).square().mean(),
        }
        vectors = {name: gradient_vector(loss, shared) for name, loss in components.items()}
        result["rollout"] = {
            "weighted_losses": {name: float(value.detach()) for name, value in components.items()},
            "shared_gradient_norm": {name: float(value.norm()) for name, value in vectors.items()},
            "absolute_vs_delta_shared_gradient_cosine": float(torch.nn.functional.cosine_similarity(vectors["absolute"], vectors["raw_delta"], dim=0)),
            "absolute_vs_temporal_shared_gradient_cosine": float(torch.nn.functional.cosine_similarity(vectors["absolute"], vectors["temporal_diff"], dim=0)),
            "raw_output_equals_absolute_rain": bool(torch.equal(raw, rain)),
        }
    return result


def summarize(records: list[dict]) -> dict:
    result = {}
    for mode in records[0]["modes"]:
        result[mode] = {}
        for modality in ("rain", "radar"):
            metrics = [row["modes"][mode][modality] for row in records]
            pooled = {key: np.mean([item[key] for item in metrics], axis=0) for key in metrics[0]}
            denominator = pooled["persistence_mse_by_lead"]
            pooled["mse_over_persistence_by_lead"] = np.divide(
                pooled["mse_by_lead"], denominator, out=np.full_like(denominator, np.nan), where=denominator > 0
            )
            signal = np.sqrt(pooled["predicted_delta_energy_by_lead"] * pooled["target_delta_energy_by_lead"])
            pooled["temporal_delta_cosine_by_lead"] = np.divide(
                pooled["delta_cross_by_lead"], signal, out=np.full_like(signal, np.nan), where=signal > 0
            )
            result[mode][modality] = {key: [float(v) if np.isfinite(v) else None for v in value] for key, value in pooled.items()}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--month-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-month", type=int, default=8)
    parser.add_argument("--gradient-samples-per-month", type=int, default=2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(2025)
    cfg = OmegaConf.load(args.config)
    trainer = DiagnosticTrainer(cfg, args.checkpoint)
    print(f"Strictly loaded model on {trainer.device}", flush=True)
    records = []
    n_past = int(cfg.dataset.n_past)
    n_future = int(cfg.dataset.n_futures)
    modes = ("normal", "gt_rain_history", "frozen_rain_history", "frozen_radar", "all_autoregressive")
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "config": str(args.config.resolve()),
        "ema_step": trainer.global_step,
        "weight_source": "ema_model only; online_model weights excluded",
        "rain_residual_enabled": bool(cfg.train.loss.rain_residual.enabled),
        "rollout_delta_enabled": bool(cfg.train.next_pred.rollout_branch.delta_loss.enabled),
        "frame_patch_size": int(cfg.rain_prediction_model.frame_patch_size),
        "saved_training_filter_enabled": bool(cfg.dataset.train.rain_ratio_filter_enabled),
        "sampling": "Uniformly spaced valid windows, no rain/delta filtering and no augmentation.",
        "gradient_scope": "Eval mode, dropout disabled, configured teacher-forced modality weights; historical extra residual-delta term included; separate rollout absolute/temporal/delta terms; no optimizer.",
        "modes": {
            "normal": "GT radar/satellite feedback, predicted rain feedback, matching saved validation settings.",
            "gt_rain_history": "Same weather inputs, GT rain feedback after each temporal patch.",
            "frozen_rain_history": "Same weather inputs, last observed rain repeated as future feedback; initial rain history unchanged.",
            "frozen_radar": "All input radar times fixed to last observed radar; same GT satellite feedback and predicted rain feedback.",
            "all_autoregressive": "All future modalities use prediction feedback.",
        },
        "records": records,
    }
    for month in args.month_dirs:
        dataset = RainTimeSeriesDataset(
            inp_dirs=[str(month)], n_past=n_past, n_futures=n_future,
            time_interval=int(cfg.dataset.time_interval), img_resize=int(cfg.dataset.img_size),
            stack_data=True, is_cycled=False, rain_ratio_filter_enabled=False,
            aug_enabled=False, batching_method="per_stream", iterate_over_all=True,
        )
        indices = np.linspace(0, len(dataset) - 1, args.samples_per_month, dtype=int).tolist()
        gradient_positions = set(np.linspace(0, len(indices) - 1, args.gradient_samples_per_month, dtype=int).tolist())
        for position, index in enumerate(indices):
            item = dataset[index]
            batch = {key: value.unsqueeze(0) for key, value in item.items() if isinstance(value, torch.Tensor)}
            context, target, past_time, future_time = trainer._prepare_val_inference_batch(batch)
            anchors = trainer._split_modalities(context[:, :, -1:])
            row = {"month": month.name, "window_index": index, "times": [[str(t) for t in times] for times in dataset.times_pairs[index]], "modes": {}}
            for mode in modes:
                modified_context = context.clone()
                weather = {key: value for key, value in target.items()}
                feedback_rain = None
                if mode == "gt_rain_history":
                    feedback_rain = target["rain"]
                elif mode == "frozen_rain_history":
                    feedback_rain = anchors["rain"].expand_as(target["rain"])
                elif mode == "frozen_radar":
                    modified_context[:, :trainer.radar_c] = anchors["radar"]
                    weather["radar"] = anchors["radar"].expand_as(target["radar"])
                with torch.no_grad(), trainer.accelerator.autocast():
                    pred = trainer._rollout_predict_with_settings(
                        context=modified_context, total_future_frames=n_future,
                        mode="frame", rollout_block_size=1, detach_history=True,
                        context_time=past_time, future_time=future_time,
                        future_modalities=weather, use_gt_future_modalities=mode != "all_autoregressive",
                        future_rain=feedback_rain, use_gt_future_rain=feedback_rain is not None,
                    )
                rain_np = pred["rain"].float().cpu().numpy()[:, 0]
                target_np = target["rain"].cpu().numpy()[:, 0]
                anchor_np = anchors["rain"].cpu().numpy()[:, 0, 0]
                previous_gt = np.concatenate([anchor_np[:, None], target_np[:, :-1]], axis=1)
                previous_pred = np.concatenate([anchor_np[:, None], rain_np[:, :-1]], axis=1)
                row["modes"][mode] = {
                    name: forecast_metrics(pred[name], target[name], anchors[name]) for name in ("radar", "rain")
                }
                row["modes"][mode]["rain_regions"] = [
                    region_metrics(previous_gt[:, lead], target_np[:, lead], rain_np[:, lead], previous_pred[:, lead], 0.01)
                    for lead in range(n_future)
                ]
                if mode == "normal":
                    np.savez_compressed(args.output_dir / f"{month.name}_window{index}.npz", prediction=rain_np, target=target_np, anchor=anchor_np)
                del pred
            if position in gradient_positions:
                row["gradients"] = gradient_checks(trainer, batch)
            records.append(row)
            (args.output_dir / "report.partial.json").write_text(json.dumps(report, indent=2, allow_nan=False))
            print(f"Completed {month.name} window {position + 1}/{len(indices)} index={index}", flush=True)
    report["summary_by_month"] = {
        month.name: summarize([row for row in records if row["month"] == month.name]) for month in args.month_dirs
    }
    report["peak_cuda_memory_gb"] = torch.cuda.max_memory_allocated() / 1024 ** 3
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Report saved: {args.output_dir / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
