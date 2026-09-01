import argparse
import json
import math
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import nn

from src.residual_diffusion_stage.losses import (
    ChannelMoments,
    diffusion_residual_loss,
    residual_autoencoder_loss,
)
from src.residual_diffusion_stage.models import ConditionalDiffusion, LatentNormalizer, ResidualAutoencoder
from src.trainer.rain_trainer_ts_next_frame import RainTSNextFrameTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAGE_CONFIG = Path(__file__).with_name("config.yaml")
DEFAULT_BASE_CONFIG = "rain_trainer_ts_next_frame_cross_local_finetune"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the isolated residual AE or latent diffusion stage")
    parser.add_argument("--stage", choices=("ae", "diffusion"), required=True)
    parser.add_argument("--coarse-checkpoint", type=Path, required=True)
    parser.add_argument("--ae-checkpoint", type=Path, default=None)
    parser.add_argument("--diffusion-checkpoint", type=Path, default=None)
    parser.add_argument("--base-config-name", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--stage-config", type=Path, default=DEFAULT_STAGE_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    return parser.parse_args()


def _resolve_coarse_checkpoint(path: Path, save_every: int) -> Path:
    path = path.expanduser().resolve()
    if (path / "model.safetensors").is_file() or (path / "pytorch_model.bin").is_file():
        return path
    meta_path = path / "meta.json"
    if meta_path.is_file():
        global_step = int(json.loads(meta_path.read_text()).get("global_step", 0))
        automatic = path.parent / "checkpoints" / f"checkpoint_{global_step // save_every}"
        if (automatic / "model.safetensors").is_file():
            return automatic
    raise FileNotFoundError(
        f"No model weights found in {path}. Pass the Accelerate directory containing model.safetensors; "
        "for the completed coarse run this is usually .../checkpoints/checkpoint_12."
    )


def _load_base_config(config_name: str) -> DictConfig:
    config_dir = PROJECT_ROOT / "src" / "config" / "ts_rain_train"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name=config_name)


def _build_coarse_trainer(
    config_name: str,
    coarse_checkpoint: Path,
    output_dir: Path,
    batch_size: int,
    accumulation_steps: int,
) -> RainTSNextFrameTrainer:
    cfg = _load_base_config(config_name)
    resolved_checkpoint = _resolve_coarse_checkpoint(coarse_checkpoint, int(cfg.train.save_every))
    with open_dict(cfg):
        cfg.train.init_model_path = str(resolved_checkpoint)
        cfg.train.resume_path = None
        cfg.train.proj_dir = str(output_dir / "coarse_runtime")
        cfg.train.debug = True
        cfg.train.strict_target_isolation = True
        cfg.train.log.log_with_time = False
        cfg.train.log.run_comment = ""
        cfg.train.gan.enabled = False
        cfg.train.next_pred.rollout_branch.use_gt_future_modalities = False
        cfg.val.rollout_use_gt_future_modalities = False
        cfg.val.save_visuals = False
        cfg.dataset.train.batch_size = batch_size
        cfg.dataset.val.batch_size = batch_size
        cfg.dataset.train.persistent_workers = False
        cfg.dataset.val.persistent_workers = False
        cfg.ema.beta = 0.0
        cfg.accelerator.gradient_accumulation_plugin.num_steps = accumulation_steps
        cfg.accelerator.project_config.project_dir = str(output_dir / "coarse_runtime")
        cfg.accelerator.project_config.logging_dir = str(output_dir / "coarse_runtime" / "tensorboard")
    trainer = RainTSNextFrameTrainer(cfg)
    trainer.model.eval()
    trainer.model.requires_grad_(False)
    return trainer


@torch.no_grad()
def _make_residual_batch(
    trainer: RainTSNextFrameTrainer,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    context, target, context_time, target_time = trainer._prepare_val_inference_batch(batch)
    with trainer.accelerator.autocast():
        coarse = trainer._rollout_predict_with_settings(
            context=context,
            total_future_frames=int(target["rain"].shape[2]),
            mode="frame",
            rollout_block_size=1,
            detach_history=True,
            context_time=context_time,
            future_time=target_time,
            future_modalities=None,
            use_gt_future_modalities=False,
        )["rain"]
    target_rain = target["rain"]
    return context, coarse, target_rain, target_rain - coarse


def _limit(loader, maximum: int | None):
    for index, batch in enumerate(loader):
        if maximum is not None and index >= maximum:
            break
        yield batch


def _distributed_mean(accelerator, value_sum: torch.Tensor, count: torch.Tensor) -> float:
    value_sum = accelerator.reduce(value_sum, reduction="sum")
    count = accelerator.reduce(count, reduction="sum")
    return float((value_sum / count.clamp_min(1)).item())


def _distributed_csi(accelerator, hits: torch.Tensor, misses: torch.Tensor, false_alarms: torch.Tensor) -> float:
    hits = accelerator.reduce(hits, reduction="sum")
    misses = accelerator.reduce(misses, reduction="sum")
    false_alarms = accelerator.reduce(false_alarms, reduction="sum")
    denominator = hits + misses + false_alarms
    return float((hits / denominator.clamp_min(1)).item())


def _save_checkpoint(
    *,
    trainer: RainTSNextFrameTrainer,
    model: nn.Module,
    output_path: Path,
    stage: str,
    epoch: int,
    val_loss: float,
    stage_cfg: DictConfig,
    coarse_checkpoint: Path,
    ae_checkpoint: Path | None = None,
    initial_diffusion_checkpoint: Path | None = None,
    latent_stats: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> None:
    trainer.accelerator.wait_for_everyone()
    if not trainer.accelerator.is_main_process:
        return
    payload = {
        "stage": stage,
        "epoch": epoch,
        "val_loss": val_loss,
        "model": trainer.accelerator.get_state_dict(model),
        "stage_config": OmegaConf.to_container(stage_cfg, resolve=True),
        "coarse_checkpoint": str(coarse_checkpoint),
        "ae_checkpoint": None if ae_checkpoint is None else str(ae_checkpoint),
        "initial_diffusion_checkpoint": (
            None if initial_diffusion_checkpoint is None else str(initial_diffusion_checkpoint)
        ),
    }
    if latent_stats is not None:
        payload["latent_mean"], payload["latent_std"] = latent_stats
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def _train_ae_epoch(
    trainer: RainTSNextFrameTrainer,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_cfg: dict,
    maximum: int | None,
    gradient_clip_norm: float,
) -> float:
    accelerator = trainer.accelerator
    model.train()
    loss_sum = torch.zeros((), device=trainer.device)
    count = torch.zeros((), device=trainer.device)
    optimizer.zero_grad(set_to_none=True)
    for batch in _limit(trainer.train_dataloadaer, maximum):
        _, _, _, residual = _make_residual_batch(trainer, batch)
        with accelerator.accumulate(model):
            with accelerator.autocast():
                reconstruction, _ = model(residual)
                loss, _ = residual_autoencoder_loss(reconstruction, residual, loss_cfg)
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        loss_sum += loss.detach()
        count += 1
    return _distributed_mean(accelerator, loss_sum, count)


@torch.no_grad()
def _validate_ae(
    trainer: RainTSNextFrameTrainer,
    model: nn.Module,
    loss_cfg: dict,
    maximum: int | None,
) -> float:
    accelerator = trainer.accelerator
    model.eval()
    loss_sum = torch.zeros((), device=trainer.device)
    count = torch.zeros((), device=trainer.device)
    for batch in _limit(trainer.val_dataloader, maximum):
        _, _, _, residual = _make_residual_batch(trainer, batch)
        with accelerator.autocast():
            reconstruction, _ = model(residual)
            loss, _ = residual_autoencoder_loss(reconstruction, residual, loss_cfg)
        loss_sum += loss
        count += 1
    return _distributed_mean(accelerator, loss_sum, count)


@torch.no_grad()
def _calibrate_latent(
    trainer: RainTSNextFrameTrainer,
    model: nn.Module,
    channels: int,
    maximum: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    moments = ChannelMoments(channels)
    for batch in _limit(trainer.train_dataloadaer, maximum):
        _, _, _, residual = _make_residual_batch(trainer, batch)
        with trainer.accelerator.autocast():
            latent = model.encode(residual)
        moments.update(latent)
    mean, std = moments.compute()
    return mean, std.clamp_min(1.0e-4)


def _load_ae_checkpoint(path: Path, latent_channels: int, device: torch.device):
    payload = torch.load(path.expanduser().resolve(), map_location="cpu", weights_only=False)
    ae = ResidualAutoencoder(latent_channels)
    ae.load_state_dict(payload["model"])
    ae.to(device).eval().requires_grad_(False)
    if "latent_mean" not in payload or "latent_std" not in payload:
        raise ValueError(f"AE checkpoint {path} has no calibrated latent statistics; use the AE final.pt")
    normalizer = LatentNormalizer(latent_channels).to(device)
    normalizer.set_stats(payload["latent_mean"], payload["latent_std"])
    return ae, normalizer


def _train_diffusion_epoch(
    trainer: RainTSNextFrameTrainer,
    model: nn.Module,
    ae: ResidualAutoencoder,
    normalizer: LatentNormalizer,
    optimizer: torch.optim.Optimizer,
    loss_cfg: dict,
    maximum: int | None,
    gradient_clip_norm: float,
) -> dict[str, float]:
    accelerator = trainer.accelerator
    model.train()
    metric_sums: dict[str, torch.Tensor] = {}
    count = torch.zeros((), device=trainer.device)
    optimizer.zero_grad(set_to_none=True)
    for batch in _limit(trainer.train_dataloadaer, maximum):
        history, coarse, _, residual = _make_residual_batch(trainer, batch)
        with torch.no_grad(), accelerator.autocast():
            normalized_latent = normalizer.normalize(ae.encode(residual))
        with accelerator.accumulate(model):
            with accelerator.autocast():
                outputs = model(normalized_latent, coarse, history, return_outputs=True)
                predicted_x0 = outputs["predicted_x0"].clamp(
                    -float(loss_cfg.get("x0_clip", 8.0)),
                    float(loss_cfg.get("x0_clip", 8.0)),
                )
                predicted_residual = ae.decode(normalizer.denormalize(predicted_x0))
                loss, terms = diffusion_residual_loss(
                    outputs["noise_loss"],
                    predicted_residual,
                    residual,
                    outputs["alpha_bar"],
                    loss_cfg,
                )
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        detached_terms = {"total": loss.detach(), **{name: value.detach() for name, value in terms.items()}}
        for name, value in detached_terms.items():
            metric_sums[name] = metric_sums.get(name, torch.zeros_like(value)) + value
        count += 1
    return {
        name: _distributed_mean(accelerator, value_sum, count)
        for name, value_sum in metric_sums.items()
    }


@torch.no_grad()
def _validate_diffusion(
    trainer: RainTSNextFrameTrainer,
    model: nn.Module,
    ae: ResidualAutoencoder,
    normalizer: LatentNormalizer,
    maximum: int | None,
    sample_batches: int,
    csi_thresholds: list[float],
) -> tuple[float, dict[str, float]]:
    accelerator = trainer.accelerator
    model.eval()
    loss_sum = torch.zeros((), device=trainer.device)
    coarse_l1_sum = torch.zeros((), device=trainer.device)
    final_l1_sum = torch.zeros((), device=trainer.device)
    sampled_count = torch.zeros((), device=trainer.device)
    count = torch.zeros((), device=trainer.device)
    event_counts = {
        source: {
            threshold: {
                name: torch.zeros((), device=trainer.device)
                for name in ("hits", "misses", "false_alarms")
            }
            for threshold in csi_thresholds
        }
        for source in ("coarse", "final")
    }
    for batch_index, batch in enumerate(_limit(trainer.val_dataloader, maximum)):
        history, coarse, target, residual = _make_residual_batch(trainer, batch)
        with accelerator.autocast():
            latent = ae.encode(residual)
            normalized_latent = normalizer.normalize(latent)
            loss = model(normalized_latent, coarse, history)
            if batch_index < sample_batches:
                diffusion = accelerator.unwrap_model(model)
                generator = torch.Generator(device=trainer.device).manual_seed(2025 + batch_index)
                sampled = diffusion.ddim_sample(
                    tuple(normalized_latent.shape),
                    coarse,
                    history,
                    generator=generator,
                )
                correction = ae.decode(normalizer.denormalize(sampled))
                final = (coarse + correction).clamp_min(0)
                coarse_metric = trainer._denormalize_rain_for_metrics(coarse.float()).clamp_min(0)
                final_metric = trainer._denormalize_rain_for_metrics(final.float()).clamp_min(0)
                target_metric = trainer._denormalize_rain_for_metrics(target.float()).clamp_min(0)
                coarse_l1_sum += (coarse_metric - target_metric).abs().mean()
                final_l1_sum += (final_metric - target_metric).abs().mean()
                for source, prediction in (("coarse", coarse_metric), ("final", final_metric)):
                    for threshold in csi_thresholds:
                        predicted_event = prediction >= threshold
                        target_event = target_metric >= threshold
                        counts = event_counts[source][threshold]
                        counts["hits"] += (predicted_event & target_event).sum()
                        counts["misses"] += ((~predicted_event) & target_event).sum()
                        counts["false_alarms"] += (predicted_event & (~target_event)).sum()
                sampled_count += 1
        loss_sum += loss
        count += 1
    val_loss = _distributed_mean(accelerator, loss_sum, count)
    metrics = {
        "coarse_l1": _distributed_mean(accelerator, coarse_l1_sum, sampled_count),
        "final_l1": _distributed_mean(accelerator, final_l1_sum, sampled_count),
    }
    for source, by_threshold in event_counts.items():
        for threshold, counts in by_threshold.items():
            metrics[f"{source}_csi@{threshold:g}"] = _distributed_csi(
                accelerator,
                counts["hits"],
                counts["misses"],
                counts["false_alarms"],
            )
    return val_loss, metrics


def _run_ae(args: argparse.Namespace, stage_cfg: DictConfig, trainer: RainTSNextFrameTrainer) -> None:
    training_cfg = stage_cfg.training
    epochs = int(args.epochs if args.epochs is not None else training_cfg.ae_epochs)
    model = ResidualAutoencoder(int(stage_cfg.model.latent_channels)).to(trainer.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.learning_rate),
        weight_decay=float(training_cfg.weight_decay),
    )
    model, optimizer = trainer.accelerator.prepare(model, optimizer)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
        eta_min=float(training_cfg.get("minimum_learning_rate", 1.0e-6)),
    )
    best = math.inf
    metrics_path = args.output_dir / "ae" / "metrics.jsonl"
    for epoch in range(1, epochs + 1):
        train_loss = _train_ae_epoch(
            trainer,
            model,
            optimizer,
            OmegaConf.to_container(stage_cfg.ae_loss, resolve=True),
            args.max_train_batches,
            float(training_cfg.gradient_clip_norm),
        )
        val_loss = _validate_ae(
            trainer,
            model,
            OmegaConf.to_container(stage_cfg.ae_loss, resolve=True),
            args.max_val_batches,
        )
        scheduler.step()
        row = {"epoch": epoch, "stage": "ae", "lr": scheduler.get_last_lr()[0], "train_loss": train_loss, "val_loss": val_loss}
        if trainer.accelerator.is_main_process:
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row))
        _save_checkpoint(
            trainer=trainer,
            model=model,
            output_path=args.output_dir / "ae" / "last.pt",
            stage="ae",
            epoch=epoch,
            val_loss=val_loss,
            stage_cfg=stage_cfg,
            coarse_checkpoint=args.coarse_checkpoint,
        )
        if val_loss < best:
            best = val_loss
            _save_checkpoint(
                trainer=trainer,
                model=model,
                output_path=args.output_dir / "ae" / "best.pt",
                stage="ae",
                epoch=epoch,
                val_loss=val_loss,
                stage_cfg=stage_cfg,
                coarse_checkpoint=args.coarse_checkpoint,
            )
    trainer.accelerator.wait_for_everyone()
    best_payload = torch.load(args.output_dir / "ae" / "best.pt", map_location="cpu", weights_only=False)
    trainer.accelerator.unwrap_model(model).load_state_dict(best_payload["model"])
    calibration_batches = int(training_cfg.latent_calibration_batches)
    if args.max_train_batches is not None:
        calibration_batches = min(calibration_batches, int(args.max_train_batches))
    stats = _calibrate_latent(
        trainer,
        trainer.accelerator.unwrap_model(model),
        int(stage_cfg.model.latent_channels),
        calibration_batches,
    )
    _save_checkpoint(
        trainer=trainer,
        model=model,
        output_path=args.output_dir / "ae" / "final.pt",
        stage="ae",
        epoch=epochs,
        val_loss=best,
        stage_cfg=stage_cfg,
        coarse_checkpoint=args.coarse_checkpoint,
        latent_stats=stats,
    )
    if trainer.accelerator.is_main_process:
        print(json.dumps({"latent_mean": stats[0].tolist(), "latent_std": stats[1].tolist()}))


def _run_diffusion(args: argparse.Namespace, stage_cfg: DictConfig, trainer: RainTSNextFrameTrainer) -> None:
    if args.ae_checkpoint is None:
        raise ValueError("--ae-checkpoint is required for diffusion training; use the AE final.pt")
    training_cfg = stage_cfg.training
    epochs = int(args.epochs if args.epochs is not None else training_cfg.diffusion_epochs)
    latent_channels = int(stage_cfg.model.latent_channels)
    ae, normalizer = _load_ae_checkpoint(args.ae_checkpoint, latent_channels, trainer.device)
    model = ConditionalDiffusion(OmegaConf.to_container(stage_cfg.model, resolve=True)).to(trainer.device)
    if args.diffusion_checkpoint is not None:
        initial_path = args.diffusion_checkpoint.expanduser().resolve()
        initial_payload = torch.load(initial_path, map_location="cpu", weights_only=False)
        model.load_state_dict(initial_payload["model"])
        if trainer.accelerator.is_main_process:
            print(json.dumps({"initialized_diffusion_from": str(initial_path)}))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg.learning_rate),
        weight_decay=float(training_cfg.weight_decay),
    )
    model, optimizer = trainer.accelerator.prepare(model, optimizer)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs, 1),
        eta_min=float(training_cfg.get("minimum_learning_rate", 1.0e-6)),
    )
    best = math.inf
    metrics_path = args.output_dir / "diffusion" / "metrics.jsonl"
    for epoch in range(1, epochs + 1):
        train_metrics = _train_diffusion_epoch(
            trainer,
            model,
            ae,
            normalizer,
            optimizer,
            OmegaConf.to_container(stage_cfg.diffusion_loss, resolve=True),
            args.max_train_batches,
            float(training_cfg.gradient_clip_norm),
        )
        val_loss, sampled_metrics = _validate_diffusion(
            trainer,
            model,
            ae,
            normalizer,
            args.max_val_batches,
            int(training_cfg.get("sample_val_batches", 2)),
            [float(value) for value in trainer.val_cfg.get("csi_thresholds", [0.1, 0.3, 0.5])],
        )
        scheduler.step()
        row = {
            "epoch": epoch,
            "stage": "diffusion",
            "lr": scheduler.get_last_lr()[0],
            "train_loss": train_metrics["total"],
            "val_loss": val_loss,
            **{f"train_{name}": value for name, value in train_metrics.items() if name != "total"},
            **sampled_metrics,
        }
        if trainer.accelerator.is_main_process:
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row))
        _save_checkpoint(
            trainer=trainer,
            model=model,
            output_path=args.output_dir / "diffusion" / "last.pt",
            stage="diffusion",
            epoch=epoch,
            val_loss=val_loss,
            stage_cfg=stage_cfg,
            coarse_checkpoint=args.coarse_checkpoint,
            ae_checkpoint=args.ae_checkpoint,
            initial_diffusion_checkpoint=args.diffusion_checkpoint,
        )
        selection_score = sampled_metrics["final_l1"]
        if selection_score < best:
            best = selection_score
            _save_checkpoint(
                trainer=trainer,
                model=model,
                output_path=args.output_dir / "diffusion" / "best.pt",
                stage="diffusion",
                epoch=epoch,
                val_loss=selection_score,
                stage_cfg=stage_cfg,
                coarse_checkpoint=args.coarse_checkpoint,
                ae_checkpoint=args.ae_checkpoint,
                initial_diffusion_checkpoint=args.diffusion_checkpoint,
            )
    _save_checkpoint(
        trainer=trainer,
        model=model,
        output_path=args.output_dir / "diffusion" / "final.pt",
        stage="diffusion",
        epoch=epochs,
        val_loss=best,
        stage_cfg=stage_cfg,
        coarse_checkpoint=args.coarse_checkpoint,
        ae_checkpoint=args.ae_checkpoint,
        initial_diffusion_checkpoint=args.diffusion_checkpoint,
    )


def main() -> None:
    args = _parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    stage_cfg = OmegaConf.load(args.stage_config.expanduser().resolve())
    training_cfg = stage_cfg.training
    batch_size = int(args.batch_size if args.batch_size is not None else training_cfg.batch_size)
    if args.max_train_batches is None:
        args.max_train_batches = training_cfg.get("max_train_batches")
    if args.max_val_batches is None:
        args.max_val_batches = training_cfg.get("max_val_batches")
    torch.manual_seed(int(training_cfg.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(training_cfg.seed))
    trainer = _build_coarse_trainer(
        args.base_config_name,
        args.coarse_checkpoint,
        args.output_dir,
        batch_size,
        int(training_cfg.gradient_accumulation_steps),
    )
    try:
        if args.stage == "ae":
            _run_ae(args, stage_cfg, trainer)
        else:
            _run_diffusion(args, stage_cfg, trainer)
    finally:
        trainer._close_tensorboard_writer()


if __name__ == "__main__":
    main()
