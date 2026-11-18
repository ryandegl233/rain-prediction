import argparse
import json
import math
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from src.residual_diffusion_stage.metrics import DeterministicMetricAccumulator
from src.residual_diffusion_stage.robust_occurrence_corrector import (
    RobustOccurrenceCorrector,
    robust_occurrence_correction_loss,
)
from src.residual_diffusion_stage.train import (
    DEFAULT_BASE_CONFIG,
    _build_coarse_trainer,
    _distributed_mean,
    _limit,
    _make_residual_batch,
)
from src.trainer.rain_trainer_ts_next_frame import RainTSNextFrameTrainer


DEFAULT_STAGE_CONFIG = Path(__file__).with_name("config_v4_robust_occurrence.yaml")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the isolated robust occurrence corrector")
    parser.add_argument("--coarse-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-corrector-checkpoint", type=Path, required=True)
    parser.add_argument("--base-config-name", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--stage-config", type=Path, default=DEFAULT_STAGE_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args()


def _loss(
    trainer: RainTSNextFrameTrainer,
    outputs: dict[str, torch.Tensor],
    coarse: torch.Tensor,
    target: torch.Tensor,
    loss_cfg: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    center = trainer._denormalize_rain_for_metrics(outputs["center"].float())
    raw_center = trainer._denormalize_rain_for_metrics(outputs["center_raw"].float())
    coarse_metric = trainer._denormalize_rain_for_metrics(coarse.float())
    target_metric = trainer._denormalize_rain_for_metrics(target.float())
    mu = trainer._denormalize_rain_for_metrics((coarse + outputs["mu"]).float()) - coarse_metric
    return robust_occurrence_correction_loss(
        center=center,
        raw_center=raw_center,
        target=target_metric,
        coarse=coarse_metric,
        mu=mu,
        gate=outputs["gate"].float(),
        occurrence_logits=outputs["occurrence_logits"].float(),
        suppression=outputs["suppression"].float(),
        cfg=loss_cfg,
    )


def _mean_terms(
    trainer: RainTSNextFrameTrainer,
    sums: dict[str, torch.Tensor],
    count: torch.Tensor,
) -> dict[str, float]:
    return {
        name: _distributed_mean(trainer.accelerator, value_sum, count)
        for name, value_sum in sums.items()
    }


def _train_epoch(
    trainer: RainTSNextFrameTrainer,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_cfg: dict,
    maximum: int | None,
    gradient_clip_norm: float,
) -> dict[str, float]:
    accelerator = trainer.accelerator
    model.train()
    sums: dict[str, torch.Tensor] = {}
    count = torch.zeros((), device=trainer.device)
    optimizer.zero_grad(set_to_none=True)
    for batch in _limit(trainer.train_dataloadaer, maximum):
        history, coarse, target, _ = _make_residual_batch(trainer, batch)
        with accelerator.accumulate(model):
            with accelerator.autocast():
                outputs = model(coarse, history)
                loss, terms = _loss(trainer, outputs, coarse, target, loss_cfg)
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        detached = {"total": loss.detach(), **{name: value.detach() for name, value in terms.items()}}
        for name, value in detached.items():
            sums[name] = sums.get(name, torch.zeros_like(value)) + value
        count += 1
    return _mean_terms(trainer, sums, count)


def _flatten_metrics(prefix: str, result: dict) -> dict[str, float]:
    overall = result["overall"]
    flattened = {
        f"{prefix}_mae": overall["mae"],
        f"{prefix}_rmse": overall["rmse"],
        f"{prefix}_bias": overall["bias"],
        f"{prefix}_mean_csi": overall["mean_csi"],
    }
    for key, values in overall.items():
        if not key.startswith("threshold@"):
            continue
        threshold = key.removeprefix("threshold@")
        for name in ("csi", "pod", "far", "f1", "hss"):
            flattened[f"{prefix}_{name}@{threshold}"] = values[name]
    for lead in result["by_lead"]:
        lead_number = int(lead["lead_index"]) + 1
        for key, values in lead.items():
            if not key.startswith("threshold@"):
                continue
            threshold = key.removeprefix("threshold@")
            for name in ("csi", "pod", "far"):
                flattened[f"{prefix}_t{lead_number}_{name}@{threshold}"] = values[name]
    return flattened


@torch.no_grad()
def _validate(
    trainer: RainTSNextFrameTrainer,
    model: nn.Module,
    loss_cfg: dict,
    maximum: int | None,
) -> dict[str, float]:
    accelerator = trainer.accelerator
    model.eval()
    sums: dict[str, torch.Tensor] = {}
    count = torch.zeros((), device=trainer.device)
    thresholds = [float(value) for value in loss_cfg["metric_thresholds"]]
    robust_metrics = None
    raw_metrics = None
    for batch in _limit(trainer.val_dataloader, maximum):
        history, coarse, target, _ = _make_residual_batch(trainer, batch)
        with accelerator.autocast():
            outputs = model(coarse, history)
            loss, terms = _loss(trainer, outputs, coarse, target, loss_cfg)
        center = trainer._denormalize_rain_for_metrics(outputs["center"].float()).clamp_min(0.0)
        raw_center = trainer._denormalize_rain_for_metrics(outputs["center_raw"].float()).clamp_min(0.0)
        target_metric = trainer._denormalize_rain_for_metrics(target.float()).clamp_min(0.0)
        if robust_metrics is None or raw_metrics is None:
            lead_times = int(target.shape[2])
            robust_metrics = DeterministicMetricAccumulator(thresholds, lead_times, trainer.device)
            raw_metrics = DeterministicMetricAccumulator(thresholds, lead_times, trainer.device)
        robust_metrics.update(center, target_metric)
        raw_metrics.update(raw_center, target_metric)
        detached = {"total": loss.detach(), **{name: value.detach() for name, value in terms.items()}}
        for name, value in detached.items():
            sums[name] = sums.get(name, torch.zeros_like(value)) + value
        count += 1
    if robust_metrics is None or raw_metrics is None:
        raise RuntimeError("Validation loader produced no batches")
    robust_metrics.reduce(accelerator)
    raw_metrics.reduce(accelerator)
    return {
        **_mean_terms(trainer, sums, count),
        **_flatten_metrics("metric", robust_metrics.result()),
        **_flatten_metrics("raw_metric", raw_metrics.result()),
    }


def _selection_score(metrics: dict[str, float], cfg: DictConfig) -> float:
    low_threshold = float(cfg.low_threshold)
    strong_threshold = float(cfg.strong_threshold)
    return (
        float(cfg.mae_weight) * metrics["metric_mae"]
        + float(cfg.low_threshold_far_weight) * metrics[f"metric_far@{low_threshold:g}"]
        + float(cfg.mean_csi_penalty_weight) * (1.0 - metrics["metric_mean_csi"])
        + float(cfg.strong_csi_penalty_weight) * (1.0 - metrics[f"metric_csi@{strong_threshold:g}"])
    )


def _save_checkpoint(
    *,
    trainer: RainTSNextFrameTrainer,
    model: nn.Module,
    path: Path,
    epoch: int,
    selection_score: float,
    stage_cfg: DictConfig,
    coarse_checkpoint: Path,
    initial_corrector_checkpoint: Path,
) -> None:
    trainer.accelerator.wait_for_everyone()
    if not trainer.accelerator.is_main_process:
        return
    payload = {
        "stage": "corrector",
        "corrector_variant": "robust_occurrence",
        "epoch": epoch,
        "val_loss": selection_score,
        "selection_score": selection_score,
        "model": trainer.accelerator.get_state_dict(model),
        "stage_config": OmegaConf.to_container(stage_cfg, resolve=True),
        "coarse_checkpoint": str(coarse_checkpoint),
        "initial_corrector_checkpoint": str(initial_corrector_checkpoint),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def main() -> None:
    args = _parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    stage_cfg = OmegaConf.load(args.stage_config.expanduser().resolve())
    training_cfg = stage_cfg.training
    batch_size = int(args.batch_size if args.batch_size is not None else training_cfg.batch_size)
    maximum_train = args.max_train_batches
    if maximum_train is None:
        maximum_train = training_cfg.get("max_train_batches")
    maximum_val = args.max_val_batches
    if maximum_val is None:
        maximum_val = training_cfg.get("max_val_batches")
    seed = int(training_cfg.seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    trainer = _build_coarse_trainer(
        args.base_config_name,
        args.coarse_checkpoint,
        args.output_dir,
        batch_size,
        int(training_cfg.gradient_accumulation_steps),
    )
    try:
        model_cfg = OmegaConf.to_container(stage_cfg.model, resolve=True)
        model = RobustOccurrenceCorrector(model_cfg).to(trainer.device)
        initial_payload = torch.load(
            args.initial_corrector_checkpoint.expanduser().resolve(),
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(initial_payload["model"])
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(training_cfg.corrector_learning_rate),
            weight_decay=float(training_cfg.weight_decay),
        )
        model, optimizer = trainer.accelerator.prepare(model, optimizer)
        epochs = int(args.epochs if args.epochs is not None else training_cfg.corrector_epochs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(epochs, 1),
            eta_min=float(training_cfg.minimum_learning_rate),
        )
        loss_cfg = OmegaConf.to_container(stage_cfg.corrector_loss, resolve=True)
        metrics_path = args.output_dir / "corrector" / "metrics.jsonl"

        initial_metrics = _validate(trainer, model, loss_cfg, maximum_val)
        initial_score = _selection_score(initial_metrics, stage_cfg.selection)
        initial_row = {"epoch": 0, "stage": "robust_corrector", "selection_score": initial_score, **initial_metrics}
        if trainer.accelerator.is_main_process:
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with metrics_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(initial_row) + "\n")
            print(json.dumps(initial_row))
        _save_checkpoint(
            trainer=trainer,
            model=model,
            path=args.output_dir / "corrector" / "initial.pt",
            epoch=0,
            selection_score=initial_score,
            stage_cfg=stage_cfg,
            coarse_checkpoint=args.coarse_checkpoint,
            initial_corrector_checkpoint=args.initial_corrector_checkpoint,
        )
        _save_checkpoint(
            trainer=trainer,
            model=model,
            path=args.output_dir / "corrector" / "best.pt",
            epoch=0,
            selection_score=initial_score,
            stage_cfg=stage_cfg,
            coarse_checkpoint=args.coarse_checkpoint,
            initial_corrector_checkpoint=args.initial_corrector_checkpoint,
        )
        if args.eval_only:
            return

        best = {
            "selection": initial_score,
            "mae": initial_metrics["metric_mae"],
            "far": initial_metrics["metric_far@0.01"],
        }
        best_csi = initial_metrics["metric_mean_csi"]
        epochs_without_improvement = 0
        for epoch in range(1, epochs + 1):
            train_metrics = _train_epoch(
                trainer,
                model,
                optimizer,
                loss_cfg,
                maximum_train,
                float(training_cfg.gradient_clip_norm),
            )
            val_metrics = _validate(trainer, model, loss_cfg, maximum_val)
            scheduler.step()
            score = _selection_score(val_metrics, stage_cfg.selection)
            row = {
                "epoch": epoch,
                "stage": "robust_corrector",
                "lr": scheduler.get_last_lr()[0],
                "selection_score": score,
                **{f"train_{name}": value for name, value in train_metrics.items()},
                **{f"val_{name}": value for name, value in val_metrics.items()},
            }
            if trainer.accelerator.is_main_process:
                with metrics_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row) + "\n")
                print(json.dumps(row))
            _save_checkpoint(
                trainer=trainer,
                model=model,
                path=args.output_dir / "corrector" / "last.pt",
                epoch=epoch,
                selection_score=score,
                stage_cfg=stage_cfg,
                coarse_checkpoint=args.coarse_checkpoint,
                initial_corrector_checkpoint=args.initial_corrector_checkpoint,
            )

            improved = score < best["selection"]
            if improved:
                best["selection"] = score
                epochs_without_improvement = 0
                _save_checkpoint(
                    trainer=trainer,
                    model=model,
                    path=args.output_dir / "corrector" / "best.pt",
                    epoch=epoch,
                    selection_score=score,
                    stage_cfg=stage_cfg,
                    coarse_checkpoint=args.coarse_checkpoint,
                    initial_corrector_checkpoint=args.initial_corrector_checkpoint,
                )
            else:
                epochs_without_improvement += 1
            if val_metrics["metric_mae"] < best["mae"]:
                best["mae"] = val_metrics["metric_mae"]
                _save_checkpoint(
                    trainer=trainer,
                    model=model,
                    path=args.output_dir / "corrector" / "best_mae.pt",
                    epoch=epoch,
                    selection_score=score,
                    stage_cfg=stage_cfg,
                    coarse_checkpoint=args.coarse_checkpoint,
                    initial_corrector_checkpoint=args.initial_corrector_checkpoint,
                )
            if val_metrics["metric_far@0.01"] < best["far"]:
                best["far"] = val_metrics["metric_far@0.01"]
                _save_checkpoint(
                    trainer=trainer,
                    model=model,
                    path=args.output_dir / "corrector" / "best_far_0p01.pt",
                    epoch=epoch,
                    selection_score=score,
                    stage_cfg=stage_cfg,
                    coarse_checkpoint=args.coarse_checkpoint,
                    initial_corrector_checkpoint=args.initial_corrector_checkpoint,
                )
            if val_metrics["metric_mean_csi"] > best_csi:
                best_csi = val_metrics["metric_mean_csi"]
                _save_checkpoint(
                    trainer=trainer,
                    model=model,
                    path=args.output_dir / "corrector" / "best_mean_csi.pt",
                    epoch=epoch,
                    selection_score=score,
                    stage_cfg=stage_cfg,
                    coarse_checkpoint=args.coarse_checkpoint,
                    initial_corrector_checkpoint=args.initial_corrector_checkpoint,
                )
            if (
                epoch >= int(training_cfg.early_stopping_min_epochs)
                and epochs_without_improvement >= int(training_cfg.early_stopping_patience)
            ):
                break

        best_path = args.output_dir / "corrector" / "best.pt"
        if not best_path.is_file():
            best_path = args.output_dir / "corrector" / "initial.pt"
        best_payload = torch.load(best_path, map_location="cpu", weights_only=False)
        trainer.accelerator.unwrap_model(model).load_state_dict(best_payload["model"])
        _save_checkpoint(
            trainer=trainer,
            model=model,
            path=args.output_dir / "corrector" / "final.pt",
            epoch=int(best_payload["epoch"]),
            selection_score=float(best_payload["selection_score"]),
            stage_cfg=stage_cfg,
            coarse_checkpoint=args.coarse_checkpoint,
            initial_corrector_checkpoint=args.initial_corrector_checkpoint,
        )
    finally:
        trainer._close_tensorboard_writer()


if __name__ == "__main__":
    main()
