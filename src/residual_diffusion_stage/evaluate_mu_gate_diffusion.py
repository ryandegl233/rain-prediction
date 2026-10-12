import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.residual_diffusion_stage.metrics import (
    DeterministicMetricAccumulator,
    EnsembleMetricAccumulator,
    EXTENDED_THRESHOLDS_MM_PER_MIN,
    PRIMARY_THRESHOLDS_MM_PER_MIN,
    center_ensemble_residuals,
)
from src.residual_diffusion_stage.models import ConditionalDiffusion
from src.residual_diffusion_stage.train import (
    DEFAULT_BASE_CONFIG,
    _build_coarse_trainer,
    _limit,
    _load_ae_checkpoint,
    _load_corrector_checkpoint,
    _make_residual_batch,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate coarse + gate*mu + diffusion delta")
    parser.add_argument("--coarse-checkpoint", type=Path, required=True)
    parser.add_argument("--ae-checkpoint", type=Path, required=True)
    parser.add_argument("--corrector-checkpoint", type=Path, required=True)
    parser.add_argument("--diffusion-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-config-name", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--stage-config", type=Path, required=True)
    parser.add_argument("--ensemble-size", type=int, default=None)
    parser.add_argument("--ddim-steps", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--stochastic-scales", type=float, nargs="+", default=None)
    parser.add_argument(
        "--center-ensemble",
        dest="center_ensemble",
        action="store_true",
        default=True,
        help="Center sampled residuals so the ensemble mean remains the deterministic center (default).",
    )
    parser.add_argument(
        "--no-center-ensemble",
        dest="center_ensemble",
        action="store_false",
        help="Disable mean-preserving centering for an explicitly uncentered ablation.",
    )
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


def _denormalize_members(trainer, members: torch.Tensor) -> torch.Tensor:
    batch, ensemble_size, channels, time, height, width = members.shape
    flat = members.reshape(batch * ensemble_size, channels, time, height, width)
    flat = trainer._denormalize_rain_for_metrics(flat.float()).clamp_min(0)
    return flat.reshape(batch, ensemble_size, channels, time, height, width)


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    if args.ensemble_size is not None and args.ensemble_size < 2:
        raise ValueError("Probability evaluation requires --ensemble-size >= 2")
    stage_cfg = OmegaConf.load(args.stage_config.expanduser().resolve())
    evaluation_cfg = stage_cfg.get("evaluation", {})
    ensemble_size = int(args.ensemble_size or evaluation_cfg.get("ensemble_size", 16))
    max_val_batches = int(args.max_val_batches or evaluation_cfg.get("max_val_batches", 60))
    stochastic_scales = sorted(
        set(
            float(value)
            for value in (
                args.stochastic_scales
                if args.stochastic_scales is not None
                else evaluation_cfg.get("stochastic_scales", [0.25, 0.5, 0.75, 1.0])
            )
        )
    )
    primary_thresholds = [
        float(value) for value in evaluation_cfg.get("primary_thresholds", PRIMARY_THRESHOLDS_MM_PER_MIN)
    ]
    extended_thresholds = [
        float(value) for value in evaluation_cfg.get("extended_thresholds", EXTENDED_THRESHOLDS_MM_PER_MIN)
    ]
    output_dir = args.output_dir.expanduser().resolve()
    trainer = _build_coarse_trainer(
        args.base_config_name,
        args.coarse_checkpoint,
        output_dir / "evaluation_runtime",
        batch_size=1,
        accumulation_steps=1,
    )
    latent_channels = int(stage_cfg.model.latent_channels)
    model_cfg = OmegaConf.to_container(stage_cfg.model, resolve=True)
    ae, normalizer = _load_ae_checkpoint(args.ae_checkpoint, latent_channels, trainer.device)
    corrector = _load_corrector_checkpoint(
        args.corrector_checkpoint,
        model_cfg,
        trainer.device,
        trainer.rain_norm_mean,
        trainer.rain_norm_std,
    )
    diffusion_payload = torch.load(
        args.diffusion_checkpoint.expanduser().resolve(), map_location="cpu", weights_only=False
    )
    diffusion_cfg = diffusion_payload.get("stage_config", {}).get("model", model_cfg)
    if "latent_mean" not in diffusion_payload or "latent_std" not in diffusion_payload:
        raise ValueError("The v3 diffusion checkpoint has no delta-latent calibration statistics")
    normalizer.set_stats(diffusion_payload["latent_mean"], diffusion_payload["latent_std"])
    diffusion = ConditionalDiffusion(diffusion_cfg).to(trainer.device)
    diffusion.load_state_dict(diffusion_payload["model"])
    diffusion.eval().requires_grad_(False)

    deterministic_accumulators = None
    probability_accumulators = None
    evaluated_batches = 0
    try:
        for batch_index, batch in enumerate(_limit(trainer.val_dataloader, max_val_batches)):
            history, coarse, target, _ = _make_residual_batch(trainer, batch)
            if int(coarse.shape[-2]) % 4 != 0 or int(coarse.shape[-1]) % 4 != 0:
                raise ValueError(f"Spatial shape must be divisible by four, got {tuple(coarse.shape[-2:])}")
            with trainer.accelerator.autocast():
                correction = corrector(coarse, history)
                center = correction["center"]
                latent_shape = (
                    int(coarse.shape[0]),
                    latent_channels,
                    int(coarse.shape[2]),
                    int(coarse.shape[3]) // 4,
                    int(coarse.shape[4]) // 4,
                )
                delta_members = []
                for member_index in range(ensemble_size):
                    generator = torch.Generator(device=trainer.device).manual_seed(
                        args.seed + batch_index * ensemble_size + member_index
                    )
                    sampled = diffusion.ddim_sample(
                        latent_shape,
                        center,
                        history,
                        steps=args.ddim_steps,
                        generator=generator,
                        deterministic_residual=correction["mu"],
                        residual_gate=correction["gate"],
                    )
                    delta_members.append(ae.decode(normalizer.denormalize(sampled)).float())
            deltas = torch.stack(delta_members, dim=1)
            if args.center_ensemble:
                deltas = center_ensemble_residuals(deltas, variance_correction=True)

            target_metric = trainer._denormalize_rain_for_metrics(target.float()).clamp_min(0)
            coarse_metric = trainer._denormalize_rain_for_metrics(coarse.float()).clamp_min(0)
            center_metric = trainer._denormalize_rain_for_metrics(center.float()).clamp_min(0)
            if deterministic_accumulators is None:
                lead_times = int(target.shape[2])
                deterministic_accumulators = {
                    "coarse": DeterministicMetricAccumulator(extended_thresholds, lead_times, trainer.device),
                    "deterministic_mu_gate": DeterministicMetricAccumulator(
                        extended_thresholds, lead_times, trainer.device
                    ),
                    **{
                        f"ensemble_mean_scale_{scale:g}": DeterministicMetricAccumulator(
                            extended_thresholds, lead_times, trainer.device
                        )
                        for scale in stochastic_scales
                    },
                }
                probability_accumulators = {
                    f"scale_{scale:g}": EnsembleMetricAccumulator(
                        primary_thresholds,
                        lead_times,
                        trainer.device,
                        reliability_bins=int(evaluation_cfg.get("reliability_bins", 10)),
                    )
                    for scale in stochastic_scales
                }
            deterministic_accumulators["coarse"].update(coarse_metric, target_metric)
            deterministic_accumulators["deterministic_mu_gate"].update(center_metric, target_metric)
            for scale in stochastic_scales:
                members_metric = _denormalize_members(trainer, center[:, None] + scale * deltas)
                mean_metric = members_metric.mean(dim=1)
                deterministic_accumulators[f"ensemble_mean_scale_{scale:g}"].update(mean_metric, target_metric)
                probability_accumulators[f"scale_{scale:g}"].update(
                    members_metric, target_metric, coarse_metric
                )
            evaluated_batches += 1
            if trainer.accelerator.is_main_process and evaluated_batches % 5 == 0:
                print(f"Evaluated {evaluated_batches}/{max_val_batches} validation batches")
    finally:
        trainer._close_tensorboard_writer()

    if deterministic_accumulators is None or probability_accumulators is None:
        raise RuntimeError("Validation loader produced no batches")
    for accumulator in deterministic_accumulators.values():
        accumulator.reduce(trainer.accelerator)
    for accumulator in probability_accumulators.values():
        accumulator.reduce(trainer.accelerator)
    report = {
        "metric_protocol": {
            "unit": "mm/min",
            "event_comparison": "pixel-wise >= threshold",
            "aggregation": "counts pooled across samples and pixels, reported overall and by lead",
            "primary_thresholds": primary_thresholds,
            "extended_thresholds": extended_thresholds,
            "probability_reference": "coarse deterministic forecast",
        },
        "ensemble_size": ensemble_size,
        "ddim_steps": diffusion.sample_steps if args.ddim_steps is None else args.ddim_steps,
        "center_ensemble": bool(args.center_ensemble),
        "variance_correction_after_centering": bool(args.center_ensemble),
        "evaluated_batches_per_process": evaluated_batches,
        "checkpoints": {
            "coarse": str(args.coarse_checkpoint),
            "ae": str(args.ae_checkpoint),
            "corrector": str(args.corrector_checkpoint),
            "diffusion": str(args.diffusion_checkpoint),
        },
        "deterministic": {
            name: accumulator.result() for name, accumulator in deterministic_accumulators.items()
        },
        "probabilistic": {
            name: accumulator.result() for name, accumulator in probability_accumulators.items()
        },
    }
    if trainer.accelerator.is_main_process:
        report_dir = output_dir / "evaluation"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "mu_gate_diffusion_metrics.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Full report: {report_path}")


if __name__ == "__main__":
    main()
