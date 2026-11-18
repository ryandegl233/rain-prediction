import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm
from omegaconf import OmegaConf

from src.residual_diffusion_stage.evaluate_mu_gate_diffusion import _denormalize_members
from src.residual_diffusion_stage.metrics import (
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
    parser = argparse.ArgumentParser(description="Render final rainfall, probability, and uncertainty maps")
    parser.add_argument("--coarse-checkpoint", type=Path, required=True)
    parser.add_argument("--ae-checkpoint", type=Path, required=True)
    parser.add_argument("--corrector-checkpoint", type=Path, required=True)
    parser.add_argument("--diffusion-checkpoint", type=Path, required=True)
    parser.add_argument("--stage-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-config-name", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--ensemble-size", type=int, default=16)
    parser.add_argument("--ddim-steps", type=int, default=30)
    parser.add_argument("--candidate-batches", type=int, default=10)
    parser.add_argument("--deterministic-scale", type=float, default=0.25)
    parser.add_argument("--probability-scale", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


def _select_event_case(trainer, candidate_batches: int) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = None
    selected_score = -1
    for batch_index, batch in enumerate(_limit(trainer.val_dataloader, candidate_batches)):
        history, coarse, target, _ = _make_residual_batch(trainer, batch)
        target_metric = trainer._denormalize_rain_for_metrics(target.float()).clamp_min(0)
        score = int((target_metric >= 0.1).sum().item())
        if score > selected_score:
            selected_score = score
            selected = (batch_index, history.detach(), coarse.detach(), target.detach())
    if selected is None:
        raise RuntimeError("Validation loader produced no candidate cases")
    return selected


def _rain_grid(
    path: Path,
    target: np.ndarray,
    coarse: np.ndarray,
    deterministic: np.ndarray,
    final_prediction: np.ndarray,
) -> None:
    arrays = [target, coarse, deterministic, final_prediction]
    labels = ["Ground truth", "Coarse", "Coarse + gate x mu", "Final ensemble mean"]
    lead_times = target.shape[0]
    positive = np.concatenate([array[array >= 0.01] for array in arrays])
    vmax = max(float(np.quantile(positive, 0.995)) if positive.size else 0.5, 0.5)
    figure, axes = plt.subplots(len(arrays), lead_times, figsize=(3.2 * lead_times, 11), squeeze=False)
    image = None
    for row, (array, label) in enumerate(zip(arrays, labels, strict=True)):
        for lead in range(lead_times):
            masked = np.ma.masked_less(array[lead], 0.01)
            image = axes[row, lead].imshow(masked, cmap="turbo", norm=LogNorm(vmin=0.01, vmax=vmax))
            axes[row, lead].set_facecolor("white")
            axes[row, lead].set_xticks([])
            axes[row, lead].set_yticks([])
            axes[row, lead].set_title(f"t+{lead + 1}" if row == 0 else "")
            if lead == 0:
                axes[row, lead].set_ylabel(label)
    figure.colorbar(image, ax=axes, fraction=0.018, pad=0.015, label="Rainfall (mm/min)")
    figure.suptitle("Deterministic rainfall forecast", fontsize=15)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _probability_grid(path: Path, probabilities: np.ndarray, thresholds: list[float]) -> None:
    lead_times = probabilities.shape[1]
    figure, axes = plt.subplots(
        len(thresholds),
        lead_times,
        figsize=(3.2 * lead_times, 2.5 * len(thresholds) + 0.8),
        squeeze=False,
    )
    image = None
    for threshold_index, threshold in enumerate(thresholds):
        for lead in range(lead_times):
            image = axes[threshold_index, lead].imshow(
                probabilities[threshold_index, lead], cmap="magma", vmin=0, vmax=1
            )
            axes[threshold_index, lead].set_xticks([])
            axes[threshold_index, lead].set_yticks([])
            if threshold_index == 0:
                axes[threshold_index, lead].set_title(f"t+{lead + 1}")
            if lead == 0:
                axes[threshold_index, lead].set_ylabel(f"P(rain >= {threshold:g})")
    figure.colorbar(image, ax=axes, fraction=0.018, pad=0.015, label="Probability")
    figure.suptitle("Diffusion exceedance probability", fontsize=15)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _uncertainty_grid(path: Path, standard_deviation: np.ndarray, interval_width: np.ndarray) -> None:
    arrays = [standard_deviation, interval_width]
    labels = ["Ensemble std", "P90 - P10"]
    lead_times = standard_deviation.shape[0]
    figure, axes = plt.subplots(2, lead_times, figsize=(3.2 * lead_times, 5.8), squeeze=False)
    for row, (array, label) in enumerate(zip(arrays, labels, strict=True)):
        vmax = max(float(np.quantile(array, 0.995)), 1.0e-4)
        image = None
        for lead in range(lead_times):
            image = axes[row, lead].imshow(array[lead], cmap="viridis", vmin=0, vmax=vmax)
            axes[row, lead].set_xticks([])
            axes[row, lead].set_yticks([])
            if row == 0:
                axes[row, lead].set_title(f"t+{lead + 1}")
            if lead == 0:
                axes[row, lead].set_ylabel(label)
        figure.colorbar(image, ax=axes[row].tolist(), fraction=0.018, pad=0.015, label="mm/min")
    figure.suptitle("Predictive uncertainty", fontsize=15)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    if args.ensemble_size < 2:
        raise ValueError("--ensemble-size must be at least 2")
    stage_cfg = OmegaConf.load(args.stage_config.expanduser().resolve())
    model_cfg = OmegaConf.to_container(stage_cfg.model, resolve=True)
    output_dir = args.output_dir.expanduser().resolve()
    trainer = _build_coarse_trainer(
        args.base_config_name,
        args.coarse_checkpoint,
        output_dir / "visualization_runtime",
        batch_size=1,
        accumulation_steps=1,
    )
    latent_channels = int(stage_cfg.model.latent_channels)
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
    normalizer.set_stats(diffusion_payload["latent_mean"], diffusion_payload["latent_std"])
    diffusion_cfg = diffusion_payload.get("stage_config", {}).get("model", model_cfg)
    diffusion = ConditionalDiffusion(diffusion_cfg).to(trainer.device)
    diffusion.load_state_dict(diffusion_payload["model"])
    diffusion.eval().requires_grad_(False)
    try:
        sample_index, history, coarse, target = _select_event_case(trainer, args.candidate_batches)
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
            deltas = []
            for member_index in range(args.ensemble_size):
                generator = torch.Generator(device=trainer.device).manual_seed(args.seed + member_index)
                sampled = diffusion.ddim_sample(
                    latent_shape,
                    center,
                    history,
                    steps=args.ddim_steps,
                    generator=generator,
                    deterministic_residual=correction["mu"],
                    residual_gate=correction["gate"],
                )
                deltas.append(ae.decode(normalizer.denormalize(sampled)).float())
        # Keep the deterministic center as the exact ensemble mean.  The
        # uncentered samples are still available through their spread, but
        # cannot shift the reported rainfall mean due to finite sampling.
        delta_ensemble = center_ensemble_residuals(torch.stack(deltas, dim=1), variance_correction=True)
        deterministic_members = _denormalize_members(
            trainer, center[:, None] + args.deterministic_scale * delta_ensemble
        )
        probability_members = _denormalize_members(
            trainer, center[:, None] + args.probability_scale * delta_ensemble
        )
        target_metric = trainer._denormalize_rain_for_metrics(target.float()).clamp_min(0)
        coarse_metric = trainer._denormalize_rain_for_metrics(coarse.float()).clamp_min(0)
        center_metric = trainer._denormalize_rain_for_metrics(center.float()).clamp_min(0)
        final_prediction = deterministic_members.mean(dim=1)
        # The first lead has a known low-threshold false-alarm problem. Keep its
        # operational rainfall field deterministic while retaining diffusion
        # variability for later leads.
        final_prediction[:, :, 0] = center_metric[:, :, 0]
        thresholds = [
            float(value)
            for value in stage_cfg.get("evaluation", {}).get(
                "primary_thresholds", PRIMARY_THRESHOLDS_MM_PER_MIN
            )
        ]
        probabilities = torch.stack(
            [(probability_members >= threshold).float().mean(dim=1) for threshold in thresholds], dim=1
        )
        raw_probabilities = probabilities.clone()
        occurrence_prior = torch.sigmoid((center_metric - thresholds[0]) / 0.01)
        probabilities[:, 0, :, 0] = probabilities[:, 0, :, 0] * occurrence_prior[:, :, 0]
        standard_deviation = probability_members.std(dim=1, unbiased=True)
        interval_width = torch.quantile(probability_members, 0.9, dim=1) - torch.quantile(
            probability_members, 0.1, dim=1
        )
    finally:
        trainer._close_tensorboard_writer()

    fields_dir = output_dir / "final_visualization"
    fields_dir.mkdir(parents=True, exist_ok=True)
    to_numpy = lambda value: value[0, 0].detach().float().cpu().numpy()
    target_np = to_numpy(target_metric)
    coarse_np = to_numpy(coarse_metric)
    center_np = to_numpy(center_metric)
    final_np = to_numpy(final_prediction)
    probability_np = probabilities[0, :, 0].detach().float().cpu().numpy()
    plot_thresholds = [
        float(value)
        for value in stage_cfg.get("evaluation", {}).get(
            "probability_plot_thresholds", [0.1, 0.2, 0.5]
        )
    ]
    plot_indices = []
    for plot_threshold in plot_thresholds:
        matches = [index for index, threshold in enumerate(thresholds) if math.isclose(threshold, plot_threshold)]
        if not matches:
            raise ValueError(
                f"Probability plot threshold {plot_threshold:g} is not present in primary thresholds {thresholds}"
            )
        plot_indices.append(matches[0])
    standard_deviation_np = to_numpy(standard_deviation)
    interval_width_np = to_numpy(interval_width)
    _rain_grid(fields_dir / "final_prediction.png", target_np, coarse_np, center_np, final_np)
    _probability_grid(
        fields_dir / "exceedance_probability.png",
        probability_np[plot_indices],
        plot_thresholds,
    )
    _uncertainty_grid(fields_dir / "uncertainty.png", standard_deviation_np, interval_width_np)
    np.savez_compressed(
        fields_dir / "forecast_fields.npz",
        target=target_np,
        coarse=coarse_np,
        deterministic_mu_gate=center_np,
        final_prediction=final_np,
        exceedance_probability=probability_np,
        raw_exceedance_probability=raw_probabilities[0, :, 0].detach().float().cpu().numpy(),
        thresholds=np.asarray(thresholds),
        ensemble_std=standard_deviation_np,
        p90_minus_p10=interval_width_np,
    )
    summary = {
        "selected_validation_batch": sample_index,
        "selection_rule": f"most >=0.1 mm/min target pixels among first {args.candidate_batches} batches",
        "ensemble_size": args.ensemble_size,
        "ddim_steps": args.ddim_steps,
        "deterministic_scale": args.deterministic_scale,
        "probability_scale": args.probability_scale,
        "thresholds_mm_per_min": thresholds,
        "probability_plot_thresholds_mm_per_min": plot_thresholds,
        "coarse_mae": float((coarse_metric - target_metric).abs().mean().item()),
        "deterministic_mu_gate_mae": float((center_metric - target_metric).abs().mean().item()),
        "final_prediction_mae": float((final_prediction - target_metric).abs().mean().item()),
        "t1_low_threshold_calibration": "raw_probability * sigmoid((deterministic_center - 0.01) / 0.01)",
        "t1_final_rainfall_source": "deterministic_mu_gate",
    }
    (fields_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Figures: {fields_dir}")


if __name__ == "__main__":
    main()
