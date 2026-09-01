import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.residual_diffusion_stage.models import ConditionalDiffusion
from src.residual_diffusion_stage.train import (
    DEFAULT_BASE_CONFIG,
    DEFAULT_STAGE_CONFIG,
    _build_coarse_trainer,
    _limit,
    _load_ae_checkpoint,
    _make_residual_batch,
)


DEFAULT_SCALES = [0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25]


class ScaleMetricAccumulator:
    def __init__(self, scales: list[float], thresholds: list[float], lead_times: int, device: torch.device) -> None:
        self.scales = scales
        self.thresholds = thresholds
        shape = (len(scales), lead_times)
        event_shape = (len(scales), lead_times, len(thresholds))
        self.absolute_error = torch.zeros(shape, device=device, dtype=torch.float64)
        self.squared_error = torch.zeros(shape, device=device, dtype=torch.float64)
        self.signed_error = torch.zeros(shape, device=device, dtype=torch.float64)
        self.pixel_count = torch.zeros(shape, device=device, dtype=torch.float64)
        self.hits = torch.zeros(event_shape, device=device, dtype=torch.float64)
        self.misses = torch.zeros(event_shape, device=device, dtype=torch.float64)
        self.false_alarms = torch.zeros(event_shape, device=device, dtype=torch.float64)
        self.correct_negatives = torch.zeros(event_shape, device=device, dtype=torch.float64)

    def update(self, scale_index: int, prediction: torch.Tensor, target: torch.Tensor) -> None:
        if prediction.shape != target.shape:
            raise ValueError(f"prediction/target shape mismatch: {tuple(prediction.shape)} != {tuple(target.shape)}")
        if prediction.ndim != 5:
            raise ValueError(f"prediction and target must be [B,C,T,H,W], got {tuple(prediction.shape)}")
        error = (prediction - target).double()
        reduce_dims = (0, 1, 3, 4)
        self.absolute_error[scale_index] += error.abs().sum(dim=reduce_dims)
        self.squared_error[scale_index] += error.square().sum(dim=reduce_dims)
        self.signed_error[scale_index] += error.sum(dim=reduce_dims)
        pixels_per_lead = prediction.shape[0] * prediction.shape[1] * prediction.shape[3] * prediction.shape[4]
        self.pixel_count[scale_index] += pixels_per_lead

        for threshold_index, threshold in enumerate(self.thresholds):
            predicted_event = prediction >= threshold
            target_event = target >= threshold
            self.hits[scale_index, :, threshold_index] += (predicted_event & target_event).sum(dim=reduce_dims)
            self.misses[scale_index, :, threshold_index] += ((~predicted_event) & target_event).sum(dim=reduce_dims)
            self.false_alarms[scale_index, :, threshold_index] += (
                predicted_event & (~target_event)
            ).sum(dim=reduce_dims)
            self.correct_negatives[scale_index, :, threshold_index] += (
                (~predicted_event) & (~target_event)
            ).sum(dim=reduce_dims)

    def reduce(self, accelerator) -> None:
        for name in (
            "absolute_error",
            "squared_error",
            "signed_error",
            "pixel_count",
            "hits",
            "misses",
            "false_alarms",
            "correct_negatives",
        ):
            setattr(self, name, accelerator.reduce(getattr(self, name), reduction="sum"))

    @staticmethod
    def _event_metrics(hits: float, misses: float, false_alarms: float, correct_negatives: float) -> dict[str, float]:
        csi = hits / max(hits + misses + false_alarms, 1.0)
        pod = hits / max(hits + misses, 1.0)
        far = false_alarms / max(hits + false_alarms, 1.0)
        hss_numerator = 2.0 * (hits * correct_negatives - misses * false_alarms)
        hss_denominator = (hits + misses) * (misses + correct_negatives) + (
            hits + false_alarms
        ) * (false_alarms + correct_negatives)
        return {"csi": csi, "pod": pod, "far": far, "hss": hss_numerator / max(hss_denominator, 1.0)}

    def result(self) -> list[dict]:
        rows = []
        for scale_index, scale in enumerate(self.scales):
            count = float(self.pixel_count[scale_index].sum().item())
            overall = {
                "mae": float(self.absolute_error[scale_index].sum().item()) / max(count, 1.0),
                "rmse": (float(self.squared_error[scale_index].sum().item()) / max(count, 1.0)) ** 0.5,
                "bias": float(self.signed_error[scale_index].sum().item()) / max(count, 1.0),
            }
            csi_values = []
            for threshold_index, threshold in enumerate(self.thresholds):
                event = self._event_metrics(
                    float(self.hits[scale_index, :, threshold_index].sum().item()),
                    float(self.misses[scale_index, :, threshold_index].sum().item()),
                    float(self.false_alarms[scale_index, :, threshold_index].sum().item()),
                    float(self.correct_negatives[scale_index, :, threshold_index].sum().item()),
                )
                overall[f"threshold@{threshold:g}"] = event
                csi_values.append(event["csi"])
            overall["mean_csi"] = sum(csi_values) / len(csi_values)

            by_lead = []
            for lead_index in range(self.pixel_count.shape[1]):
                lead_count = float(self.pixel_count[scale_index, lead_index].item())
                lead = {
                    "lead_index": lead_index,
                    "mae": float(self.absolute_error[scale_index, lead_index].item()) / max(lead_count, 1.0),
                    "rmse": (
                        float(self.squared_error[scale_index, lead_index].item()) / max(lead_count, 1.0)
                    )
                    ** 0.5,
                    "bias": float(self.signed_error[scale_index, lead_index].item()) / max(lead_count, 1.0),
                }
                for threshold_index, threshold in enumerate(self.thresholds):
                    lead[f"threshold@{threshold:g}"] = self._event_metrics(
                        float(self.hits[scale_index, lead_index, threshold_index].item()),
                        float(self.misses[scale_index, lead_index, threshold_index].item()),
                        float(self.false_alarms[scale_index, lead_index, threshold_index].item()),
                        float(self.correct_negatives[scale_index, lead_index, threshold_index].item()),
                    )
                by_lead.append(lead)
            rows.append({"scale": scale, "overall": overall, "by_lead": by_lead})
        return rows


def summarize_ablation(rows: list[dict], thresholds: list[float]) -> dict:
    if not rows:
        raise ValueError("Cannot summarize an empty scale ablation")
    coarse = next((row for row in rows if abs(float(row["scale"])) < 1.0e-12), None)
    if coarse is None:
        raise ValueError("Scale ablation must include scale=0 as the coarse-only baseline")
    coarse_mae = float(coarse["overall"]["mae"])
    eligible = [row for row in rows if float(row["overall"]["mae"]) <= coarse_mae]
    recommended = max(eligible, key=lambda row: (float(row["overall"]["mean_csi"]), -float(row["overall"]["mae"])))

    best_by_metric = {
        "mae": min(rows, key=lambda row: float(row["overall"]["mae"]))["scale"],
        "rmse": min(rows, key=lambda row: float(row["overall"]["rmse"]))["scale"],
        "absolute_bias": min(rows, key=lambda row: abs(float(row["overall"]["bias"])))["scale"],
        "mean_csi": max(rows, key=lambda row: float(row["overall"]["mean_csi"]))["scale"],
    }
    for threshold in thresholds:
        key = f"threshold@{threshold:g}"
        best_by_metric[f"csi@{threshold:g}"] = max(rows, key=lambda row: float(row["overall"][key]["csi"]))[
            "scale"
        ]
        best_by_metric[f"pod@{threshold:g}"] = max(rows, key=lambda row: float(row["overall"][key]["pod"]))[
            "scale"
        ]
        best_by_metric[f"far@{threshold:g}"] = min(rows, key=lambda row: float(row["overall"][key]["far"]))[
            "scale"
        ]
        best_by_metric[f"hss@{threshold:g}"] = max(rows, key=lambda row: float(row["overall"][key]["hss"]))[
            "scale"
        ]
    return {
        "selection_rule": "maximize mean CSI among scales whose MAE is no worse than scale=0",
        "recommended_scale": recommended["scale"],
        "recommended_metrics": recommended["overall"],
        "coarse_metrics": coarse["overall"],
        "best_scale_by_metric": best_by_metric,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablate the residual addition scale on the validation split")
    parser.add_argument("--coarse-checkpoint", type=Path, required=True)
    parser.add_argument("--ae-checkpoint", type=Path, required=True)
    parser.add_argument("--diffusion-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-config-name", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--stage-config", type=Path, default=DEFAULT_STAGE_CONFIG)
    parser.add_argument("--scales", type=float, nargs="+", default=DEFAULT_SCALES)
    parser.add_argument("--thresholds", type=float, nargs="+", default=None)
    parser.add_argument("--ensemble-size", type=int, default=4)
    parser.add_argument("--ddim-steps", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=60)
    parser.add_argument("--seed", type=int, default=2025)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = _parse_args()
    if args.ensemble_size <= 0:
        raise ValueError(f"--ensemble-size must be positive, got {args.ensemble_size}")
    scales = sorted(set(float(scale) for scale in args.scales))
    if 0.0 not in scales:
        scales.insert(0, 0.0)
    output_dir = args.output_dir.expanduser().resolve()
    stage_cfg = OmegaConf.load(args.stage_config.expanduser().resolve())
    trainer = _build_coarse_trainer(
        args.base_config_name,
        args.coarse_checkpoint,
        output_dir / "ablation",
        batch_size=1,
        accumulation_steps=1,
    )
    thresholds = (
        [float(value) for value in trainer.val_cfg.get("csi_thresholds", [0.1, 0.3, 0.5])]
        if args.thresholds is None
        else [float(value) for value in args.thresholds]
    )
    latent_channels = int(stage_cfg.model.latent_channels)
    ae, normalizer = _load_ae_checkpoint(args.ae_checkpoint, latent_channels, trainer.device)
    diffusion_payload = torch.load(
        args.diffusion_checkpoint.expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    model_cfg = diffusion_payload.get("stage_config", {}).get("model")
    if model_cfg is None:
        model_cfg = OmegaConf.to_container(stage_cfg.model, resolve=True)
    diffusion = ConditionalDiffusion(model_cfg).to(trainer.device)
    diffusion.load_state_dict(diffusion_payload["model"])
    diffusion.eval().requires_grad_(False)

    accumulator = None
    evaluated_batches = 0
    try:
        for batch_index, batch in enumerate(_limit(trainer.val_dataloader, args.max_val_batches)):
            history, coarse, target, _ = _make_residual_batch(trainer, batch)
            if int(coarse.shape[3]) % 4 != 0 or int(coarse.shape[4]) % 4 != 0:
                raise ValueError(f"Coarse spatial shape must be divisible by four, got {tuple(coarse.shape[-2:])}")
            latent_shape = (
                int(coarse.shape[0]),
                latent_channels,
                int(coarse.shape[2]),
                int(coarse.shape[3]) // 4,
                int(coarse.shape[4]) // 4,
            )
            correction_sum = torch.zeros_like(target)
            with trainer.accelerator.autocast():
                for member_index in range(args.ensemble_size):
                    generator = torch.Generator(device=trainer.device).manual_seed(
                        args.seed + batch_index * args.ensemble_size + member_index
                    )
                    sampled = diffusion.ddim_sample(
                        latent_shape,
                        coarse,
                        history,
                        steps=args.ddim_steps,
                        generator=generator,
                    )
                    correction_sum += ae.decode(normalizer.denormalize(sampled)).float()
            correction = correction_sum / args.ensemble_size
            target_metric = trainer._denormalize_rain_for_metrics(target.float()).clamp_min(0)
            if accumulator is None:
                accumulator = ScaleMetricAccumulator(scales, thresholds, int(target.shape[2]), trainer.device)
            for scale_index, scale in enumerate(scales):
                prediction = coarse.float() if scale == 0.0 else coarse.float() + scale * correction
                prediction_metric = trainer._denormalize_rain_for_metrics(prediction).clamp_min(0)
                accumulator.update(scale_index, prediction_metric, target_metric)
            evaluated_batches += 1
            if trainer.accelerator.is_main_process and evaluated_batches % 5 == 0:
                print(f"Evaluated {evaluated_batches}/{args.max_val_batches} validation batches")
    finally:
        trainer._close_tensorboard_writer()

    if accumulator is None:
        raise RuntimeError("Validation loader produced no batches")
    accumulator.reduce(trainer.accelerator)
    rows = accumulator.result()
    summary = summarize_ablation(rows, thresholds)
    report = {
        "coarse_checkpoint": str(args.coarse_checkpoint),
        "ae_checkpoint": str(args.ae_checkpoint),
        "diffusion_checkpoint": str(args.diffusion_checkpoint),
        "ensemble_size": args.ensemble_size,
        "ddim_steps": diffusion.sample_steps if args.ddim_steps is None else args.ddim_steps,
        "evaluated_batches_per_process": evaluated_batches,
        "thresholds": thresholds,
        "scales": scales,
        "summary": summary,
        "results": rows,
    }
    if trainer.accelerator.is_main_process:
        report_dir = output_dir / "ablation"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "residual_scale_ablation.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        table_path = report_dir / "residual_scale_ablation.jsonl"
        with table_path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps({"scale": row["scale"], **row["overall"]}) + "\n")
        print(json.dumps(summary, indent=2))
        print(f"Full report: {report_path}")


if __name__ == "__main__":
    main()
