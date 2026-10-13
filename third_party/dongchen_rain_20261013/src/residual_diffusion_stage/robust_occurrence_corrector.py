from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from src.residual_diffusion_stage.losses import deterministic_correction_loss
from src.residual_diffusion_stage.models import DeterministicResidualCorrector


class RobustOccurrenceCorrector(DeterministicResidualCorrector):
    """Suppress uncertain drizzle while preserving confident and intense rainfall."""

    def __init__(self, cfg: dict) -> None:
        super().__init__(cfg)
        robust_cfg = cfg.get("robust_occurrence", {})
        self.minimum_keep = float(robust_cfg.get("minimum_keep", 0.1))
        self.occurrence_power = float(robust_cfg.get("occurrence_power", 1.5))
        self.logit_temperature = float(robust_cfg.get("logit_temperature", 1.0))
        self.protection_threshold = float(robust_cfg.get("protection_threshold", 0.05))
        self.protection_temperature = float(robust_cfg.get("protection_temperature", 0.01))
        self.rain_norm_mean = robust_cfg.get("rain_norm_mean")
        self.rain_norm_std = robust_cfg.get("rain_norm_std")
        if not 0.0 <= self.minimum_keep <= 1.0:
            raise ValueError(f"minimum_keep must be in [0, 1], got {self.minimum_keep}")
        if self.occurrence_power <= 0.0:
            raise ValueError(f"occurrence_power must be positive, got {self.occurrence_power}")
        if self.logit_temperature <= 0.0 or self.protection_temperature <= 0.0:
            raise ValueError("Occurrence and protection temperatures must be positive")
        if (self.rain_norm_mean is None) != (self.rain_norm_std is None):
            raise ValueError("rain_norm_mean and rain_norm_std must both be set or both be null")
        if self.rain_norm_std is not None and float(self.rain_norm_std) <= 0.0:
            raise ValueError(f"rain_norm_std must be positive, got {self.rain_norm_std}")

    def forward(self, coarse_rain: torch.Tensor, history: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = super().forward(coarse_rain, history)
        occurrence_logits = outputs.get("occurrence_logits")
        if occurrence_logits is None:
            raise RuntimeError("RobustOccurrenceCorrector requires corrector_use_occurrence_head=true")

        raw_center = outputs["center"]
        if self.rain_norm_std is None:
            raw_center_physical = raw_center
        else:
            raw_center_physical = raw_center * float(self.rain_norm_std) + float(self.rain_norm_mean)
        raw_center_physical = raw_center_physical.clamp_min(0.0)

        occurrence_probability = torch.sigmoid(occurrence_logits / self.logit_temperature)
        occurrence_keep = self.minimum_keep + (1.0 - self.minimum_keep) * occurrence_probability.pow(
            self.occurrence_power
        )
        intensity_protection = torch.sigmoid(
            (raw_center_physical - self.protection_threshold) / self.protection_temperature
        )
        suppression = occurrence_keep + (1.0 - occurrence_keep) * intensity_protection
        robust_center_physical = raw_center_physical * suppression

        if self.rain_norm_std is None:
            robust_center = robust_center_physical
        else:
            robust_center = (robust_center_physical - float(self.rain_norm_mean)) / float(self.rain_norm_std)

        return {
            **outputs,
            "center_raw": raw_center,
            "center": robust_center,
            "occurrence_probability": occurrence_probability,
            "intensity_protection": intensity_protection,
            "suppression": suppression,
        }


def robust_occurrence_correction_loss(
    *,
    center: torch.Tensor,
    raw_center: torch.Tensor,
    target: torch.Tensor,
    coarse: torch.Tensor,
    mu: torch.Tensor,
    gate: torch.Tensor,
    occurrence_logits: torch.Tensor,
    suppression: torch.Tensor,
    cfg: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    base_loss, terms = deterministic_correction_loss(
        center,
        target,
        coarse,
        mu,
        gate,
        cfg,
        occurrence_logits=occurrence_logits,
    )
    active_threshold = float(cfg.get("active_threshold", 0.01))
    weak_rain_cap = float(cfg.get("weak_rain_cap", 0.05))
    light_rain_upper = float(cfg.get("light_rain_upper", 0.1))
    strong_rain_threshold = float(cfg.get("strong_rain_threshold", 0.1))
    dry_mask = target < active_threshold
    light_rain_mask = (target >= active_threshold) & (target < light_rain_upper)
    strong_rain_mask = target >= strong_rain_threshold
    zero = torch.zeros((), device=target.device, dtype=target.dtype)

    weak_dry_mass = center.clamp(min=0.0, max=weak_rain_cap)[dry_mask].mean() if dry_mask.any() else zero
    light_rain_l1 = (center - target).abs()[light_rain_mask].mean() if light_rain_mask.any() else zero
    strong_rain_retention = (
        (center - raw_center).abs()[strong_rain_mask].mean() if strong_rain_mask.any() else zero
    )
    target_occurrence = (target >= active_threshold).to(target.dtype)
    occurrence_probability = torch.sigmoid(occurrence_logits)
    occurrence_brier = F.mse_loss(occurrence_probability, target_occurrence)
    dry_suppression = (1.0 - suppression[dry_mask]).mean() if dry_mask.any() else zero

    total = (
        base_loss
        + float(cfg.get("weak_dry_mass_weight", 0.0)) * weak_dry_mass
        + float(cfg.get("light_rain_l1_weight", 0.0)) * light_rain_l1
        + float(cfg.get("strong_rain_retention_weight", 0.0)) * strong_rain_retention
        + float(cfg.get("occurrence_brier_weight", 0.0)) * occurrence_brier
    )
    return total, {
        **terms,
        "weak_dry_mass": weak_dry_mass,
        "light_rain_l1": light_rain_l1,
        "strong_rain_retention": strong_rain_retention,
        "occurrence_brier": occurrence_brier,
        "dry_suppression": dry_suppression,
        "mean_suppression": suppression.mean(),
    }


def load_robust_corrector_checkpoint(
    path: Path,
    device: torch.device,
) -> RobustOccurrenceCorrector:
    payload = torch.load(path.expanduser().resolve(), map_location="cpu", weights_only=False)
    if payload.get("corrector_variant") != "robust_occurrence":
        raise ValueError(f"Checkpoint {path} is not a robust occurrence corrector")
    model_cfg = payload["stage_config"]["model"]
    model = RobustOccurrenceCorrector(model_cfg)
    model.load_state_dict(payload["model"])
    return model.to(device).eval().requires_grad_(False)
