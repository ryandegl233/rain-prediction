"""Limit occurrence gating to weak-rain predictions without altering rollout state."""

import torch

from src.networks.time_series.causal_patch_transformer_soft_occurrence import (
    RainCausalPatchTransformerSoftOccurrence,
)


class RainCausalPatchTransformerBandOccurrence(RainCausalPatchTransformerSoftOccurrence):
    """Suppress uncertain weak rain while preserving moderate/strong predictions exactly."""

    def __init__(
        self,
        occurrence_band_lower_mm_min: float = 0.01,
        occurrence_band_upper_mm_min: float = 0.1,
        occurrence_max_suppression: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if occurrence_band_lower_mm_min < 0:
            raise ValueError("occurrence_band_lower_mm_min must be >= 0")
        if occurrence_band_upper_mm_min <= occurrence_band_lower_mm_min:
            raise ValueError("occurrence band upper threshold must exceed the lower threshold")
        if not 0.0 <= occurrence_max_suppression <= 1.0:
            raise ValueError("occurrence_max_suppression must be in [0, 1]")
        self.occurrence_band_lower_mm_min = float(occurrence_band_lower_mm_min)
        self.occurrence_band_upper_mm_min = float(occurrence_band_upper_mm_min)
        self.occurrence_max_suppression = float(occurrence_max_suppression)

    def forward_ar(self, *args, **kwargs) -> dict[str, torch.Tensor] | torch.Tensor:
        output = super().forward_ar(*args, **kwargs)
        if not isinstance(output, dict):
            return output

        rain = output["rain_ungated"]
        probability = output["occurrence_probability"]
        lower = rain.new_tensor(self.occurrence_band_lower_mm_min)
        upper = rain.new_tensor(self.occurrence_band_upper_mm_min)
        in_band = (rain >= lower) & (rain < upper)

        # Suppression fades to zero at 0.1 mm/min, avoiding a discontinuity near strong rain.
        band_weight = ((upper - rain) / (upper - lower)).clamp(0.0, 1.0)
        uncertainty = (1.0 - probability).pow(self.occurrence_gate_gamma)
        suppression = (
            self.soft_gate_strength.to(probability.dtype)
            * self.occurrence_max_suppression
            * band_weight
            * uncertainty
        )
        gated_rain = rain * (1.0 - suppression)

        output["rain"] = torch.where(in_band, gated_rain, rain)
        output["rain_rollout"] = rain
        output["occurrence_band_mask"] = in_band
        output["occurrence_attenuation"] = torch.where(in_band, 1.0 - suppression, torch.ones_like(rain))
        return output
