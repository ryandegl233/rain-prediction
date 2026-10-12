"""Add an independently trained soft rain-occurrence gate to the frozen predictor."""

import torch
from einops import rearrange
from torch import nn

from src.networks.time_series.causal_patch_transformer_ccc_init_fixed import (
    RainCausalPatchTransformerCCCInitFixed,
)


class PixelOccurrenceHead(nn.Module):
    def __init__(self, dim: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = int(patch_size)
        self.norm = nn.LayerNorm(dim)
        self.projection = nn.Linear(dim, self.patch_size**2)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, tokens: torch.Tensor, height: int, width: int) -> torch.Tensor:
        logits = self.projection(self.norm(tokens))
        return rearrange(
            logits,
            "b t (h w) (ph pw) -> b 1 t (h ph) (w pw)",
            h=height,
            w=width,
            ph=self.patch_size,
            pw=self.patch_size,
        )


class RainCausalPatchTransformerSoftOccurrence(RainCausalPatchTransformerCCCInitFixed):
    def __init__(self, occurrence_gate_gamma: float = 2.0, **kwargs) -> None:
        super().__init__(**kwargs)
        if self.frame_patch_size != 1:
            raise ValueError("Soft occurrence gating currently requires frame_patch_size=1")
        if occurrence_gate_gamma <= 0:
            raise ValueError("occurrence_gate_gamma must be > 0")
        self.occurrence_gate_gamma = float(occurrence_gate_gamma)
        self.occurrence_head = PixelOccurrenceHead(self.dim, self.patch_size)
        self.register_buffer("soft_gate_strength", torch.zeros(()), persistent=True)

    @torch.no_grad()
    def set_soft_gate_strength(self, strength: float) -> None:
        if not 0.0 <= strength <= 1.0:
            raise ValueError(f"soft gate strength must be in [0, 1], got {strength}")
        self.soft_gate_strength.fill_(strength)

    def _decode_target_tokens(
        self,
        target_tokens: torch.Tensor,
        hp: int,
        wp: int,
        h: int,
        w: int,
        batch_size: int,
        predict_frames: int,
        context_anchor: dict[str, torch.Tensor] | None = None,
        film_condition: dict[str, torch.Tensor] | None = None,
        return_modality_dict: bool = True,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        output = super()._decode_target_tokens(
            target_tokens=target_tokens,
            hp=hp,
            wp=wp,
            h=h,
            w=w,
            batch_size=batch_size,
            predict_frames=predict_frames,
            context_anchor=context_anchor,
            film_condition=film_condition,
            return_modality_dict=return_modality_dict,
        )
        if not isinstance(output, dict):
            return output
        output["occurrence_logits"] = self.occurrence_head(target_tokens, hp, wp)
        return output

    def forward_ar(
        self,
        context_x: torch.Tensor | None,
        target_x: torch.Tensor,
        predict_frames: int | None = None,
        strict_target_isolation: bool = False,
        return_modality_dict: bool = True,
        context_modality_available: torch.Tensor | None = None,
        context_time: torch.Tensor | None = None,
        target_time: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        output = super().forward_ar(
            context_x=context_x,
            target_x=target_x,
            predict_frames=predict_frames,
            strict_target_isolation=strict_target_isolation,
            return_modality_dict=return_modality_dict,
            context_modality_available=context_modality_available,
            context_time=context_time,
            target_time=target_time,
        )
        if not isinstance(output, dict):
            raise ValueError("Soft occurrence gating requires return_modality_dict=True")
        frames = int(output["rain"].shape[2])
        rain_reference = target_x[:, -self.rain_out_channels :, -frames:]
        ungated_rain = output["rain"] + rain_reference
        probability = output["occurrence_logits"].sigmoid()
        attenuation = 1.0 - self.soft_gate_strength.to(probability.dtype) * (
            1.0 - probability.pow(self.occurrence_gate_gamma)
        )
        output["rain_delta"] = output["rain"]
        output["rain_ungated"] = ungated_rain
        output["occurrence_probability"] = probability
        output["rain"] = ungated_rain * attenuation
        return output


def freeze_for_occurrence_training(model: nn.Module) -> dict:
    groups = {"frozen": 0, "occurrence": 0}
    trainable_names = []
    for name, parameter in model.named_parameters():
        trainable = name.startswith("occurrence_head.")
        parameter.requires_grad_(trainable)
        groups["occurrence" if trainable else "frozen"] += parameter.numel()
        if trainable:
            trainable_names.append(name)
    if not trainable_names:
        raise ValueError("No occurrence-head parameters found")
    return {"parameter_counts": groups, "trainable_names": trainable_names}
