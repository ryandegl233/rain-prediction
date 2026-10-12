"""Checkpoint-compatible explicit K/V encoding for causal multimodal history."""

from typing import Any

import torch
from torch import nn

from src.networks.time_series.causal_patch_transformer_band_occurrence import (
    RainCausalPatchTransformerBandOccurrence,
)
from src.networks.time_series.causal_patch_transformer_motion_dissipation import (
    CausalNeighborhoodCorrection,
)
from src.networks.time_series.causal_patch_transformer_next_frame import _CrossModalConditioner


class ExplicitKVConditioner(_CrossModalConditioner):
    """Preserve pointwise fusion and add an independently initialized K/V correction."""

    def __init__(
        self,
        correction_kv_heads: int,
        correction_radius: int,
        correction_max_relative_time: int,
        correction_query_chunk_size: int,
        correction_scale_init: float,
        correction_recompute: bool,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.correction = CausalNeighborhoodCorrection(
            adapters=self.adapters,
            dim=self.attention.embed_dim,
            query_heads=self.attention.num_heads,
            kv_heads=correction_kv_heads,
            radius=correction_radius,
            max_relative_time=correction_max_relative_time,
            query_chunk_size=correction_query_chunk_size,
            scale_init=correction_scale_init,
            recompute=correction_recompute,
        )

    @torch.no_grad()
    def initialize_correction(self) -> None:
        for source, target in zip(self.adapters, self.correction.adapters, strict=True):
            target.load_state_dict(source.state_dict(), strict=True)
        nn.init.zeros_(self.correction.output.weight)
        nn.init.zeros_(self.correction.output.bias)

    def forward(
        self,
        target_tokens: torch.Tensor,
        context_x: torch.Tensor,
        context_generated_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pointwise = super().forward(target_tokens, context_x)
        correction = self.correction(target_tokens, context_x, context_generated_mask)
        return pointwise + correction


class RainCausalPatchTransformerExplicitKV(RainCausalPatchTransformerBandOccurrence):
    """Band-occurrence model with an explicit causal neighborhood K/V residual."""

    def __init__(
        self,
        correction_kv_heads: int = 2,
        correction_radius: int = 1,
        correction_max_relative_time: int = 16,
        correction_query_chunk_size: int = 128,
        correction_scale_init: float = 0.1,
        correction_recompute: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        original = self.cross_modal_adapter
        if original is None:
            raise ValueError("Explicit K/V requires the pretrained pointwise cross-modal adapter")
        adapter = ExplicitKVConditioner(
            dim=original.attention.embed_dim,
            heads=original.attention.num_heads,
            modality_channels=tuple(layer.in_channels for layer in original.adapters),
            patch_size=original.patch_size,
            frame_patch_size=original.frame_patch_size,
            consensus_calibration_enabled=original.consensus_calibration is not None,
            correction_kv_heads=correction_kv_heads,
            correction_radius=correction_radius,
            correction_max_relative_time=correction_max_relative_time,
            correction_query_chunk_size=correction_query_chunk_size,
            correction_scale_init=correction_scale_init,
            correction_recompute=correction_recompute,
        )
        missing, unexpected = adapter.load_state_dict(original.state_dict(), strict=False)
        if unexpected or any(not key.startswith("correction.") for key in missing):
            raise ValueError("Unexpected pointwise adapter state mismatch")
        adapter.initialize_correction()
        self.cross_modal_adapter = adapter

    def _forward_explicit_kv(
        self,
        x: torch.Tensor,
        predict_frames: int,
        strict_target_isolation: bool,
        frame_times: torch.Tensor | None,
        context_generated_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"Expected [B,C,T,H,W], got {tuple(x.shape)}")
        batch, channels, frames, height, width = x.shape
        if channels != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {channels}")
        if frames > self.max_frames or not 0 < predict_frames <= frames:
            raise ValueError("Invalid frame count or predict_frames")
        if frames % self.frame_patch_size or predict_frames % self.frame_patch_size:
            raise ValueError("Frame counts must be divisible by frame_patch_size")

        tokens, hp, wp, _, _, temporal_positions = self._encode_tokens(
            x=x, frame_offset=0, frame_times=frame_times,
        )
        target_token_frames = predict_frames // self.frame_patch_size
        context_token_frames = int(tokens.shape[1]) - target_token_frames
        context_frames = frames - predict_frames
        for block in self.blocks:
            tokens = block(
                tokens,
                context_frames=context_token_frames,
                strict_target_isolation=strict_target_isolation,
                hp=hp,
                wp=wp,
                temporal_positions=temporal_positions,
            )
        tokens = self.norm(tokens)[:, -target_token_frames:]
        tokens = self.cross_modal_adapter(
            tokens,
            context_x=x[:, :, :context_frames],
            context_generated_mask=context_generated_mask,
        )
        if self.local_window_refiner is not None:
            tokens = self.local_window_refiner(tokens, height=hp, width=wp)
        output = self._decode_target_tokens(
            target_tokens=tokens,
            hp=hp,
            wp=wp,
            h=height,
            w=width,
            batch_size=batch,
            predict_frames=target_token_frames,
            context_anchor=self._last_context_anchor(x=x, context_frames=context_frames),
            film_condition=self._build_film_condition_sequence(
                x=x, context_frames=context_frames, predict_frames=predict_frames,
            ),
            return_modality_dict=True,
        )
        if not isinstance(output, dict):
            raise RuntimeError("Explicit K/V model requires modality dictionary output")
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
        context_generated_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        if target_x.ndim != 5:
            raise ValueError(f"target_x must be [B,C,T,H,W], got {tuple(target_x.shape)}")
        target_frames = target_x.shape[2] if predict_frames is None else int(predict_frames)
        frame_times = None
        if context_x is None:
            x = target_x
            frame_times = target_time
            if context_generated_mask is not None:
                raise ValueError("A generated mask requires context_x")
        else:
            if context_generated_mask is not None and context_generated_mask.shape != (
                context_x.shape[0], context_x.shape[2],
            ):
                raise ValueError("context_generated_mask must be [B, context_frames]")
            masked_context = self._apply_context_modality_mask_token(
                context_x=context_x,
                context_modality_available=context_modality_available,
            )
            x = torch.cat([masked_context, target_x], dim=2)
            if context_time is not None or target_time is not None:
                if context_time is None or target_time is None:
                    raise ValueError("context_time and target_time must be provided together")
                frame_times = torch.cat([context_time, target_time], dim=1)

        output = self._forward_explicit_kv(
            x=x,
            predict_frames=target_frames,
            strict_target_isolation=strict_target_isolation,
            frame_times=frame_times,
            context_generated_mask=context_generated_mask,
        )
        frames = int(output["rain"].shape[2])
        rain_reference = target_x[:, -self.rain_out_channels :, -frames:]
        ungated = output["rain"] + rain_reference
        probability = output["occurrence_logits"].sigmoid()
        lower = ungated.new_tensor(self.occurrence_band_lower_mm_min)
        upper = ungated.new_tensor(self.occurrence_band_upper_mm_min)
        in_band = (ungated >= lower) & (ungated < upper)
        band_weight = ((upper - ungated) / (upper - lower)).clamp(0.0, 1.0)
        suppression = (
            self.soft_gate_strength.to(probability.dtype)
            * self.occurrence_max_suppression
            * band_weight
            * (1.0 - probability).pow(self.occurrence_gate_gamma)
        )
        gated = torch.where(in_band, ungated * (1.0 - suppression), ungated)
        output.update({
            "rain_delta": output["rain"],
            "rain_ungated": ungated,
            "rain_rollout": ungated,
            "rain": gated,
            "occurrence_probability": probability,
            "occurrence_band_mask": in_band,
            "occurrence_attenuation": torch.where(in_band, 1.0 - suppression, torch.ones_like(ungated)),
        })
        return output if return_modality_dict else gated


def freeze_for_explicit_kv(model: nn.Module) -> dict[str, Any]:
    counts = {"frozen": 0, "explicit_kv": 0}
    trainable_names = []
    for name, parameter in model.named_parameters():
        trainable = name.startswith("cross_modal_adapter.correction.")
        parameter.requires_grad_(trainable)
        counts["explicit_kv" if trainable else "frozen"] += parameter.numel()
        if trainable:
            trainable_names.append(name)
    if not trainable_names:
        raise ValueError("No explicit K/V parameters found")
    return {"parameter_counts": counts, "trainable_names": trainable_names}
