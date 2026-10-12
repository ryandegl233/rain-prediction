"""Preserve the learned pointwise path and add an initially zero correction."""

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from src.networks.time_series.causal_patch_transformer_ccc_init_fixed import RainCausalPatchTransformerCCCInitFixed
from src.networks.time_series.causal_patch_transformer_neighborhood import NeighborhoodCrossModalConditioner, neighborhood_indices
from src.networks.time_series.causal_patch_transformer_next_frame import _CrossModalConditioner


class HistoryCorrection(NeighborhoodCrossModalConditioner):
    def forward(self, target_tokens: torch.Tensor, context_x: torch.Tensor) -> torch.Tensor:
        if context_x.shape[2] < self.frame_patch_size:
            return torch.zeros_like(target_tokens)
        batch, frames, positions, dim = target_tokens.shape
        inputs = torch.split(context_x, [layer.in_channels for layer in self.adapters], dim=1)
        features = [layer(value) for layer, value in zip(self.adapters, inputs, strict=True)]
        height, width = features[0].shape[-2:]
        if positions != height * width:
            raise ValueError("History and query grids differ")
        memory = torch.stack([value.flatten(-2).permute(0, 3, 2, 1) for value in features], dim=3)
        memory = self.memory_norm(memory.flatten(2, 3))
        query = self.query_norm(target_tokens).permute(0, 2, 1, 3)
        weights, biases = self.attention.in_proj_weight.chunk(3), self.attention.in_proj_bias.chunk(3)
        heads, head_dim = self.attention.num_heads, dim // self.attention.num_heads
        q = F.linear(query, weights[0], biases[0]).reshape(batch, positions, frames, heads, head_dim)
        k = F.linear(memory, weights[1], biases[1]).reshape(batch, positions, -1, heads, head_dim)
        v = F.linear(memory, weights[2], biases[2]).reshape(batch, positions, -1, heads, head_dim)
        indices, valid = neighborhood_indices(height, width, self.radius, q.device)
        chunks = []
        for start in range(0, positions, self.query_chunk_size):
            stop = min(start + self.query_chunk_size, positions)
            args = (q[:, start:stop], k, v, indices[start:stop], valid[start:stop])
            attended = checkpoint(self._attend_chunk, *args, use_reentrant=False) if self.training and self.recompute and torch.is_grad_enabled() else self._attend_chunk(*args)
            chunks.append(attended)
        delta = self.attention.out_proj(torch.cat(chunks, dim=1)).permute(0, 2, 1, 3)
        return torch.tanh(self.gate).to(delta.dtype) * delta


class PreservedPointwiseConditioner(_CrossModalConditioner):
    def __init__(self, correction_radius: int = 1, correction_chunk_size: int = 256,
                 correction_scale_init: float = 0.1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.consensus_calibration is not None:
            raise ValueError("This correction experiment requires CCC disabled")
        if not math.isfinite(correction_scale_init) or not 0 < correction_scale_init < 1:
            raise ValueError("Correction scale must be finite and in (0, 1)")
        self.correction_scale_init = correction_scale_init
        self.correction_enabled = True
        self.correction = HistoryCorrection(
            radius=correction_radius, query_chunk_size=correction_chunk_size, recompute=True, **kwargs,
        )
        self.initialize_correction()

    @torch.no_grad()
    def initialize_correction(self) -> None:
        base = {key: value for key, value in self.state_dict().items() if not key.startswith("correction.")}
        self.correction.load_state_dict(base, strict=True)
        self.correction.gate.fill_(math.atanh(self.correction_scale_init))
        nn.init.zeros_(self.correction.attention.out_proj.weight)
        nn.init.zeros_(self.correction.attention.out_proj.bias)

    def forward(self, target_tokens: torch.Tensor, context_x: torch.Tensor) -> torch.Tensor:
        base = super().forward(target_tokens, context_x)
        if not self.correction_enabled:
            return base
        return base + self.correction(target_tokens, context_x)


class RainCausalPatchTransformerResidualNeighbor(RainCausalPatchTransformerCCCInitFixed):
    def __init__(self, correction_radius: int = 1, correction_chunk_size: int = 256,
                 correction_scale_init: float = 0.1, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        original = self.cross_modal_adapter
        if original is None or original.consensus_calibration is not None:
            raise ValueError("Preserved correction requires pointwise attention on, CCC off")
        adapter = PreservedPointwiseConditioner(
            dim=original.attention.embed_dim, heads=original.attention.num_heads,
            modality_channels=tuple(layer.in_channels for layer in original.adapters),
            patch_size=original.patch_size, frame_patch_size=original.frame_patch_size,
            correction_radius=correction_radius, correction_chunk_size=correction_chunk_size,
            correction_scale_init=correction_scale_init,
        )
        missing, unexpected = adapter.load_state_dict(original.state_dict(), strict=False)
        if unexpected or any(not key.startswith("correction.") for key in missing):
            raise ValueError("Unexpected pointwise state mismatch")
        adapter.initialize_correction()
        self.cross_modal_adapter = adapter


def freeze_original_model(model: nn.Module) -> dict:
    trainable_names = []
    counts = {"frozen": 0, "correction": 0}
    for name, parameter in model.named_parameters():
        trainable = name.startswith("cross_modal_adapter.correction.")
        parameter.requires_grad_(trainable)
        counts["correction" if trainable else "frozen"] += parameter.numel()
        if trainable:
            trainable_names.append(name)
    if not trainable_names:
        raise ValueError("No correction parameters found")
    return {"parameter_counts": counts, "trainable_names": trainable_names}
