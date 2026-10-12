"""Checkpoint-compatible local history cross attention with bounded activation storage."""

from typing import Any

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from src.networks.time_series.causal_patch_transformer_ccc_init_fixed import RainCausalPatchTransformerCCCInitFixed
from src.networks.time_series.causal_patch_transformer_next_frame import _CrossModalConditioner


def neighborhood_indices(height: int, width: int, radius: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(height * width, device=device)
    offsets = torch.arange(-radius, radius + 1, device=device)
    dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
    y = positions[:, None] // width + dy.flatten()[None]
    x = positions[:, None] % width + dx.flatten()[None]
    valid = (y >= 0) & (y < height) & (x >= 0) & (x < width)
    indices = y.clamp(0, height - 1) * width + x.clamp(0, width - 1)
    return indices, valid


class NeighborhoodCrossModalConditioner(_CrossModalConditioner):
    def __init__(self, radius: int = 1, query_chunk_size: int = 256, recompute: bool = True, **kwargs: Any) -> None:
        if radius not in (0, 1) or query_chunk_size < 1:
            raise ValueError("Use radius 0/1 and a positive query chunk size")
        super().__init__(**kwargs)
        if self.consensus_calibration is not None:
            raise ValueError("This isolated experiment requires CCC disabled")
        self.radius = radius
        self.query_chunk_size = query_chunk_size
        self.recompute = recompute

    def _attend_chunk(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      indices: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        batch, count, target_frames, heads, head_dim = q.shape
        history_tokens = k.shape[2]
        query = q.permute(0, 1, 3, 2, 4).reshape(batch * count, heads, target_frames, head_dim)
        key = k[:, indices].permute(0, 1, 4, 2, 3, 5).reshape(batch * count, heads, -1, head_dim)
        value = v[:, indices].permute(0, 1, 4, 2, 3, 5).reshape(batch * count, heads, -1, head_dim)
        mask = valid.repeat_interleave(history_tokens, dim=-1)
        mask = mask[None, :, None, None, :].expand(batch, -1, -1, -1, -1).reshape(batch * count, 1, 1, -1)
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=mask, dropout_p=0.0)
        return attended.reshape(batch, count, heads, target_frames, head_dim).permute(0, 1, 3, 2, 4).flatten(-2)

    def forward(self, target_tokens: torch.Tensor, context_x: torch.Tensor) -> torch.Tensor:
        if context_x.shape[2] < self.frame_patch_size:
            return target_tokens
        batch, target_frames, positions, dim = target_tokens.shape
        inputs = torch.split(context_x, [adapter.in_channels for adapter in self.adapters], dim=1)
        features = [adapter(value) for adapter, value in zip(self.adapters, inputs, strict=True)]
        height, width = features[0].shape[-2:]
        if height * width != positions:
            raise ValueError("History and query patch grids must match")
        memory = torch.stack([value.flatten(-2).permute(0, 3, 2, 1) for value in features], dim=3)
        memory = self.memory_norm(memory.flatten(2, 3))
        query = self.query_norm(target_tokens).permute(0, 2, 1, 3)
        weights = self.attention.in_proj_weight.chunk(3)
        biases = self.attention.in_proj_bias.chunk(3)
        heads = self.attention.num_heads
        head_dim = dim // heads
        q = F.linear(query, weights[0], biases[0]).reshape(batch, positions, target_frames, heads, head_dim)
        k = F.linear(memory, weights[1], biases[1]).reshape(batch, positions, -1, heads, head_dim)
        v = F.linear(memory, weights[2], biases[2]).reshape(batch, positions, -1, heads, head_dim)
        indices, valid = neighborhood_indices(height, width, self.radius, q.device)
        chunks = []
        for start in range(0, positions, self.query_chunk_size):
            stop = min(start + self.query_chunk_size, positions)
            args = (q[:, start:stop], k, v, indices[start:stop], valid[start:stop])
            if self.training and self.recompute and torch.is_grad_enabled():
                attended = checkpoint(self._attend_chunk, *args, use_reentrant=False)
            else:
                attended = self._attend_chunk(*args)
            chunks.append(attended)
        conditioned = self.attention.out_proj(torch.cat(chunks, dim=1)).permute(0, 2, 1, 3)
        return target_tokens + torch.tanh(self.gate).to(target_tokens.dtype) * conditioned


class RainCausalPatchTransformerNeighborhood(RainCausalPatchTransformerCCCInitFixed):
    def __init__(self, neighborhood_radius: int = 1, neighborhood_query_chunk_size: int = 256,
                 neighborhood_recompute: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        original = self.cross_modal_adapter
        if original is None or original.consensus_calibration is not None:
            raise ValueError("Neighborhood experiment requires cross attention on and CCC off")
        adapter = NeighborhoodCrossModalConditioner(
            dim=original.attention.embed_dim, heads=original.attention.num_heads,
            modality_channels=tuple(layer.in_channels for layer in original.adapters),
            patch_size=original.patch_size, frame_patch_size=original.frame_patch_size,
            radius=neighborhood_radius, query_chunk_size=neighborhood_query_chunk_size,
            recompute=neighborhood_recompute,
        )
        adapter.load_state_dict(original.state_dict(), strict=True)
        self.cross_modal_adapter = adapter
