"""Causal neighborhood fusion with explicit rain transport and source/sink correction."""

import copy
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from src.networks.time_series.causal_patch_transformer_band_occurrence import (
    RainCausalPatchTransformerBandOccurrence,
)
from src.networks.time_series.causal_patch_transformer_neighborhood import neighborhood_indices


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input = nn.Linear(dim, hidden_dim * 2)
        self.output = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.input(x).chunk(2, dim=-1)
        return self.output(value * F.silu(gate))


class CausalNeighborhoodCorrection(nn.Module):
    """3x3 GQA correction with spatial/time/modality/source identity."""

    def __init__(
        self,
        adapters: nn.ModuleList,
        dim: int,
        query_heads: int,
        kv_heads: int,
        radius: int,
        max_relative_time: int,
        query_chunk_size: int,
        scale_init: float,
        recompute: bool,
    ) -> None:
        super().__init__()
        if radius != 1:
            raise ValueError("The first motion experiment requires a fixed 3x3 neighborhood")
        if query_heads % kv_heads != 0 or dim % query_heads != 0:
            raise ValueError("GQA requires dim % query_heads == 0 and query_heads % kv_heads == 0")
        if not 0 < scale_init <= 1:
            raise ValueError("scale_init must be in (0, 1]")
        self.adapters = copy.deepcopy(adapters)
        self.dim = int(dim)
        self.query_heads = int(query_heads)
        self.kv_heads = int(kv_heads)
        self.head_dim = dim // query_heads
        self.radius = int(radius)
        self.max_relative_time = int(max_relative_time)
        self.query_chunk_size = int(query_chunk_size)
        self.recompute = bool(recompute)
        self.query_norm = nn.RMSNorm(dim)
        self.memory_norm = nn.RMSNorm(dim)
        self.query = nn.Linear(dim, query_heads * self.head_dim)
        self.key = nn.Linear(dim, kv_heads * self.head_dim)
        self.value = nn.Linear(dim, kv_heads * self.head_dim)
        self.modality_embedding = nn.Parameter(torch.zeros(len(adapters), dim))
        self.source_embedding = nn.Parameter(torch.zeros(2, dim))
        self.spatial_bias = nn.Parameter(torch.zeros(query_heads, (2 * radius + 1) ** 2))
        self.time_bias = nn.Parameter(torch.zeros(query_heads, max_relative_time + 1))
        self.ffn_norm = nn.RMSNorm(dim)
        self.ffn = SwiGLU(dim, hidden_dim=dim * 2)
        self.output = nn.Linear(dim, dim)
        self.scale = nn.Parameter(torch.tensor(math.atanh(scale_init)))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        indices: torch.Tensor,
        valid: torch.Tensor,
        time_index: torch.Tensor,
    ) -> torch.Tensor:
        batch, count, target_frames, heads, head_dim = query.shape
        history_tokens = key.shape[2]
        neighborhood = int(indices.shape[1])
        key = key[:, indices].permute(0, 1, 4, 2, 3, 5)
        value = value[:, indices].permute(0, 1, 4, 2, 3, 5)
        repeat = self.query_heads // self.kv_heads
        key = key.repeat_interleave(repeat, dim=2).reshape(batch * count, heads, neighborhood * history_tokens, head_dim)
        value = value.repeat_interleave(repeat, dim=2).reshape(
            batch * count, heads, neighborhood * history_tokens, head_dim,
        )
        query = query.permute(0, 1, 3, 2, 4).reshape(batch * count, heads, target_frames, head_dim)
        scores = torch.matmul(query.float(), key.float().transpose(-2, -1)) / math.sqrt(head_dim)
        spatial = self.spatial_bias[:, :, None].expand(-1, -1, history_tokens).reshape(heads, -1)
        temporal = self.time_bias[:, time_index].reshape(heads, target_frames, history_tokens)
        temporal = temporal[:, :, None, :].expand(-1, -1, neighborhood, -1).reshape(heads, target_frames, -1)
        scores = scores + spatial[None, :, None, :] + temporal[None]
        mask = valid.repeat_interleave(history_tokens, dim=-1)
        mask = mask[None, :, None, None, :].expand(batch, count, 1, target_frames, -1).reshape(
            batch * count, 1, target_frames, -1,
        )
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attended = torch.matmul(scores.softmax(dim=-1).to(value.dtype), value)
        return attended.reshape(batch, count, heads, target_frames, head_dim).permute(0, 1, 3, 2, 4).flatten(-2)

    def forward(
        self,
        target_tokens: torch.Tensor,
        context_x: torch.Tensor,
        context_generated_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, target_frames, positions, _ = target_tokens.shape
        modality_inputs = torch.split(context_x, [adapter.in_channels for adapter in self.adapters], dim=1)
        features = [adapter(value) for adapter, value in zip(self.adapters, modality_inputs, strict=True)]
        history_frames, height, width = features[0].shape[2:]
        if height * width != positions:
            raise ValueError("History and target patch grids differ")
        if context_generated_mask is None:
            generated = torch.zeros(batch, history_frames, dtype=torch.long, device=context_x.device)
        else:
            if context_generated_mask.shape != (batch, context_x.shape[2]):
                raise ValueError("context_generated_mask must be [B, context_frames]")
            step = context_x.shape[2] // history_frames
            generated = context_generated_mask.reshape(batch, history_frames, step).any(dim=-1).long()

        memory_modalities = []
        for modality, feature in enumerate(features):
            tokens = feature.flatten(-2).permute(0, 3, 2, 1)
            tokens = tokens + self.modality_embedding[modality]
            tokens = tokens + self.source_embedding[generated][:, None]
            memory_modalities.append(tokens)
        memory = torch.stack(memory_modalities, dim=3).flatten(2, 3)
        memory = self.memory_norm(memory)
        query_tokens = self.query_norm(target_tokens).permute(0, 2, 1, 3)
        query = self.query(query_tokens).reshape(batch, positions, target_frames, self.query_heads, self.head_dim)
        key = self.key(memory).reshape(batch, positions, -1, self.kv_heads, self.head_dim)
        value = self.value(memory).reshape(batch, positions, -1, self.kv_heads, self.head_dim)
        indices, valid = neighborhood_indices(height, width, self.radius, context_x.device)
        history_index = torch.arange(history_frames, device=context_x.device).repeat_interleave(len(features))
        target_index = torch.arange(target_frames, device=context_x.device)[:, None]
        relative_time = history_frames + target_index - history_index[None]
        relative_time = relative_time.clamp(0, self.max_relative_time).long()
        chunks = []
        for start in range(0, positions, self.query_chunk_size):
            stop = min(start + self.query_chunk_size, positions)
            args = (query[:, start:stop], key, value, indices[start:stop], valid[start:stop], relative_time)
            if self.training and self.recompute and torch.is_grad_enabled():
                attended = checkpoint(self._attend, *args, use_reentrant=False)
            else:
                attended = self._attend(*args)
            chunks.append(attended)
        delta = torch.cat(chunks, dim=1).permute(0, 2, 1, 3)
        delta = delta + self.ffn(self.ffn_norm(delta))
        return torch.tanh(self.scale).to(delta.dtype) * self.output(delta)


class PreservedCausalFusion(nn.Module):
    """Keep the pretrained pointwise path and add an independent correction."""

    def __init__(self, pointwise: nn.Module, **correction_kwargs: Any) -> None:
        super().__init__()
        self.pointwise = pointwise
        self.correction = CausalNeighborhoodCorrection(adapters=pointwise.adapters, **correction_kwargs)

    def forward(
        self,
        target_tokens: torch.Tensor,
        context_x: torch.Tensor,
        context_generated_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base = self.pointwise(target_tokens, context_x)
        return base + self.correction(target_tokens, context_x, context_generated_mask)


class MotionDissipationHead(nn.Module):
    """Predict pixel displacement and positive growth/dissipation fields."""

    def __init__(self, in_channels: int, hidden_channels: int, max_flow_pixels: float, source_bias: float) -> None:
        super().__init__()
        self.max_flow_pixels = float(max_flow_pixels)
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(),
        )
        self.flow = nn.Conv2d(hidden_channels, 2, 1)
        self.growth = nn.Conv2d(hidden_channels, 1, 1)
        self.dissipation = nn.Conv2d(hidden_channels, 1, 1)
        for layer in (self.flow, self.growth, self.dissipation):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.constant_(self.growth.bias, source_bias)
        nn.init.constant_(self.dissipation.bias, source_bias)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, _, frames, _, _ = x.shape
        features = self.features(x.permute(0, 2, 1, 3, 4).flatten(0, 1))
        restore = lambda value: value.unflatten(0, (batch, frames)).permute(0, 2, 1, 3, 4)
        return {
            "flow": restore(torch.tanh(self.flow(features)) * self.max_flow_pixels),
            "growth": restore(F.softplus(self.growth(features))),
            "dissipation": restore(F.softplus(self.dissipation(features))),
        }


class RainCausalPatchTransformerMotionDissipation(RainCausalPatchTransformerBandOccurrence):
    """Band occurrence predictor with causal neighborhood and transport correction."""

    def __init__(
        self,
        correction_kv_heads: int = 2,
        correction_radius: int = 1,
        correction_max_relative_time: int = 16,
        correction_query_chunk_size: int = 128,
        correction_scale_init: float = 0.1,
        correction_recompute: bool = True,
        transport_hidden_channels: int = 32,
        transport_max_flow_pixels: float = 24.0,
        transport_source_bias: float = -6.0,
        transport_correction_strength: float = 0.05,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        pointwise = self.cross_modal_adapter
        if pointwise is None:
            raise ValueError("Motion model requires the pretrained pointwise cross-modal adapter")
        heads = pointwise.attention.num_heads
        self.cross_modal_adapter = PreservedCausalFusion(
            pointwise=pointwise,
            dim=self.dim,
            query_heads=heads,
            kv_heads=correction_kv_heads,
            radius=correction_radius,
            max_relative_time=correction_max_relative_time,
            query_chunk_size=correction_query_chunk_size,
            scale_init=correction_scale_init,
            recompute=correction_recompute,
        )
        self.transport_head = MotionDissipationHead(
            in_channels=self.radar_out_channels + self.satellite_out_channels + 2 * self.rain_out_channels,
            hidden_channels=transport_hidden_channels,
            max_flow_pixels=transport_max_flow_pixels,
            source_bias=transport_source_bias,
        )
        if not 0 < transport_correction_strength <= 1:
            raise ValueError("transport_correction_strength must be in (0, 1]")
        self.transport_correction_strength = float(transport_correction_strength)

    @staticmethod
    def _warp(rain: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        batch, channels, frames, height, width = rain.shape
        rain_bt = rain.permute(0, 2, 1, 3, 4).flatten(0, 1).float()
        flow_bt = flow.permute(0, 2, 1, 3, 4).flatten(0, 1).float()
        y = torch.linspace(-1, 1, height, device=rain.device, dtype=torch.float32)
        x = torch.linspace(-1, 1, width, device=rain.device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1)[None].expand(batch * frames, -1, -1, -1).clone()
        grid[..., 0] -= 2 * flow_bt[:, 0] / max(width - 1, 1)
        grid[..., 1] -= 2 * flow_bt[:, 1] / max(height - 1, 1)
        warped = F.grid_sample(rain_bt, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        return warped.unflatten(0, (batch, frames)).permute(0, 2, 1, 3, 4).to(rain.dtype)

    def _raw_forward(
        self,
        x: torch.Tensor,
        predict_frames: int,
        strict_target_isolation: bool,
        frame_times: torch.Tensor | None,
        context_generated_mask: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        batch, channels, frames, height, width = x.shape
        if channels != self.in_channels or frames > self.max_frames or predict_frames <= 0 or predict_frames > frames:
            raise ValueError("Invalid input shape or predict_frames")
        tokens, hp, wp, _, _, temporal_positions = self._encode_tokens(x=x, frame_offset=0, frame_times=frame_times)
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
        if self.cross_modal_adapter is not None:
            tokens = self.cross_modal_adapter(tokens, x[:, :, :context_frames], context_generated_mask)
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
            context_anchor=self._last_context_anchor(x, context_frames),
            film_condition=self._build_film_condition_sequence(
                x=x, context_frames=context_frames, predict_frames=predict_frames,
            ),
            return_modality_dict=True,
        )
        if not isinstance(output, dict):
            raise RuntimeError("Motion model requires modality dictionary output")
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
        target_frames = target_x.shape[2] if predict_frames is None else int(predict_frames)
        if context_x is None:
            x, frame_times = target_x, target_time
            context_frames = 0
        else:
            masked = self._apply_context_modality_mask_token(context_x, context_modality_available)
            x = torch.cat([masked, target_x], dim=2)
            frame_times = None
            if context_time is not None or target_time is not None:
                if context_time is None or target_time is None:
                    raise ValueError("context_time and target_time must be provided together")
                frame_times = torch.cat([context_time, target_time], dim=1)
            context_frames = context_x.shape[2]
        decoded = self._raw_forward(
            x, target_frames, strict_target_isolation, frame_times, context_generated_mask,
        )
        frames = decoded["rain"].shape[2]
        previous = target_x[:, -self.rain_out_channels :, -frames:]
        direct = decoded["rain"] + previous
        head_input = torch.cat([decoded["radar"], decoded["satellite"], direct, previous], dim=1)
        auxiliary = self.transport_head(head_input)
        advected = self._warp(previous, auxiliary["flow"])
        source = auxiliary["growth"] - auxiliary["dissipation"]
        transport = (advected + source).clamp_min(0)
        ungated = direct + self.transport_correction_strength * (transport - previous)
        probability = decoded["occurrence_logits"].sigmoid()
        lower = ungated.new_tensor(self.occurrence_band_lower_mm_min)
        upper = ungated.new_tensor(self.occurrence_band_upper_mm_min)
        in_band = (ungated >= lower) & (ungated < upper)
        band_weight = ((upper - ungated) / (upper - lower)).clamp(0, 1)
        suppression = self.soft_gate_strength.to(probability.dtype) * self.occurrence_max_suppression
        suppression = suppression * band_weight * (1 - probability).pow(self.occurrence_gate_gamma)
        rain = torch.where(in_band, ungated * (1 - suppression), ungated)
        decoded.update({
            "rain": rain,
            "rain_rollout": ungated,
            "rain_direct": direct,
            "rain_ungated": ungated,
            "rain_advected": advected,
            "rain_growth": auxiliary["growth"],
            "rain_dissipation": auxiliary["dissipation"],
            "rain_transport": transport,
            "flow": auxiliary["flow"],
            "occurrence_probability": probability,
            "occurrence_band_mask": in_band,
            "occurrence_attenuation": torch.where(in_band, 1 - suppression, torch.ones_like(rain)),
        })
        return decoded if return_modality_dict else rain


def configure_motion_finetuning(model: nn.Module, last_blocks: int) -> dict[str, Any]:
    if last_blocks < 1:
        raise ValueError("last_blocks must be >= 1")
    block_count = len(model.blocks)
    first_trainable = max(0, block_count - last_blocks)
    groups = {"frozen": 0, "backbone": 0, "new_heads": 0, "occurrence": 0}
    names: dict[str, list[str]] = {key: [] for key in groups if key != "frozen"}
    for name, parameter in model.named_parameters():
        is_backbone = name.startswith("rain_decoder.") or any(
            name.startswith(f"blocks.{index}.") for index in range(first_trainable, block_count)
        )
        is_new = name.startswith("cross_modal_adapter.correction.") or name.startswith("transport_head.")
        is_occurrence = name.startswith("occurrence_head.")
        trainable = is_backbone or is_new or is_occurrence
        parameter.requires_grad_(trainable)
        group = "new_heads" if is_new else "occurrence" if is_occurrence else "backbone" if is_backbone else "frozen"
        groups[group] += parameter.numel()
        if trainable:
            names[group].append(name)
    return {"parameter_counts": groups, "trainable_names": names, "last_blocks": last_blocks}
