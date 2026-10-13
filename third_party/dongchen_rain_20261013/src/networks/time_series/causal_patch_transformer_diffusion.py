import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers.drop import DropPath
from timm.layers.mlp import Mlp
from timm.layers.patch_embed import PatchEmbed
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

from src.networks.modules.reconstruction import ConvStem, ResNetDecoder2D


def _timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.

    Args:
        timesteps: [B, L] or [B]
        dim: embedding dim
    Returns:
        [B, L, dim]
    """
    if timesteps.ndim == 1:
        timesteps = timesteps[:, None]
    timesteps = timesteps.float()
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    )
    args = timesteps[..., None] * freqs[None, None, :]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
    return emb


class DiffusionTimestepEmbedder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.proj(_timestep_embedding(timesteps, self.proj[0].in_features))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _resolve_rotary_dim(head_dim: int, fraction: float, require_multiple_of: int) -> int:
    if fraction <= 0:
        return 0
    rotary_dim = int(head_dim * fraction)
    rotary_dim = max(require_multiple_of, rotary_dim)
    rotary_dim = min(rotary_dim, head_dim)
    rotary_dim = rotary_dim - (rotary_dim % require_multiple_of)
    if rotary_dim < require_multiple_of:
        return 0
    return rotary_dim


def _normalize_temporal_positions(positions: torch.Tensor, expected_len: int, device: torch.device) -> torch.Tensor:
    if positions.ndim == 2:
        if positions.shape[0] == 0:
            raise ValueError("Empty temporal positions.")
        positions = positions[0]
    if positions.ndim != 1:
        raise ValueError(f"temporal positions must be [T] or [B,T], got {tuple(positions.shape)}")
    if positions.shape[0] != expected_len:
        raise ValueError(f"temporal positions length mismatch: got {positions.shape[0]}, expected {expected_len}")
    return positions.to(device=device, dtype=torch.long)


class RotaryEmbedding1D(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        if dim <= 0 or dim % 2 != 0:
            raise ValueError(f"1D RoPE dim must be positive and even, got {dim}")
        self.dim = dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _build_cos_sin(self, positions: torch.Tensor, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        pos = positions.to(device=device, dtype=torch.float32)
        inv_freq = self.inv_freq.to(device=device)
        freqs = pos[..., None] * inv_freq[None, ...]
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)

    def _apply_axis(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        cos, sin = self._build_cos_sin(positions=positions, device=x.device, dtype=x.dtype)
        if cos.ndim == 1:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        if cos.ndim == 2:
            cos = cos.unsqueeze(0).unsqueeze(0)  # [1,1,L,D]
            sin = sin.unsqueeze(0).unsqueeze(0)
        elif cos.ndim == 3:
            cos = cos.unsqueeze(1)  # [B,1,L,D]
            sin = sin.unsqueeze(1)
        else:
            raise ValueError(f"Unsupported cos/sin ndim={cos.ndim}")

        x_rope, x_pass = x[..., : self.dim], x[..., self.dim :]
        x_rope = (x_rope * cos) + (_rotate_half(x_rope) * sin)
        return torch.cat([x_rope, x_pass], dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._apply_axis(q, query_positions), self._apply_axis(k, key_positions)


class RotaryEmbedding2D(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0) -> None:
        super().__init__()
        if dim <= 0 or dim % 4 != 0:
            raise ValueError(f"2D RoPE dim must be positive and divisible by 4, got {dim}")
        self.dim = dim
        self.dim_h = dim // 2
        self.dim_w = dim - self.dim_h
        inv_freq_h = 1.0 / (theta ** (torch.arange(0, self.dim_h, 2, dtype=torch.float32) / self.dim_h))
        inv_freq_w = 1.0 / (theta ** (torch.arange(0, self.dim_w, 2, dtype=torch.float32) / self.dim_w))
        self.register_buffer("inv_freq_h", inv_freq_h, persistent=False)
        self.register_buffer("inv_freq_w", inv_freq_w, persistent=False)

    def _build_cos_sin(
        self,
        h_ids: torch.Tensor,
        w_ids: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = h_ids.to(device=device, dtype=torch.float32)
        w = w_ids.to(device=device, dtype=torch.float32)
        freqs_h = h[..., None] * self.inv_freq_h.to(device=device)[None, ...]
        freqs_w = w[..., None] * self.inv_freq_w.to(device=device)[None, ...]
        emb_h = torch.cat([freqs_h, freqs_h], dim=-1)
        emb_w = torch.cat([freqs_w, freqs_w], dim=-1)
        emb = torch.cat([emb_h, emb_w], dim=-1)
        return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)

    def _apply_axis(self, x: torch.Tensor, hw_ids: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        h_ids, w_ids = hw_ids
        cos, sin = self._build_cos_sin(h_ids=h_ids, w_ids=w_ids, device=x.device, dtype=x.dtype)
        if cos.ndim == 1:
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
        if cos.ndim == 2:
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        elif cos.ndim == 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        else:
            raise ValueError(f"Unsupported cos/sin ndim={cos.ndim}")
        x_rope, x_pass = x[..., : self.dim], x[..., self.dim :]
        x_rope = (x_rope * cos) + (_rotate_half(x_rope) * sin)
        return torch.cat([x_rope, x_pass], dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        query_hw_ids: tuple[torch.Tensor, torch.Tensor],
        key_hw_ids: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._apply_axis(q, query_hw_ids), self._apply_axis(k, key_hw_ids)


class InterleavedMRoPE3D(nn.Module):
    """
    Future non-separated attention utility (Qwen3.5-style interleaved multi-axis RoPE).
    Expected position IDs:
      - [3, L] or [B, 3, L], corresponding to (t, h, w).
    """

    def __init__(
        self,
        dim: int,
        theta: float = 10000.0,
        mrope_section: Optional[tuple[int, int, int]] = None,
        interleaved: bool = True,
    ) -> None:
        super().__init__()
        if dim <= 0 or dim % 2 != 0:
            raise ValueError(f"MRoPE dim must be positive and even, got {dim}")
        self.dim = dim
        self.interleaved = interleaved

        half_dim = dim // 2
        if mrope_section is None:
            a = half_dim // 3
            b = half_dim // 3
            c = half_dim - a - b
            mrope_section = (a, b, c)
        if sum(mrope_section) != half_dim:
            raise ValueError(f"sum(mrope_section) must equal dim/2 ({half_dim}), got {mrope_section}")
        self.mrope_section = mrope_section

        def _build_inv(n: int) -> torch.Tensor:
            if n <= 0:
                return torch.empty(0, dtype=torch.float32)
            axis_dim = n * 2
            return 1.0 / (theta ** (torch.arange(0, axis_dim, 2, dtype=torch.float32) / axis_dim))

        self.register_buffer("inv_freq_t", _build_inv(mrope_section[0]), persistent=False)
        self.register_buffer("inv_freq_h", _build_inv(mrope_section[1]), persistent=False)
        self.register_buffer("inv_freq_w", _build_inv(mrope_section[2]), persistent=False)

    @staticmethod
    def build_3d_position_ids(
        frames: int,
        hp: int,
        wp: int,
        frame_offset: int = 0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        t = torch.arange(frame_offset, frame_offset + frames, device=device, dtype=torch.long)
        h = torch.arange(hp, device=device, dtype=torch.long)
        w = torch.arange(wp, device=device, dtype=torch.long)
        t_ids = t[:, None, None].expand(frames, hp, wp).reshape(-1)
        h_ids = h[None, :, None].expand(frames, hp, wp).reshape(-1)
        w_ids = w[None, None, :].expand(frames, hp, wp).reshape(-1)
        return torch.stack([t_ids, h_ids, w_ids], dim=0)  # [3, L]

    def _axis_freq(self, pos: torch.Tensor, inv_freq: torch.Tensor, device: torch.device) -> torch.Tensor:
        if inv_freq.numel() == 0:
            return torch.empty((*pos.shape, 0), device=device, dtype=torch.float32)
        return pos.to(device=device, dtype=torch.float32)[..., None] * inv_freq.to(device=device)[None, ...]

    def _interleave(self, parts: list[torch.Tensor]) -> torch.Tensor:
        max_dim = max(p.shape[-1] for p in parts)
        out = []
        for i in range(max_dim):
            for p in parts:
                if i < p.shape[-1]:
                    out.append(p[..., i : i + 1])
        return torch.cat(out, dim=-1) if out else torch.empty((*parts[0].shape[:-1], 0), device=parts[0].device)

    def _build_freqs(self, pos_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
        if pos_ids.ndim == 2:
            t_ids, h_ids, w_ids = pos_ids[0], pos_ids[1], pos_ids[2]
        elif pos_ids.ndim == 3:
            t_ids, h_ids, w_ids = pos_ids[:, 0, :], pos_ids[:, 1, :], pos_ids[:, 2, :]
        else:
            raise ValueError(f"MRoPE pos_ids must be [3,L] or [B,3,L], got {tuple(pos_ids.shape)}")

        parts = [
            self._axis_freq(t_ids, self.inv_freq_t, device=device),
            self._axis_freq(h_ids, self.inv_freq_h, device=device),
            self._axis_freq(w_ids, self.inv_freq_w, device=device),
        ]
        freqs = self._interleave(parts) if self.interleaved else torch.cat(parts, dim=-1)
        return freqs

    def _apply_axis(self, x: torch.Tensor, pos_ids: torch.Tensor) -> torch.Tensor:
        freqs = self._build_freqs(pos_ids=pos_ids, device=x.device)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(dtype=x.dtype)
        sin = emb.sin().to(dtype=x.dtype)
        if cos.ndim == 2:
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        elif cos.ndim == 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        x_rope, x_pass = x[..., : self.dim], x[..., self.dim :]
        x_rope = (x_rope * cos) + (_rotate_half(x_rope) * sin)
        return torch.cat([x_rope, x_pass], dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        query_pos_ids: torch.Tensor,
        key_pos_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._apply_axis(q, query_pos_ids), self._apply_axis(k, key_pos_ids)


class SdpaMultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        return x.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, h, l, d = x.shape
        return x.transpose(1, 2).reshape(b, l, h * d)

    def _build_attn_bias(self, attn_mask: Optional[torch.Tensor], q: torch.Tensor) -> Optional[torch.Tensor]:
        if attn_mask is None:
            return None
        if attn_mask.ndim == 2:
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1,1,Lq,Lk]
        elif attn_mask.ndim == 3:
            attn_mask = attn_mask.unsqueeze(1)  # [B,1,Lq,Lk]
        elif attn_mask.ndim != 4:
            raise ValueError(f"Unsupported attn_mask ndim={attn_mask.ndim}")

        if attn_mask.dtype == torch.bool:
            bias = torch.zeros_like(attn_mask, dtype=q.dtype, device=q.device)
            bias.masked_fill_(attn_mask, torch.finfo(q.dtype).min)
            return bias
        return attn_mask.to(device=q.device, dtype=q.dtype)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        rope_fn=None,
    ) -> torch.Tensor:
        if key is None:
            key = query
        if value is None:
            value = key

        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))
        if rope_fn is not None:
            q, k = rope_fn(q, k)

        attn_bias = self._build_attn_bias(attn_mask=attn_mask, q=q)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_bias,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        out = self._merge_heads(out)
        return self.out_proj(out)


class CausalSpatiotemporalBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        use_spatial_rope_2d: bool = False,
        use_temporal_rope_1d: bool = False,
        rope_theta_spatial: float = 10000.0,
        rope_theta_temporal: float = 10000.0,
        rope_fraction_spatial: float = 1.0,
        rope_fraction_temporal: float = 1.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_spatial_rope_2d = use_spatial_rope_2d
        self.use_temporal_rope_1d = use_temporal_rope_1d

        self.norm_spatial = nn.LayerNorm(dim)
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.spatial_attn_rope = SdpaMultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)

        self.norm_temporal = nn.LayerNorm(dim)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.temporal_attn_rope = SdpaMultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout)

        self.spatial_rotary_dim = _resolve_rotary_dim(
            head_dim=self.head_dim,
            fraction=rope_fraction_spatial,
            require_multiple_of=4,
        )
        self.temporal_rotary_dim = _resolve_rotary_dim(
            head_dim=self.head_dim,
            fraction=rope_fraction_temporal,
            require_multiple_of=2,
        )
        if self.use_spatial_rope_2d and self.spatial_rotary_dim == 0:
            raise ValueError("Spatial RoPE enabled but derived spatial_rotary_dim is 0.")
        if self.use_temporal_rope_1d and self.temporal_rotary_dim == 0:
            raise ValueError("Temporal RoPE enabled but derived temporal_rotary_dim is 0.")

        self.spatial_rope = (
            RotaryEmbedding2D(dim=self.spatial_rotary_dim, theta=rope_theta_spatial)
            if self.use_spatial_rope_2d
            else None
        )
        self.temporal_rope = (
            RotaryEmbedding1D(dim=self.temporal_rotary_dim, theta=rope_theta_temporal)
            if self.use_temporal_rope_1d
            else None
        )
        self._spatial_hw_index_cache: dict[tuple[int, int, str, int], tuple[torch.Tensor, torch.Tensor]] = {}

        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=nn.GELU,
            drop=dropout,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def _build_temporal_causal_mask(self, t: int, device: torch.device) -> torch.Tensor:
        # True means "masked out" in nn.MultiheadAttention bool mask.
        return torch.triu(torch.ones(t, t, dtype=torch.bool, device=device), diagonal=1)

    def _build_temporal_mask(
        self,
        t: int,
        device: torch.device,
        context_frames: Optional[int] = None,
        strict_target_isolation: bool = False,
    ) -> torch.Tensor:
        """
        Build temporal mask for attention over [T] frames.

        Base mask is always causal (no future attention).
        Optional strict_target_isolation blocks target-target cross-frame attention
        in the same forward call, which is useful for strict AR training:
        each target frame only sees historical context + itself.
        """
        mask = self._build_temporal_causal_mask(t=t, device=device)
        if not strict_target_isolation:
            return mask

        if context_frames is None or context_frames < 0 or context_frames >= t:
            raise ValueError(
                "strict_target_isolation=True requires context_frames in [0, T-1]. "
                f"Got context_frames={context_frames}, T={t}."
            )

        # target range: [context_frames, t)
        target_idx = torch.arange(context_frames, t, device=device)
        # Keep diagonal (self) unmasked; mask all other target-target links.
        mask[target_idx[:, None], target_idx[None, :]] = True
        mask[target_idx, target_idx] = False
        return mask

    def _build_target_query_mask(
        self,
        context_frames: int,
        target_frames: int,
        device: torch.device,
        strict_target_isolation: bool = False,
    ) -> torch.Tensor:
        """
        Build attention mask for target-only temporal attention.
        Query length = target_frames, key/value length = context_frames + target_frames.
        """
        total = context_frames + target_frames
        q_pos = torch.arange(context_frames, total, device=device)
        k_pos = torch.arange(total, device=device)
        # Causal: target query at global time q can only attend keys <= q.
        mask = k_pos[None, :] > q_pos[:, None]  # [Tt, Tc+Tt]
        if strict_target_isolation and target_frames > 0:
            # Disallow target-target interactions except self.
            target_global = torch.arange(context_frames, total, device=device)
            target_mask = torch.ones((target_frames, target_frames), dtype=torch.bool, device=device)
            target_mask.fill_(True)
            target_mask.fill_diagonal_(False)  # keep self
            mask[:, context_frames:] = target_mask
            # context part remains causal-allowed
        return mask

    def _get_spatial_hw_ids(self, hp: int, wp: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        key = (hp, wp, device.type, -1 if device.index is None else device.index)
        if key in self._spatial_hw_index_cache:
            return self._spatial_hw_index_cache[key]
        h = torch.arange(hp, device=device, dtype=torch.long)
        w = torch.arange(wp, device=device, dtype=torch.long)
        h_ids = h[:, None].expand(hp, wp).reshape(-1)
        w_ids = w[None, :].expand(hp, wp).reshape(-1)
        self._spatial_hw_index_cache[key] = (h_ids, w_ids)
        return h_ids, w_ids

    def _spatial_step(self, x: torch.Tensor, hp: Optional[int] = None, wp: Optional[int] = None) -> torch.Tensor:
        b, t, n, d = x.shape
        xs = rearrange(x, "b t n d -> (b t) n d")
        xs_norm = self.norm_spatial(xs)
        if self.use_spatial_rope_2d:
            if hp is None or wp is None:
                raise ValueError("Spatial RoPE requires hp/wp.")
            h_ids, w_ids = self._get_spatial_hw_ids(hp=hp, wp=wp, device=x.device)
            xs_attn = self.spatial_attn_rope(
                query=xs_norm,
                attn_mask=None,
                rope_fn=lambda q, k: self.spatial_rope(  # type: ignore[operator]
                    q,
                    k,
                    query_hw_ids=(h_ids, w_ids),
                    key_hw_ids=(h_ids, w_ids),
                ),
            )
        else:
            xs_attn, _ = self.spatial_attn(xs_norm, xs_norm, xs_norm, need_weights=False)
        xs = xs + self.drop_path(xs_attn)
        return rearrange(xs, "(b t) n d -> b t n d", b=b, t=t)

    def _temporal_self_step(
        self,
        x: torch.Tensor,
        context_frames: Optional[int] = None,
        strict_target_isolation: bool = False,
        temporal_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, t, n, d = x.shape
        xt = rearrange(x, "b t n d -> (b n) t d")
        xt_norm = self.norm_temporal(xt)
        causal_mask = self._build_temporal_mask(
            t=t,
            device=x.device,
            context_frames=context_frames,
            strict_target_isolation=strict_target_isolation,
        )
        if self.use_temporal_rope_1d:
            if temporal_positions is None:
                raise ValueError("Temporal RoPE requires temporal_positions.")
            pos = _normalize_temporal_positions(temporal_positions, expected_len=t, device=x.device)
            xt_attn = self.temporal_attn_rope(
                query=xt_norm,
                attn_mask=causal_mask,
                rope_fn=lambda q, k: self.temporal_rope(  # type: ignore[operator]
                    q,
                    k,
                    query_positions=pos,
                    key_positions=pos,
                ),
            )
        else:
            xt_attn, _ = self.temporal_attn(
                xt_norm,
                xt_norm,
                xt_norm,
                attn_mask=causal_mask,
                need_weights=False,
            )
        xt = xt + self.drop_path(xt_attn)
        x = rearrange(xt, "(b n) t d -> b t n d", b=b, n=n)
        xm = self.norm_mlp(x)
        x = x + self.drop_path(self.mlp(xm))
        return x

    def forward(
        self,
        x: torch.Tensor,
        context_frames: Optional[int] = None,
        strict_target_isolation: bool = False,
        hp: Optional[int] = None,
        wp: Optional[int] = None,
        temporal_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: [B, T, N, D]
        """
        x = self._spatial_step(x, hp=hp, wp=wp)
        x = self._temporal_self_step(
            x,
            context_frames=context_frames,
            strict_target_isolation=strict_target_isolation,
            temporal_positions=temporal_positions,
        )
        return x

    def encode_context(
        self,
        context: torch.Tensor,
        hp: Optional[int] = None,
        wp: Optional[int] = None,
        temporal_positions: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode context-only frames for cache construction.
        Returns:
            context_out: context after full block [B, Tc, N, D]
            context_temporal_kv: context after spatial step [B, Tc, N, D]
        """
        context_temporal_kv = self._spatial_step(context, hp=hp, wp=wp)
        context_out = self._temporal_self_step(
            context_temporal_kv,
            context_frames=None,
            strict_target_isolation=False,
            temporal_positions=temporal_positions,
        )
        return context_out, context_temporal_kv

    def forward_target_with_context(
        self,
        target: torch.Tensor,
        context_temporal_kv: torch.Tensor,
        hp: Optional[int] = None,
        wp: Optional[int] = None,
        target_temporal_positions: Optional[torch.Tensor] = None,
        context_temporal_positions: Optional[torch.Tensor] = None,
        strict_target_isolation: bool = False,
        return_spatial: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Inference path: update target tokens only with cached context tokens as KV.

        Args:
            target:  [B, Tt, N, D]
            context_temporal_kv: [B, Tc, N, D], context after spatial sublayer.
        Returns:
            updated target tokens [B, Tt, N, D]
        """
        b, tt, n, d = target.shape
        tc = context_temporal_kv.shape[1]

        # 1) Spatial attention only for target frames.
        target_spatial = self._spatial_step(target, hp=hp, wp=wp)

        # 2) Temporal attention: query=target_spatial, key/value=[context_spatial, target_spatial].
        target_t = rearrange(target_spatial, "b t n d -> (b n) t d")
        context_t = rearrange(context_temporal_kv, "b t n d -> (b n) t d")
        kv_t = torch.cat([context_t, target_t], dim=1)
        q_norm = self.norm_temporal(target_t)
        kv_norm = self.norm_temporal(kv_t)
        attn_mask = self._build_target_query_mask(
            context_frames=tc,
            target_frames=tt,
            device=target.device,
            strict_target_isolation=strict_target_isolation,
        )
        if self.use_temporal_rope_1d:
            if target_temporal_positions is None or context_temporal_positions is None:
                raise ValueError("Temporal RoPE with context cache requires target/context temporal positions.")
            q_pos = _normalize_temporal_positions(target_temporal_positions, expected_len=tt, device=target.device)
            c_pos = _normalize_temporal_positions(context_temporal_positions, expected_len=tc, device=target.device)
            k_pos = torch.cat([c_pos, q_pos], dim=0)
            tgt_attn = self.temporal_attn_rope(
                query=q_norm,
                key=kv_norm,
                value=kv_norm,
                attn_mask=attn_mask,
                rope_fn=lambda q, k: self.temporal_rope(  # type: ignore[operator]
                    q,
                    k,
                    query_positions=q_pos,
                    key_positions=k_pos,
                ),
            )
        else:
            tgt_attn, _ = self.temporal_attn(
                q_norm,
                kv_norm,
                kv_norm,
                attn_mask=attn_mask,
                need_weights=False,
            )
        target_t = target_t + self.drop_path(tgt_attn)
        target = rearrange(target_t, "(b n) t d -> b t n d", b=b, n=n)

        # 3) MLP.
        tm = self.norm_mlp(target)
        target = target + self.drop_path(self.mlp(tm))
        if return_spatial:
            return target, target_spatial
        return target


class RainCausalPatchTransformerDiffusion(nn.Module):
    """
    Diffusion backbone:
    conv stem -> timm PatchEmbed -> causal spatiotemporal transformer -> ResNet decoder.

    Inputs are expected in [B, C, T, H, W].
    Typical usage for Stage-1 TF AR diffusion:
    - Put clean history and current noisy target into the time axis.
    - Use causal temporal attention to prevent future leakage.
    - Decode only the latest `predict_frames` frames as epsilon/x0 prediction target.
    - Use 3 modality-specific decoders for radar / satellite / rain.
    """

    def __init__(
        self,
        in_channels: int = 12,
        out_channels: Optional[int] = 1,
        radar_out_channels: int = 1,
        satellite_out_channels: int = 10,
        rain_out_channels: Optional[int] = None,
        input_size: int = 256,
        patch_size: int = 4,
        stem_channels: int = 128,
        dim: int = 512,
        depth: int = 8,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.1,
        max_frames: int = 32,
        decoder_base_channels: int = 256,
        use_spatial_rope_2d: bool = False,
        use_temporal_rope_1d: bool = False,
        rope_theta_spatial: float = 10000.0,
        rope_theta_temporal: float = 10000.0,
        rope_fraction_spatial: float = 1.0,
        rope_fraction_temporal: float = 1.0,
        register_mrope_3d: bool = True,
        mrope_theta: float = 10000.0,
        mrope_interleaved: bool = True,
        stem_pad_mode: str = "zeros",
        decoder_pad_mode: str = "zeros",
        decoder_upsample_mode: str = "nearest_conv",
        activation_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        if rain_out_channels is None:
            # Backward compatibility with previous single-head interface.
            rain_out_channels = 1 if out_channels is None else out_channels
        self.radar_out_channels = radar_out_channels
        self.satellite_out_channels = satellite_out_channels
        self.rain_out_channels = rain_out_channels
        self.out_channels = rain_out_channels
        self.input_size = input_size
        self.patch_size = patch_size
        self.max_frames = max_frames
        self.dim = dim
        self.num_heads = num_heads
        self.use_spatial_rope_2d = use_spatial_rope_2d
        self.use_temporal_rope_1d = use_temporal_rope_1d

        self.stem = ConvStem(in_channels=in_channels, stem_channels=stem_channels, pad_mode=stem_pad_mode)
        self.patch_embed = PatchEmbed(
            img_size=input_size,
            patch_size=patch_size,
            in_chans=stem_channels,
            embed_dim=dim,
            strict_img_size=False,
        )

        base_grid = input_size // patch_size
        self.base_grid = base_grid
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, base_grid * base_grid, dim))
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, max_frames, dim))
        self.timestep_embedder = DiffusionTimestepEmbedder(dim=dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([])
        for i in range(depth):
            blk = CausalSpatiotemporalBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path=dpr[i],
                use_spatial_rope_2d=use_spatial_rope_2d,
                use_temporal_rope_1d=use_temporal_rope_1d,
                rope_theta_spatial=rope_theta_spatial,
                rope_theta_temporal=rope_theta_temporal,
                rope_fraction_spatial=rope_fraction_spatial,
                rope_fraction_temporal=rope_fraction_temporal,
            )
            if activation_checkpoint:
                blk = CheckpointWrapper(blk)
            self.blocks.append(blk)
        self.norm = nn.LayerNorm(dim)

        self.radar_decoder = ResNetDecoder2D(
            in_channels=dim,
            out_channels=radar_out_channels,
            patch_size=patch_size,
            base_channels=decoder_base_channels,
            dropout=dropout,
            padding_mode=decoder_pad_mode,
            upsample_mode=decoder_upsample_mode,
            activation_checkpoint=activation_checkpoint,
        )
        self.satellite_decoder = ResNetDecoder2D(
            in_channels=dim,
            out_channels=satellite_out_channels,
            patch_size=patch_size,
            base_channels=decoder_base_channels,
            dropout=dropout,
            padding_mode=decoder_pad_mode,
            upsample_mode=decoder_upsample_mode,
            activation_checkpoint=activation_checkpoint,
        )
        self.rain_decoder = ResNetDecoder2D(
            in_channels=dim,
            out_channels=rain_out_channels,
            patch_size=patch_size,
            base_channels=decoder_base_channels,
            dropout=dropout,
            padding_mode=decoder_pad_mode,
            upsample_mode=decoder_upsample_mode,
            activation_checkpoint=activation_checkpoint,
        )
        self._context_cache: Optional[dict] = None
        self.mrope_3d: Optional[InterleavedMRoPE3D] = None
        if register_mrope_3d:
            head_dim = dim // num_heads
            mrope_dim = max(2, head_dim - (head_dim % 2))
            self.mrope_3d = InterleavedMRoPE3D(
                dim=mrope_dim,
                theta=mrope_theta,
                mrope_section=None,
                interleaved=mrope_interleaved,
            )

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

    def _resize_spatial_pos_embed(self, hp: int, wp: int) -> torch.Tensor:
        if hp == self.base_grid and wp == self.base_grid:
            return self.spatial_pos_embed
        pos = self.spatial_pos_embed.reshape(1, self.base_grid, self.base_grid, self.dim).permute(0, 3, 1, 2)
        pos = F.interpolate(pos, size=(hp, wp), mode="bilinear", align_corners=False)
        pos = pos.permute(0, 2, 3, 1).reshape(1, hp * wp, self.dim)
        return pos

    def _expand_timestep_to_frames(self, timesteps: torch.Tensor, batch_size: int, frames: int) -> torch.Tensor:
        if timesteps.ndim == 1:
            timesteps = timesteps[:, None]
        if timesteps.shape[0] != batch_size:
            raise ValueError(f"timesteps batch mismatch: got {timesteps.shape[0]}, expected {batch_size}")
        if timesteps.shape[1] == 1:
            timesteps = timesteps.expand(batch_size, frames)
        elif timesteps.shape[1] != frames:
            raise ValueError(f"timesteps frame mismatch: got {timesteps.shape[1]}, expected {frames}")
        return timesteps

    def _encode_tokens(
        self,
        x: torch.Tensor,
        diffusion_timestep: Optional[torch.Tensor],
        frame_offset: int = 0,
    ) -> tuple[torch.Tensor, int, int, int, int, torch.Tensor]:
        """
        Encode input video to token grid with positional/timestep embeddings.
        Returns:
            tokens [B, T, N, D], hp, wp, h, w, temporal_positions [T]
        """
        b, c, t, h, w = x.shape
        if frame_offset + t > self.max_frames:
            raise ValueError(
                f"frame_offset({frame_offset}) + input_frames({t}) exceeds max_frames({self.max_frames})"
            )

        x_bt = rearrange(x, "b c t h w -> (b t) c h w")
        x_bt = self.stem(x_bt)
        tokens = self.patch_embed(x_bt)
        if tokens.ndim == 4:
            tokens = tokens.flatten(2).transpose(1, 2)
        hp, wp = h // self.patch_size, w // self.patch_size
        tokens = rearrange(tokens, "(b t) n d -> b t n d", b=b, t=t)

        spatial_pe = self._resize_spatial_pos_embed(hp=hp, wp=wp).to(tokens.dtype).to(tokens.device)
        temporal_pe = self.temporal_pos_embed[:, frame_offset : frame_offset + t, :].to(tokens.dtype).to(tokens.device)
        tokens = tokens + spatial_pe[:, None, :, :] + temporal_pe[:, :, None, :]

        if diffusion_timestep is not None:
            ts = self._expand_timestep_to_frames(diffusion_timestep, batch_size=b, frames=t)
            ts_emb = self.timestep_embedder(ts).to(tokens.dtype)
            tokens = tokens + ts_emb[:, :, None, :]

        temporal_positions = torch.arange(
            frame_offset,
            frame_offset + t,
            device=x.device,
            dtype=torch.long,
        )
        return tokens, hp, wp, h, w, temporal_positions

    def clear_context_cache(self) -> None:
        self._context_cache = None

    def build_mrope3d_position_ids(
        self,
        frames: int,
        hp: int,
        wp: int,
        frame_offset: int = 0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        return InterleavedMRoPE3D.build_3d_position_ids(
            frames=frames,
            hp=hp,
            wp=wp,
            frame_offset=frame_offset,
            device=device,
        )

    def _decode_target_tokens(
        self,
        target_tokens: torch.Tensor,
        hp: int,
        wp: int,
        h: int,
        w: int,
        batch_size: int,
        predict_frames: int,
        return_modality_dict: bool = True,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        feat = rearrange(target_tokens, "b p (hp wp) d -> (b p) d hp wp", hp=hp, wp=wp)
        radar = self.radar_decoder(feat, target_hw=(h, w))
        satellite = self.satellite_decoder(feat, target_hw=(h, w))
        rain = self.rain_decoder(feat, target_hw=(h, w))

        radar = rearrange(radar, "(b p) c h w -> b c p h w", b=batch_size, p=predict_frames)
        satellite = rearrange(satellite, "(b p) c h w -> b c p h w", b=batch_size, p=predict_frames)
        rain = rearrange(rain, "(b p) c h w -> b c p h w", b=batch_size, p=predict_frames)
        if return_modality_dict:
            return {
                "radar": radar,
                "satellite": satellite,
                "rain": rain,
            }
        return rain

    @torch.no_grad()
    def build_context_cache(
        self,
        context_x: torch.Tensor,
        context_timestep: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Build inference-only cache from clean/fixed context frames.
        Only context is cached; target frames are never cached across different timesteps.
        """
        self.eval()
        if context_x.ndim != 5:
            raise ValueError(f"context_x must be [B,C,T,H,W], got {tuple(context_x.shape)}")
        b, c, tc, h, w = context_x.shape
        if c != self.in_channels:
            raise ValueError(f"in_channels mismatch: expect {self.in_channels}, got {c}")

        tokens, hp, wp, _, _, context_temporal_positions = self._encode_tokens(
            x=context_x,
            diffusion_timestep=context_timestep,
            frame_offset=0,
        )
        context_input_by_block = [tokens]
        context_temporal_kv_by_block = []
        for block in self.blocks:
            tokens, context_temporal_kv = block.encode_context(
                tokens,
                hp=hp,
                wp=wp,
                temporal_positions=context_temporal_positions,
            )
            context_temporal_kv_by_block.append(context_temporal_kv)
            context_input_by_block.append(tokens)

        self._context_cache = {
            "context_input_by_block": [t.detach() for t in context_input_by_block],
            "context_temporal_kv_by_block": [t.detach() for t in context_temporal_kv_by_block],
            "batch_size": b,
            "context_frames": tc,
            "h": h,
            "w": w,
            "hp": hp,
            "wp": wp,
            "context_temporal_positions": context_temporal_positions.detach(),
            "dtype": context_x.dtype,
            "device": context_x.device,
        }

    @torch.no_grad()
    def append_to_context_cache(
        self,
        new_context_x: torch.Tensor,
        context_timestep: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Append newly available clean frames into context cache.
        This is inference-only and assumes new frames are fixed/clean context.
        """
        if self._context_cache is None:
            self.build_context_cache(new_context_x, context_timestep=context_timestep)
            return

        cache = self._context_cache
        if new_context_x.ndim != 5:
            raise ValueError(f"new_context_x must be [B,C,T,H,W], got {tuple(new_context_x.shape)}")
        b, c, t_new, h, w = new_context_x.shape
        if b != cache["batch_size"]:
            raise ValueError(f"batch size mismatch: cache={cache['batch_size']}, new={b}")
        if c != self.in_channels:
            raise ValueError(f"in_channels mismatch: expect {self.in_channels}, got {c}")
        if h != cache["h"] or w != cache["w"]:
            raise ValueError(f"spatial shape mismatch: cache=({cache['h']},{cache['w']}), new=({h},{w})")
        if cache["context_frames"] + t_new > self.max_frames:
            raise ValueError(
                f"context cache overflow: existing={cache['context_frames']}, "
                f"new={t_new}, max_frames={self.max_frames}"
            )

        new_tokens, hp, wp, _, _, new_temporal_positions = self._encode_tokens(
            x=new_context_x,
            diffusion_timestep=context_timestep,
            frame_offset=cache["context_frames"],
        )
        if hp != cache["hp"] or wp != cache["wp"]:
            raise ValueError(f"patch grid mismatch: cache=({cache['hp']},{cache['wp']}), new=({hp},{wp})")

        old_context_input_by_block = cache["context_input_by_block"]
        old_context_temporal_kv_by_block = cache["context_temporal_kv_by_block"]
        old_context_temporal_positions = cache["context_temporal_positions"]

        new_context_input_by_block = [torch.cat([old_context_input_by_block[0], new_tokens.detach()], dim=1)]
        new_context_temporal_kv_by_block = []

        current = new_tokens
        for i, block in enumerate(self.blocks):
            context_kv_i = old_context_temporal_kv_by_block[i]
            # only update new part, using old context as KV.
            current, current_spatial = block.forward_target_with_context(
                target=current,
                context_temporal_kv=context_kv_i,
                hp=hp,
                wp=wp,
                target_temporal_positions=new_temporal_positions,
                context_temporal_positions=old_context_temporal_positions,
                strict_target_isolation=False,
                return_spatial=True,
            )
            new_context_temporal_kv_by_block.append(
                torch.cat([context_kv_i, current_spatial.detach()], dim=1)
            )
            new_context_input_by_block.append(
                torch.cat([old_context_input_by_block[i + 1], current.detach()], dim=1)
            )

        cache["context_input_by_block"] = new_context_input_by_block
        cache["context_temporal_kv_by_block"] = new_context_temporal_kv_by_block
        cache["context_temporal_positions"] = torch.cat(
            [old_context_temporal_positions, new_temporal_positions.detach()],
            dim=0,
        )
        cache["context_frames"] = cache["context_frames"] + t_new
        self._context_cache = cache

    def forward_with_context_cache(
        self,
        target_x: torch.Tensor,
        target_timestep: Optional[torch.Tensor] = None,
        predict_frames: Optional[int] = None,
        strict_target_isolation: bool = False,
        return_modality_dict: bool = True,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """
        Inference-only forward using cached context.
        Context cache should be built from clean/fixed history frames.
        """
        if self._context_cache is None:
            raise RuntimeError("Context cache is empty. Call build_context_cache(...) first.")

        cache = self._context_cache
        if target_x.ndim != 5:
            raise ValueError(f"target_x must be [B,C,T,H,W], got {tuple(target_x.shape)}")
        b, c, tt, h, w = target_x.shape
        if b != cache["batch_size"]:
            raise ValueError(f"batch size mismatch: cache={cache['batch_size']}, target={b}")
        if c != self.in_channels:
            raise ValueError(f"in_channels mismatch: expect {self.in_channels}, got {c}")
        if h != cache["h"] or w != cache["w"]:
            raise ValueError(f"spatial shape mismatch: cache=({cache['h']},{cache['w']}), target=({h},{w})")

        context_frames = cache["context_frames"]
        if predict_frames is None:
            predict_frames = tt
        if predict_frames <= 0 or predict_frames > tt:
            raise ValueError(f"predict_frames should be in [1, {tt}], got {predict_frames}")

        target, hp, wp, _, _, target_temporal_positions = self._encode_tokens(
            x=target_x,
            diffusion_timestep=target_timestep,
            frame_offset=context_frames,
        )
        if hp != cache["hp"] or wp != cache["wp"]:
            raise ValueError(f"patch grid mismatch: cache=({cache['hp']},{cache['wp']}), target=({hp},{wp})")

        for i, block in enumerate(self.blocks):
            target = block.forward_target_with_context(
                target=target,
                context_temporal_kv=cache["context_temporal_kv_by_block"][i],
                hp=hp,
                wp=wp,
                target_temporal_positions=target_temporal_positions,
                context_temporal_positions=cache["context_temporal_positions"],
                strict_target_isolation=strict_target_isolation,
            )
        target = self.norm(target)

        target = target[:, -predict_frames:, :, :]
        return self._decode_target_tokens(
            target_tokens=target,
            hp=hp,
            wp=wp,
            h=h,
            w=w,
            batch_size=b,
            predict_frames=predict_frames,
            return_modality_dict=return_modality_dict,
        )

    def forward(
        self,
        x: torch.Tensor,
        diffusion_timestep: Optional[torch.Tensor] = None,
        predict_frames: int = 1,
        strict_target_isolation: bool = False,
        return_modality_dict: bool = True,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """
        Args:
            x: [B, C, T, H, W]
            diffusion_timestep:
                optional [B], [B, 1], [B, Lt] timestep(s), usually for latest noisy target frames.
            predict_frames:
                decode only the latest frames, default 1 for frame-wise AR.
            strict_target_isolation:
                if True, disallow attention among target frames (within the same call),
                i.e. each target frame only attends to context + itself.
        Returns:
            if return_modality_dict:
                {
                    "radar": [B, C_radar, predict_frames, H, W],
                    "satellite": [B, C_satellite, predict_frames, H, W],
                    "rain": [B, C_rain, predict_frames, H, W],
                }
            else:
                rain tensor [B, C_rain, predict_frames, H, W]
        """
        if x.ndim != 5:
            raise ValueError(f"Expected x to be 5D [B,C,T,H,W], got shape={tuple(x.shape)}")

        b, c, t, h, w = x.shape
        if c != self.in_channels:
            raise ValueError(f"in_channels mismatch: model expects {self.in_channels}, got {c}")
        if t > self.max_frames:
            raise ValueError(f"Input frames {t} exceed max_frames {self.max_frames}")
        if predict_frames <= 0 or predict_frames > t:
            raise ValueError(f"predict_frames should be in [1, {t}], got {predict_frames}")
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(
                f"Input H/W must be divisible by patch_size={self.patch_size}, got H={h}, W={w}"
            )

        tokens, hp, wp, _, _, temporal_positions = self._encode_tokens(
            x=x,
            diffusion_timestep=diffusion_timestep,
            frame_offset=0,
        )

        # 3) Causal spatiotemporal transformer.
        context_frames = t - predict_frames
        for block in self.blocks:
            tokens = block(
                tokens,
                context_frames=context_frames,
                strict_target_isolation=strict_target_isolation,
                hp=hp,
                wp=wp,
                temporal_positions=temporal_positions,
            )
        tokens = self.norm(tokens)

        # 4) Decode only target frames.
        tokens = tokens[:, -predict_frames:, :, :]
        return self._decode_target_tokens(
            target_tokens=tokens,
            hp=hp,
            wp=wp,
            h=h,
            w=w,
            batch_size=b,
            predict_frames=predict_frames,
            return_modality_dict=return_modality_dict,
        )

    @staticmethod
    def _expand_timestep_to_bt(
        timestep: Optional[torch.Tensor],
        batch_size: int,
        frames: int,
        device: torch.device,
        name: str,
    ) -> torch.Tensor:
        if frames <= 0:
            return torch.zeros((batch_size, 0), device=device, dtype=torch.float32)
        if timestep is None:
            return torch.zeros((batch_size, frames), device=device, dtype=torch.float32)

        ts = timestep.to(device=device)
        if ts.ndim == 0:
            return ts.float().view(1, 1).expand(batch_size, frames)
        if ts.ndim == 1:
            if ts.shape[0] == 1:
                return ts.float().view(1, 1).expand(batch_size, frames)
            if ts.shape[0] == batch_size:
                return ts.float().view(batch_size, 1).expand(batch_size, frames)
            raise ValueError(
                f"{name} with ndim=1 must be length 1 or B={batch_size}, got shape {tuple(ts.shape)}"
            )
        if ts.ndim == 2:
            if ts.shape == (batch_size, frames):
                return ts.float()
            if ts.shape == (batch_size, 1):
                return ts.float().expand(batch_size, frames)
            if ts.shape == (1, frames):
                return ts.float().expand(batch_size, frames)
            raise ValueError(
                f"{name} with ndim=2 must be [B,T] or [B,1] or [1,T], got shape {tuple(ts.shape)}"
            )
        raise ValueError(f"{name} must be scalar/[B]/[B,T], got ndim={ts.ndim}")

    def forward_ar(
        self,
        target_x: torch.Tensor,
        target_timestep: Optional[torch.Tensor] = None,
        context_x: Optional[torch.Tensor] = None,
        context_timestep: Optional[torch.Tensor] = None,
        predict_frames: Optional[int] = None,
        strict_target_isolation: bool = False,
        return_modality_dict: bool = True,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """
        Autoregressive-friendly API:
        - `context_x` and `target_x` are passed separately.
        - concatenation and timestep assembly are handled inside network.

        Args:
            context_x: optional clean/noisy history, [B,C,Tc,H,W]
            target_x: noisy target block, [B,C,Tt,H,W]
            context_timestep: optional context timestep(s), scalar/[B]/[B,Tc]
            target_timestep: optional target timestep(s), scalar/[B]/[B,Tt]
            predict_frames: defaults to Tt
        """
        if target_x.ndim != 5:
            raise ValueError(f"target_x must be [B,C,T,H,W], got {tuple(target_x.shape)}")
        b, c, tt, h, w = target_x.shape
        if tt <= 0:
            raise ValueError("target_x must contain at least 1 frame.")

        if context_x is None:
            x = target_x
            context_frames = 0
        else:
            if context_x.ndim != 5:
                raise ValueError(f"context_x must be [B,C,T,H,W], got {tuple(context_x.shape)}")
            bc, cc, tc, hc, wc = context_x.shape
            if bc != b or cc != c or hc != h or wc != w:
                raise ValueError(
                    "context_x and target_x shape mismatch: "
                    f"context={tuple(context_x.shape)}, target={tuple(target_x.shape)}"
                )
            x = torch.cat([context_x, target_x], dim=2)
            context_frames = tc

        target_frames = tt if predict_frames is None else int(predict_frames)
        if target_frames <= 0 or target_frames > tt:
            raise ValueError(f"predict_frames should be in [1, {tt}], got {target_frames}")

        context_ts = self._expand_timestep_to_bt(
            timestep=context_timestep,
            batch_size=b,
            frames=context_frames,
            device=x.device,
            name="context_timestep",
        )
        target_ts = self._expand_timestep_to_bt(
            timestep=target_timestep,
            batch_size=b,
            frames=tt,
            device=x.device,
            name="target_timestep",
        )
        diffusion_timestep = torch.cat([context_ts, target_ts], dim=1)

        return self.forward(
            x=x,
            diffusion_timestep=diffusion_timestep,
            predict_frames=target_frames,
            strict_target_isolation=strict_target_isolation,
            return_modality_dict=return_modality_dict,
        )

    def forward_modalities(
        self,
        radar: torch.Tensor,
        satellite: torch.Tensor,
        rain: torch.Tensor,
        diffusion_timestep: Optional[torch.Tensor] = None,
        predict_frames: int = 1,
        strict_target_isolation: bool = False,
        return_modality_dict: bool = True,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        """
        Convenience wrapper for multimodal usage.
        Expected shapes:
            radar:     [B, C_r, T, H, W]
            satellite: [B, C_s, T, H, W]
            rain:      [B, C_y, T, H, W]
        """
        x = torch.cat([radar, satellite, rain], dim=1)
        return self.forward(
            x=x,
            diffusion_timestep=diffusion_timestep,
            predict_frames=predict_frames,
            strict_target_isolation=strict_target_isolation,
            return_modality_dict=return_modality_dict,
        )


if __name__ == "__main__":
    model = RainCausalPatchTransformerDiffusion(
        in_channels=12,
        out_channels=1,
        radar_out_channels=1,
        satellite_out_channels=10,
        rain_out_channels=1,
        input_size=256,
        patch_size=4,
        dim=384,
        depth=6,
        num_heads=6,
        max_frames=16,
    )
    x = torch.randn(2, 12, 6, 256, 256)
    t = torch.randint(low=1, high=1000, size=(2, 1))
    with torch.no_grad():
        y = model(x, diffusion_timestep=t, predict_frames=1, return_modality_dict=True)
    print("radar:", y["radar"].shape)
    print("satellite:", y["satellite"].shape)
    print("rain:", y["rain"].shape)
