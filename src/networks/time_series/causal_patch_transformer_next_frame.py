import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

from src.networks.modules.reconstruction import (
    ConvStem,
    ResNetBlock3D,
    ResNetDecoder2D,
    ResNetDecoder3D,
    build_3d_conv_layer,
)
from src.networks.time_series.causal_patch_transformer_diffusion import (
    CausalSpatiotemporalBlock,
)


def _valid_gn_groups(channels: int, max_groups: int = 32) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class _SpatialModalityGate(nn.Module):
    def __init__(self, channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3 * channels, hidden_channels, 1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, groups=hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 2, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self, z_radar: torch.Tensor, z_satellite: torch.Tensor, z_rain: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.cat([z_radar, z_satellite, z_rain], dim=1)
        b, _, t, _, _ = features.shape
        features = rearrange(features, "b d t h w -> (b t) d h w")
        logits = rearrange(self.net(features), "(b t) d h w -> b d t h w", b=b, t=t)
        gates = 1 + torch.tanh(logits)
        return gates[:, :1], gates[:, 1:2]


class _CrossModalConditioner(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        modality_channels: tuple[int, int, int],
        patch_size: int,
        frame_patch_size: int,
    ) -> None:
        super().__init__()
        self.frame_patch_size = frame_patch_size
        self.patch_size = patch_size
        self.adapters = nn.ModuleList(
            [
                nn.Conv3d(
                    in_channels=channels,
                    out_channels=dim,
                    kernel_size=(frame_patch_size, patch_size, patch_size),
                    stride=(frame_patch_size, patch_size, patch_size),
                )
                for channels in modality_channels
            ]
        )
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, target_tokens: torch.Tensor, context_x: torch.Tensor) -> torch.Tensor:
        if context_x.shape[2] < self.frame_patch_size:
            return target_tokens

        modality_inputs = torch.split(context_x, [adapter.in_channels for adapter in self.adapters], dim=1)
        modality_tokens = []
        for adapter, modality_input in zip(self.adapters, modality_inputs, strict=True):
            encoded = adapter(modality_input)
            modality_tokens.append(rearrange(encoded, "b d t h w -> b t (h w) d"))

        b, target_t, n, d = target_tokens.shape
        memory = torch.stack(modality_tokens, dim=3)
        memory = rearrange(memory, "b t n m d -> (b n) (t m) d")
        query = rearrange(target_tokens, "b t n d -> (b n) t d")
        conditioned = self.attention(
            self.query_norm(query),
            self.memory_norm(memory),
            self.memory_norm(memory),
            need_weights=False,
        )[0]
        conditioned = rearrange(conditioned, "(b n) t d -> b t n d", b=b, n=n, t=target_t, d=d)
        return target_tokens + torch.tanh(self.gate).to(target_tokens.dtype) * conditioned


class _WindowAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, window_size: int, shift_size: int) -> None:
        super().__init__()
        self.heads = heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm_attention = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        b, t, n, d = x.shape
        if n != height * width:
            raise ValueError(f"Expected {height * width} tokens, got {n}")
        if height % self.window_size != 0 or width % self.window_size != 0:
            raise ValueError(
                f"Feature grid {(height, width)} must be divisible by local window size {self.window_size}"
            )

        batch_time = b * t
        windows_h = height // self.window_size
        windows_w = width // self.window_size
        features = rearrange(x, "b t (h w) d -> (b t) h w d", h=height, w=width)
        if self.shift_size > 0:
            features = torch.roll(features, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        windows = rearrange(
            features,
            "bt (nh wh) (nw ww) d -> (bt nh nw) (wh ww) d",
            nh=windows_h,
            nw=windows_w,
            wh=self.window_size,
            ww=self.window_size,
        )
        attention_mask = None
        if self.shift_size > 0:
            region_ids = torch.zeros((1, height, width, 1), device=x.device, dtype=torch.long)
            h_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            region = 0
            for h_slice in h_slices:
                for w_slice in w_slices:
                    region_ids[:, h_slice, w_slice, :] = region
                    region += 1
            region_ids = torch.roll(region_ids, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            mask_windows = rearrange(
                region_ids,
                "b (nh wh) (nw ww) c -> (b nh nw) (wh ww) c",
                nh=windows_h,
                nw=windows_w,
                wh=self.window_size,
                ww=self.window_size,
            ).squeeze(-1)
            attention_mask = mask_windows[:, :, None] - mask_windows[:, None, :]
            attention_mask = attention_mask.ne(0).to(dtype=x.dtype) * -100.0
            attention_mask = attention_mask.repeat(batch_time, 1, 1).repeat_interleave(self.heads, dim=0)

        attended = self.attention(
            self.norm_attention(windows),
            self.norm_attention(windows),
            self.norm_attention(windows),
            attn_mask=attention_mask,
            need_weights=False,
        )[0]
        windows = windows + attended
        windows = windows + self.mlp(self.norm_mlp(windows))
        features = rearrange(
            windows,
            "(bt nh nw) (wh ww) d -> bt (nh wh) (nw ww) d",
            bt=batch_time,
            nh=windows_h,
            nw=windows_w,
            wh=self.window_size,
            ww=self.window_size,
        )
        if self.shift_size > 0:
            features = torch.roll(features, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        return rearrange(features, "(b t) h w d -> b t (h w) d", b=b, t=t, h=height, w=width)


class _ShiftedWindowRefiner(nn.Module):
    def __init__(self, dim: int, heads: int, window_size: int) -> None:
        super().__init__()
        if window_size < 2:
            raise ValueError(f"local window size must be at least 2, got {window_size}")
        self.blocks = nn.ModuleList(
            [
                _WindowAttentionBlock(dim, heads, window_size, shift_size=0),
                _WindowAttentionBlock(dim, heads, window_size, shift_size=window_size // 2),
            ]
        )
        self.gate = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        refined = x
        for block in self.blocks:
            refined = block(refined, height=height, width=width)
        return x + torch.tanh(self.gate).to(x.dtype) * (refined - x)


class _ResNetDownsampleStage3D(nn.Module):
    def __init__(
        self,
        channels: int,
        scale_t: int,
        scale_hw: int,
        dropout: float,
        pad_mode: str,
        causal: bool,
        conv_style: str,
    ) -> None:
        super().__init__()
        self.downsample = build_3d_conv_layer(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=(scale_t, scale_hw, scale_hw),
            padding_mode=pad_mode,
            causal=causal,
            conv_style=conv_style,
        )
        self.norm = nn.GroupNorm(num_groups=_valid_gn_groups(channels), num_channels=channels)
        self.act = nn.SiLU()
        self.block = ResNetBlock3D(
            in_channels=channels,
            out_channels=channels,
            dropout=dropout,
            padding_mode=pad_mode,
            causal=causal,
            conv_style=conv_style,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.act(self.norm(x))
        return self.block(x)


class _LightweightResNetEncoder3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        stem_channels: int,
        spatial_downsample_stages: int,
        temporal_downsample_stages: int,
        dropout: float,
        pad_mode: str,
        causal: bool,
        conv_style: str,
        activation_checkpoint: bool,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            build_3d_conv_layer(
                in_channels=in_channels,
                out_channels=stem_channels,
                kernel_size=3,
                padding_mode=pad_mode,
                causal=causal,
                conv_style=conv_style,
            ),
            nn.GroupNorm(num_groups=_valid_gn_groups(stem_channels), num_channels=stem_channels),
            nn.SiLU(),
        )

        stage_count = max(spatial_downsample_stages, temporal_downsample_stages)
        stages: list[nn.Module] = []
        for stage_idx in range(stage_count):
            scale_t = 2 if stage_idx < temporal_downsample_stages else 1
            scale_hw = 2 if stage_idx < spatial_downsample_stages else 1
            stage = _ResNetDownsampleStage3D(
                channels=stem_channels,
                scale_t=scale_t,
                scale_hw=scale_hw,
                dropout=dropout,
                pad_mode=pad_mode,
                causal=causal,
                conv_style=conv_style,
            )
            if activation_checkpoint:
                stage = CheckpointWrapper(stage)
            stages.append(stage)
        self.stages = nn.ModuleList(stages)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        for stage in self.stages:
            x = stage(x)
        return x


class RainCausalPatchTransformerNextFrame(nn.Module):
    """
    Time-series next-frame / next-block prediction backbone.

    This model keeps the same causal patch-transformer + modality decoders layout,
    but removes diffusion-specific timestep/noise semantics.
    """

    def __init__(
        self,
        in_channels: int = 12,
        out_channels: int | None = 1,
        radar_out_channels: int = 1,
        satellite_out_channels: int = 10,
        rain_out_channels: int | None = None,
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
        decoder_type: str = "3d",
        frame_patch_size: int = 1,
        encoder_type: str = "patch",
        encoder_spatial_downsample_stages: int = 0,
        encoder_temporal_downsample_stages: int = 0,
        encoder_causal: bool = True,
        encoder_conv_style: str = "wan_factorized",
        use_modality_mask_token: bool = True,
        stem_pad_mode: str = "zeros",
        decoder_pad_mode: str = "zeros",
        decoder_upsample_mode: str = "nearest_conv",
        decoder_k_size: int | None = None,
        decoder_causal: bool = True,
        decoder_condition_mode: str = "none",
        decoder_output_mode: str = "pixels",
        use_time_embedding: bool = False,
        activation_checkpoint: bool = False,
        cross_modal_adapter_enabled: bool = False,
        cross_modal_adapter_heads: int | None = None,
        local_window_refiner_enabled: bool = False,
        local_window_size: int = 7,
        local_window_heads: int | None = None,
        spatial_modality_gate_enabled: bool = False,
        spatial_modality_gate_hidden_channels: int = 32,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        if rain_out_channels is None:
            rain_out_channels = 1 if out_channels is None else out_channels

        if spatial_modality_gate_enabled:
            if str(encoder_type).lower() != "patch":
                raise ValueError("spatial modality gate requires encoder_type='patch'")
            if frame_patch_size != 1:
                raise ValueError("spatial modality gate requires frame_patch_size=1")
            if spatial_modality_gate_hidden_channels <= 0:
                raise ValueError("spatial_modality_gate_hidden_channels must be > 0")
            modality_channels = (radar_out_channels, satellite_out_channels, rain_out_channels)
            if any(channels <= 0 for channels in modality_channels):
                raise ValueError("spatial modality gate requires positive radar/satellite/rain channels")
            if sum(modality_channels) != in_channels:
                raise ValueError(
                    "spatial modality gate requires in_channels == "
                    "radar_out_channels + satellite_out_channels + rain_out_channels"
                )

        self.radar_out_channels = radar_out_channels
        self.satellite_out_channels = satellite_out_channels
        self.rain_out_channels = rain_out_channels
        self.out_channels = rain_out_channels
        self.use_modality_mask_token = bool(use_modality_mask_token)
        self.radar_mask_token = nn.Parameter(torch.zeros(1, self.radar_out_channels, 1, 1, 1))
        self.satellite_mask_token = nn.Parameter(torch.zeros(1, self.satellite_out_channels, 1, 1, 1))
        self.rain_mask_token = nn.Parameter(torch.zeros(1, self.rain_out_channels, 1, 1, 1))

        self.input_size = input_size
        self.patch_size = int(patch_size)
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be > 0, got {self.patch_size}")
        self.max_frames = max_frames
        self.dim = dim
        self.frame_patch_size = int(frame_patch_size)
        if self.frame_patch_size <= 0:
            raise ValueError(f"frame_patch_size must be > 0, got {self.frame_patch_size}")
        self.encoder_type = str(encoder_type).lower()
        if self.encoder_type not in {"patch", "resnet"}:
            raise ValueError(f"encoder_type must be 'patch' or 'resnet', got {encoder_type}")
        self.encoder_spatial_downsample_stages = int(encoder_spatial_downsample_stages)
        self.encoder_temporal_downsample_stages = int(encoder_temporal_downsample_stages)
        if self.encoder_spatial_downsample_stages < 0:
            raise ValueError(
                f"encoder_spatial_downsample_stages must be >= 0, got {self.encoder_spatial_downsample_stages}"
            )
        if self.encoder_temporal_downsample_stages < 0:
            raise ValueError(
                f"encoder_temporal_downsample_stages must be >= 0, got {self.encoder_temporal_downsample_stages}"
            )
        if self.encoder_type == "resnet":
            expected_spatial_scale = 2**self.encoder_spatial_downsample_stages
            expected_temporal_scale = 2**self.encoder_temporal_downsample_stages
            if self.patch_size != expected_spatial_scale:
                raise ValueError(
                    "resnet encoder requires patch_size == 2**encoder_spatial_downsample_stages, "
                    f"got patch_size={self.patch_size}, encoder_spatial_downsample_stages="
                    f"{self.encoder_spatial_downsample_stages}"
                )
            if self.frame_patch_size != expected_temporal_scale:
                raise ValueError(
                    "resnet encoder requires frame_patch_size == 2**encoder_temporal_downsample_stages, "
                    f"got frame_patch_size={self.frame_patch_size}, encoder_temporal_downsample_stages="
                    f"{self.encoder_temporal_downsample_stages}"
                )
        self.encoder_causal = bool(encoder_causal)
        self.encoder_conv_style = str(encoder_conv_style).lower()
        if self.encoder_conv_style not in {"full3d", "wan_factorized"}:
            raise ValueError(
                f"encoder_conv_style must be 'full3d' or 'wan_factorized', got {encoder_conv_style}"
            )
        if self.max_frames % self.frame_patch_size != 0:
            raise ValueError(
                f"max_frames({self.max_frames}) must be divisible by frame_patch_size({self.frame_patch_size})."
            )
        self.max_token_frames = self.max_frames // self.frame_patch_size
        self.decoder_type = str(decoder_type).lower()
        if self.decoder_type not in {"2d", "3d"}:
            raise ValueError(f"decoder_type must be '2d' or '3d', got {decoder_type}")
        if self.decoder_type == "2d" and self.frame_patch_size != 1:
            raise ValueError("decoder_type='2d' only supports frame_patch_size=1.")
        self.decoder_k_size: int | None = None if decoder_k_size is None else int(decoder_k_size)
        if self.decoder_k_size is not None:
            if self.decoder_k_size <= 0:
                raise ValueError(f"decoder_k_size must be > 0, got {self.decoder_k_size}")
            if self.decoder_k_size % 2 == 0:
                raise ValueError(
                    f"decoder_k_size must be odd for symmetric padding, got {self.decoder_k_size}"
                )
        self.decoder_condition_mode = str(decoder_condition_mode).lower()
        if self.decoder_condition_mode not in {"none", "film"}:
            raise ValueError(f"decoder_condition_mode must be 'none' or 'film', got {decoder_condition_mode}")
        self.decoder_output_mode = str(decoder_output_mode).lower()
        if self.decoder_output_mode not in {"pixels", "residual"}:
            raise ValueError(f"decoder_output_mode must be 'pixels' or 'residual', got {decoder_output_mode}")
        self.decoder_causal = bool(decoder_causal)
        self.use_time_embedding = bool(use_time_embedding)
        self.cross_modal_adapter_enabled = bool(cross_modal_adapter_enabled)
        self.local_window_refiner_enabled = bool(local_window_refiner_enabled)

        self.patch_embed: nn.Conv3d | None = None
        self.stem: ConvStem | None = None
        self.resnet_encoder: _LightweightResNetEncoder3D | None = None
        if self.encoder_type == "patch":
            self.patch_embed = nn.Conv3d(
                in_channels=in_channels,
                out_channels=stem_channels,
                kernel_size=(self.frame_patch_size, self.patch_size, self.patch_size),
                stride=(self.frame_patch_size, self.patch_size, self.patch_size),
            )
            self.stem = ConvStem(in_channels=stem_channels, stem_channels=stem_channels, pad_mode=stem_pad_mode)
        else:
            self.resnet_encoder = _LightweightResNetEncoder3D(
                in_channels=in_channels,
                stem_channels=stem_channels,
                spatial_downsample_stages=self.encoder_spatial_downsample_stages,
                temporal_downsample_stages=self.encoder_temporal_downsample_stages,
                dropout=dropout,
                pad_mode=stem_pad_mode,
                causal=self.encoder_causal,
                conv_style=self.encoder_conv_style,
                activation_checkpoint=activation_checkpoint,
            )
        self.token_proj = nn.Identity() if stem_channels == dim else nn.Linear(stem_channels, dim)

        self.base_grid = input_size // self.patch_size
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, self.base_grid * self.base_grid, dim))
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, self.max_token_frames, dim))
        self.time_embed_proj: nn.Module | None = None
        if self.use_time_embedding:
            self.time_embed_proj = nn.Sequential(
                nn.Linear(2, dim),
                nn.SiLU(),
                nn.Linear(dim, dim),
            )

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        blocks = []
        for i in range(depth):
            block = CausalSpatiotemporalBlock(
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
                block = CheckpointWrapper(block)
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(dim)
        self.modality_channels = {
            "radar": self.radar_out_channels,
            "satellite": self.satellite_out_channels,
            "rain": self.rain_out_channels,
        }
        self.cross_modal_adapter: _CrossModalConditioner | None = None
        if self.cross_modal_adapter_enabled:
            cross_heads = num_heads if cross_modal_adapter_heads is None else int(cross_modal_adapter_heads)
            self.cross_modal_adapter = _CrossModalConditioner(
                dim=dim,
                heads=cross_heads,
                modality_channels=(self.radar_out_channels, self.satellite_out_channels, self.rain_out_channels),
                patch_size=self.patch_size,
                frame_patch_size=self.frame_patch_size,
            )
        self.local_window_refiner: _ShiftedWindowRefiner | None = None
        if self.local_window_refiner_enabled:
            local_heads = num_heads if local_window_heads is None else int(local_window_heads)
            self.local_window_refiner = _ShiftedWindowRefiner(
                dim=dim,
                heads=local_heads,
                window_size=int(local_window_size),
            )
        if sum(self.modality_channels.values()) != self.in_channels:
            if self.decoder_condition_mode == "film" or self.decoder_output_mode == "residual":
                raise ValueError(
                    "decoder_condition_mode='film' or decoder_output_mode='residual' requires "
                    "in_channels == radar_out_channels + satellite_out_channels + rain_out_channels"
                )
        use_film = self.decoder_condition_mode == "film"
        decoder_k_size_2d = 3 if self.decoder_k_size is None else self.decoder_k_size
        decoder_k_size_3d = 7 if self.decoder_k_size is None else self.decoder_k_size

        if self.decoder_type == "2d":
            self.radar_decoder = ResNetDecoder2D(
                in_channels=dim,
                out_channels=radar_out_channels,
                patch_size=self.patch_size,
                base_channels=decoder_base_channels,
                dropout=dropout,
                padding_mode=decoder_pad_mode,
                upsample_mode=decoder_upsample_mode,
                cond_channels=self.radar_out_channels,
                use_film=use_film,
                k_size=decoder_k_size_2d,
                activation_checkpoint=activation_checkpoint,
            )
            self.satellite_decoder = ResNetDecoder2D(
                in_channels=dim,
                out_channels=satellite_out_channels,
                patch_size=self.patch_size,
                base_channels=decoder_base_channels,
                dropout=dropout,
                padding_mode=decoder_pad_mode,
                upsample_mode=decoder_upsample_mode,
                cond_channels=self.satellite_out_channels,
                use_film=use_film,
                k_size=decoder_k_size_2d,
                activation_checkpoint=activation_checkpoint,
            )
            self.rain_decoder = ResNetDecoder2D(
                in_channels=dim,
                out_channels=rain_out_channels,
                patch_size=self.patch_size,
                base_channels=decoder_base_channels,
                dropout=dropout,
                padding_mode=decoder_pad_mode,
                upsample_mode=decoder_upsample_mode,
                cond_channels=self.rain_out_channels,
                use_film=use_film,
                k_size=decoder_k_size_2d,
                activation_checkpoint=activation_checkpoint,
            )
        else:
            self.radar_decoder = ResNetDecoder3D(
                in_channels=dim,
                out_channels=radar_out_channels,
                patch_size=self.patch_size,
                frame_patch_size=self.frame_patch_size,
                base_channels=decoder_base_channels,
                dropout=dropout,
                padding_mode=decoder_pad_mode,
                upsample_mode=decoder_upsample_mode,
                cond_channels=self.radar_out_channels,
                use_film=use_film,
                activation_checkpoint=activation_checkpoint,
                causal=self.decoder_causal,
                conv_style=self.encoder_conv_style,
                k_size=decoder_k_size_3d,
            )
            self.satellite_decoder = ResNetDecoder3D(
                in_channels=dim,
                out_channels=satellite_out_channels,
                patch_size=self.patch_size,
                frame_patch_size=self.frame_patch_size,
                base_channels=decoder_base_channels,
                dropout=dropout,
                padding_mode=decoder_pad_mode,
                upsample_mode=decoder_upsample_mode,
                cond_channels=self.satellite_out_channels,
                use_film=use_film,
                activation_checkpoint=activation_checkpoint,
                causal=self.decoder_causal,
                conv_style=self.encoder_conv_style,
                k_size=decoder_k_size_3d,
            )
            self.rain_decoder = ResNetDecoder3D(
                in_channels=dim,
                out_channels=rain_out_channels,
                patch_size=self.patch_size,
                frame_patch_size=self.frame_patch_size,
                base_channels=decoder_base_channels,
                dropout=dropout,
                padding_mode=decoder_pad_mode,
                upsample_mode=decoder_upsample_mode,
                cond_channels=self.rain_out_channels,
                use_film=use_film,
                activation_checkpoint=activation_checkpoint,
                causal=self.decoder_causal,
                conv_style=self.encoder_conv_style,
                k_size=decoder_k_size_3d,
            )

        self._init_weights()
        self.spatial_modality_gate: _SpatialModalityGate | None = None
        if spatial_modality_gate_enabled:
            self.spatial_modality_gate = _SpatialModalityGate(stem_channels, spatial_modality_gate_hidden_channels)

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.radar_mask_token, std=0.02)
        nn.init.trunc_normal_(self.satellite_mask_token, std=0.02)
        nn.init.trunc_normal_(self.rain_mask_token, std=0.02)

    def _resize_spatial_pos_embed(self, hp: int, wp: int) -> torch.Tensor:
        if hp == self.base_grid and wp == self.base_grid:
            return self.spatial_pos_embed

        pos = self.spatial_pos_embed.reshape(1, self.base_grid, self.base_grid, self.dim).permute(0, 3, 1, 2)
        pos = F.interpolate(pos, size=(hp, wp), mode="bicubic", align_corners=False)
        pos = pos.permute(0, 2, 3, 1).reshape(1, hp * wp, self.dim)
        return pos

    def _encode_tokens(
        self,
        x: torch.Tensor,
        frame_offset: int = 0,
        frame_times: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int, int, int, int, torch.Tensor]:
        b, c, t, h, w = x.shape
        if t % self.frame_patch_size != 0:
            raise ValueError(f"Input T must be divisible by frame_patch_size={self.frame_patch_size}, got T={t}")
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            raise ValueError(f"Input H/W must be divisible by patch_size={self.patch_size}, got H={h}, W={w}")

        token_t = t // self.frame_patch_size
        if frame_offset + token_t > self.max_token_frames:
            raise ValueError(
                f"frame_offset({frame_offset}) + token_frames({token_t}) exceeds max_token_frames({self.max_token_frames})"
            )

        if self.encoder_type == "patch":
            if self.patch_embed is None or self.stem is None:
                raise RuntimeError("patch encoder is not initialized.")
            encoded = self.patch_embed(x)
            if self.spatial_modality_gate is not None:
                modality_contributions: list[torch.Tensor] = []
                channel_start = 0
                for channels in self.modality_channels.values():
                    channel_end = channel_start + channels
                    modality_contributions.append(
                        F.conv3d(
                            x[:, channel_start:channel_end],
                            self.patch_embed.weight[:, channel_start:channel_end],
                            bias=None,
                            stride=self.patch_embed.stride,
                            padding=self.patch_embed.padding,
                            dilation=self.patch_embed.dilation,
                        )
                    )
                    channel_start = channel_end
                z_radar, z_satellite, z_rain = modality_contributions
                gate_radar, gate_satellite = self.spatial_modality_gate(z_radar, z_satellite, z_rain)
                encoded = encoded + (gate_radar - 1) * z_radar + (gate_satellite - 1) * z_satellite
            _, _, tp, hp, wp = encoded.shape
            encoded = rearrange(encoded, "b d tp hp wp -> (b tp) d hp wp")
            encoded = self.stem(encoded)
            tokens = rearrange(encoded, "(b tp) d hp wp -> b tp (hp wp) d", b=b, tp=tp, hp=hp, wp=wp)
        else:
            if self.resnet_encoder is None:
                raise RuntimeError("resnet encoder is not initialized.")
            encoded = self.resnet_encoder(x)
            _, _, tp, hp, wp = encoded.shape
            tokens = rearrange(encoded, "b d tp hp wp -> b tp (hp wp) d")

        expected_hp = h // self.patch_size
        expected_wp = w // self.patch_size
        if tp != token_t or hp != expected_hp or wp != expected_wp:
            raise ValueError(
                "encoder output size mismatch with patch/frame scales: "
                f"got (tp={tp}, hp={hp}, wp={wp}), expected (tp={token_t}, hp={expected_hp}, wp={expected_wp})"
            )
        tokens = self.token_proj(tokens)

        spatial_pe = self._resize_spatial_pos_embed(hp=hp, wp=wp).to(tokens.dtype).to(tokens.device)
        temporal_pe = self.temporal_pos_embed[:, frame_offset : frame_offset + tp, :].to(tokens.dtype).to(tokens.device)
        tokens = tokens + spatial_pe[:, None, :, :] + temporal_pe[:, :, None, :]
        if self.use_time_embedding and frame_times is not None:
            if frame_times.ndim != 2:
                raise ValueError(f"frame_times must be [B,T], got shape={tuple(frame_times.shape)}")
            if int(frame_times.shape[0]) != b or int(frame_times.shape[1]) != t:
                raise ValueError(
                    f"frame_times shape mismatch with x: expected ({b}, {t}), got {tuple(frame_times.shape)}"
                )
            if self.time_embed_proj is None:
                raise RuntimeError("time embedding is enabled but time_embed_proj is not initialized.")
            grouped_time = rearrange(frame_times, "b (tp fp) -> b tp fp", fp=self.frame_patch_size)
            pooled_time = grouped_time.mean(dim=-1)
            time_phase = pooled_time.to(tokens.dtype) * (2.0 * torch.pi)
            time_features = torch.stack([torch.sin(time_phase), torch.cos(time_phase)], dim=-1)
            time_embed = self.time_embed_proj(time_features).unsqueeze(2)
            tokens = tokens + time_embed.to(tokens.dtype).to(tokens.device)

        temporal_positions = torch.arange(frame_offset, frame_offset + tp, device=x.device, dtype=torch.long)
        return tokens, hp, wp, h, w, temporal_positions

    def _split_modalities(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        r = self.radar_out_channels
        s = self.satellite_out_channels
        return {
            "radar": x[:, :r],
            "satellite": x[:, r : r + s],
            "rain": x[:, r + s : r + s + self.rain_out_channels],
        }

    def _apply_context_modality_mask_token(
        self,
        context_x: torch.Tensor,
        context_modality_available: torch.Tensor | None,
    ) -> torch.Tensor:
        if context_modality_available is None or not self.use_modality_mask_token:
            return context_x
        if context_x.ndim != 5:
            raise ValueError(f"context_x must be [B,C,T,H,W], got {tuple(context_x.shape)}")
        if context_modality_available.ndim != 2 or int(context_modality_available.shape[1]) != 3:
            raise ValueError(
                "context_modality_available must be [B,3] with modality order [radar, satellite, rain], "
                f"got shape={tuple(context_modality_available.shape)}"
            )
        if int(context_modality_available.shape[0]) != int(context_x.shape[0]):
            raise ValueError(
                "batch size mismatch between context_x and context_modality_available: "
                f"context_x batch={int(context_x.shape[0])}, availability batch={int(context_modality_available.shape[0])}"
            )

        available = context_modality_available.to(device=context_x.device)
        if available.dtype != torch.bool:
            available = available > 0
        missing = ~available
        b, _, t, h, w = context_x.shape

        radar = context_x[:, : self.radar_out_channels]
        satellite = context_x[:, self.radar_out_channels : self.radar_out_channels + self.satellite_out_channels]
        rain = context_x[:, self.radar_out_channels + self.satellite_out_channels :]

        radar_token = self.radar_mask_token.to(device=context_x.device, dtype=context_x.dtype).expand(b, -1, t, h, w)
        satellite_token = self.satellite_mask_token.to(device=context_x.device, dtype=context_x.dtype).expand(
            b, -1, t, h, w
        )
        rain_token = self.rain_mask_token.to(device=context_x.device, dtype=context_x.dtype).expand(b, -1, t, h, w)

        radar = torch.where(missing[:, 0].view(b, 1, 1, 1, 1), radar_token, radar)
        satellite = torch.where(missing[:, 1].view(b, 1, 1, 1, 1), satellite_token, satellite)
        rain = torch.where(missing[:, 2].view(b, 1, 1, 1, 1), rain_token, rain)
        return torch.cat([radar, satellite, rain], dim=1)

    def _last_context_anchor(self, x: torch.Tensor, context_frames: int) -> dict[str, torch.Tensor] | None:
        if context_frames <= 0:
            return None
        anchor = x[:, :, context_frames - 1 : context_frames]
        return self._split_modalities(anchor)

    def _build_film_condition_sequence(
        self,
        x: torch.Tensor,
        *,
        context_frames: int,
        predict_frames: int,
    ) -> dict[str, torch.Tensor] | None:
        if self.decoder_condition_mode != "film":
            return None
        if context_frames <= 0 or predict_frames <= 0:
            return None
        start = context_frames - 1
        end = start + predict_frames
        if start < 0 or end > int(x.shape[2]) - 1:
            return None
        cond = x[:, :, start:end]
        if int(cond.shape[2]) != predict_frames:
            return None
        return self._split_modalities(cond)

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
        use_film = self.decoder_condition_mode == "film" and film_condition is not None
        if self.decoder_type == "2d":
            feat = rearrange(target_tokens, "b p (hp wp) d -> (b p) d hp wp", hp=hp, wp=wp)
            radar_cond = None
            satellite_cond = None
            rain_cond = None
            if use_film:
                radar_cond = rearrange(
                    film_condition["radar"],
                    "b c p h w -> (b p) c h w",
                )
                satellite_cond = rearrange(
                    film_condition["satellite"],
                    "b c p h w -> (b p) c h w",
                )
                rain_cond = rearrange(
                    film_condition["rain"],
                    "b c p h w -> (b p) c h w",
                )
            radar = self.radar_decoder(feat, target_hw=(h, w), cond=radar_cond)
            satellite = self.satellite_decoder(feat, target_hw=(h, w), cond=satellite_cond)
            rain = self.rain_decoder(feat, target_hw=(h, w), cond=rain_cond)

            radar = rearrange(radar, "(b p) c h w -> b c p h w", b=batch_size, p=predict_frames)
            satellite = rearrange(satellite, "(b p) c h w -> b c p h w", b=batch_size, p=predict_frames)
            rain = rearrange(rain, "(b p) c h w -> b c p h w", b=batch_size, p=predict_frames)
        else:
            feat = rearrange(target_tokens, "b p (hp wp) d -> b d p hp wp", hp=hp, wp=wp)
            out_frames = predict_frames * self.frame_patch_size
            radar_cond = film_condition["radar"] if use_film else None
            satellite_cond = film_condition["satellite"] if use_film else None
            rain_cond = film_condition["rain"] if use_film else None
            radar = self.radar_decoder(feat, target_thw=(out_frames, h, w), cond=radar_cond)
            satellite = self.satellite_decoder(feat, target_thw=(out_frames, h, w), cond=satellite_cond)
            rain = self.rain_decoder(feat, target_thw=(out_frames, h, w), cond=rain_cond)

        if self.decoder_output_mode == "residual" and context_anchor is not None:
            out_frames = radar.shape[2]
            radar = radar + context_anchor["radar"].expand(-1, -1, out_frames, -1, -1)
            satellite = satellite + context_anchor["satellite"].expand(-1, -1, out_frames, -1, -1)
            rain = rain + context_anchor["rain"].expand(-1, -1, out_frames, -1, -1)

        if return_modality_dict:
            return {
                "radar": radar,
                "satellite": satellite,
                "rain": rain,
            }
        return rain

    def forward(
        self,
        x: torch.Tensor,
        predict_frames: int = 1,
        strict_target_isolation: bool = False,
        return_modality_dict: bool = True,
        frame_times: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected x to be 5D [B,C,T,H,W], got shape={tuple(x.shape)}")

        b, c, t, h, w = x.shape
        if c != self.in_channels:
            raise ValueError(f"in_channels mismatch: model expects {self.in_channels}, got {c}")
        if t > self.max_frames:
            raise ValueError(f"Input frames {t} exceed max_frames {self.max_frames}")
        if predict_frames <= 0 or predict_frames > t:
            raise ValueError(f"predict_frames should be in [1, {t}], got {predict_frames}")
        if t % self.frame_patch_size != 0:
            raise ValueError(f"Input T must be divisible by frame_patch_size={self.frame_patch_size}, got T={t}")
        if predict_frames % self.frame_patch_size != 0:
            raise ValueError(
                "predict_frames must be divisible by "
                f"frame_patch_size={self.frame_patch_size}, got predict_frames={predict_frames}"
            )

        tokens, hp, wp, _, _, temporal_positions = self._encode_tokens(x=x, frame_offset=0, frame_times=frame_times)
        predict_token_frames = predict_frames // self.frame_patch_size
        total_token_frames = int(tokens.shape[1])
        if predict_token_frames <= 0 or predict_token_frames > total_token_frames:
            raise ValueError(f"predict token frames should be in [1, {total_token_frames}], got {predict_token_frames}")

        context_frames = total_token_frames - predict_token_frames
        context_frames_raw = t - predict_frames
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

        tokens = tokens[:, -predict_token_frames:, :, :]
        context_anchor = self._last_context_anchor(x=x, context_frames=context_frames_raw)
        if self.cross_modal_adapter is not None:
            tokens = self.cross_modal_adapter(tokens, context_x=x[:, :, :context_frames_raw])
        if self.local_window_refiner is not None:
            tokens = self.local_window_refiner(tokens, height=hp, width=wp)
        film_condition = self._build_film_condition_sequence(
            x=x,
            context_frames=context_frames_raw,
            predict_frames=predict_frames,
        )
        return self._decode_target_tokens(
            target_tokens=tokens,
            hp=hp,
            wp=wp,
            h=h,
            w=w,
            batch_size=b,
            predict_frames=predict_token_frames,
            context_anchor=context_anchor,
            film_condition=film_condition,
            return_modality_dict=return_modality_dict,
        )

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
        if target_x.ndim != 5:
            raise ValueError(f"target_x must be [B,C,T,H,W], got {tuple(target_x.shape)}")

        b, c, tt, h, w = target_x.shape
        if tt <= 0:
            raise ValueError("target_x must contain at least 1 frame.")

        frame_times: torch.Tensor | None = None
        if context_x is None:
            x = target_x
            frame_times = target_time
        else:
            if context_x.ndim != 5:
                raise ValueError(f"context_x must be [B,C,T,H,W], got {tuple(context_x.shape)}")
            bc, cc, _tc, hc, wc = context_x.shape
            if bc != b or cc != c or hc != h or wc != w:
                raise ValueError(
                    "context_x and target_x shape mismatch: "
                    f"context={tuple(context_x.shape)}, target={tuple(target_x.shape)}"
                )
            masked_context_x = self._apply_context_modality_mask_token(
                context_x=context_x,
                context_modality_available=context_modality_available,
            )
            x = torch.cat([masked_context_x, target_x], dim=2)
            if context_time is not None or target_time is not None:
                if context_time is None or target_time is None:
                    raise ValueError("context_time and target_time should be both set or both None.")
                if context_time.ndim != 2 or target_time.ndim != 2:
                    raise ValueError(
                        "context_time and target_time must be [B,T], "
                        f"got context={tuple(context_time.shape)}, target={tuple(target_time.shape)}"
                    )
                if int(context_time.shape[0]) != b or int(target_time.shape[0]) != b:
                    raise ValueError(
                        "batch size mismatch between x and time tensors: "
                        f"x={b}, context_time={int(context_time.shape[0])}, target_time={int(target_time.shape[0])}"
                    )
                if int(context_time.shape[1]) != int(context_x.shape[2]) or int(target_time.shape[1]) != int(target_x.shape[2]):
                    raise ValueError(
                        "time length mismatch with frames: "
                        f"context_time={int(context_time.shape[1])}, context_frames={int(context_x.shape[2])}, "
                        f"target_time={int(target_time.shape[1])}, target_frames={int(target_x.shape[2])}"
                    )
                frame_times = torch.cat([context_time, target_time], dim=1)

        target_frames = tt if predict_frames is None else int(predict_frames)
        if target_frames <= 0 or target_frames > tt:
            raise ValueError(f"predict_frames should be in [1, {tt}], got {target_frames}")

        return self.forward(
            x=x,
            predict_frames=target_frames,
            strict_target_isolation=strict_target_isolation,
            return_modality_dict=return_modality_dict,
            frame_times=frame_times,
        )

    def forward_modalities(
        self,
        radar: torch.Tensor,
        satellite: torch.Tensor,
        rain: torch.Tensor,
        predict_frames: int = 1,
        strict_target_isolation: bool = False,
        return_modality_dict: bool = True,
    ) -> dict[str, torch.Tensor] | torch.Tensor:
        x = torch.cat([radar, satellite, rain], dim=1)
        return self.forward(
            x=x,
            predict_frames=predict_frames,
            strict_target_isolation=strict_target_isolation,
            return_modality_dict=return_modality_dict,
        )


if __name__ == "__main__":
    model = RainCausalPatchTransformerNextFrame(
        in_channels=12,
        out_channels=1,
        radar_out_channels=1,
        satellite_out_channels=10,
        rain_out_channels=1,
        input_size=64,
        patch_size=4,
        dim=128,
        depth=2,
        num_heads=4,
        max_frames=16,
    )
    x = torch.randn(2, 12, 6, 64, 64)
    with torch.no_grad():
        out = model(x=x, predict_frames=2, return_modality_dict=True)
    print("radar:", out["radar"].shape)
    print("satellite:", out["satellite"].shape)
    print("rain:", out["rain"].shape)
