import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

from src.networks.modules.resblock import ResnetBlock


def _valid_gn_groups(channels: int, max_groups: int = 32) -> int:
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


def validate_padding_mode(pad_mode: str) -> str:
    valid_modes = {"zeros", "reflect", "replicate", "circular"}
    normalized = str(pad_mode).lower()
    if normalized not in valid_modes:
        raise ValueError(f"padding mode must be one of {sorted(valid_modes)}, got {pad_mode}")
    return normalized


def _resolve_upsample_steps(scale: int, name: str) -> int:
    if scale <= 0:
        raise ValueError(f"{name} must be > 0, got {scale}")
    if scale == 1:
        return 0
    steps = int(math.log2(scale))
    if 2**steps != scale:
        raise ValueError(f"{name} must be power-of-two, got {scale}")
    return steps


def _validate_odd_kernel_size(k_size: int, name: str) -> int:
    kernel_size = int(k_size)
    if kernel_size <= 0:
        raise ValueError(f"{name} must be > 0, got {kernel_size}")
    if kernel_size % 2 == 0:
        raise ValueError(f"{name} must be odd for symmetric padding, got {kernel_size}")
    return kernel_size


class ConvStem(nn.Module):
    def __init__(self, in_channels: int, stem_channels: int, pad_mode: str = "zeros") -> None:
        super().__init__()
        pad_mode = validate_padding_mode(pad_mode)
        hidden = max(stem_channels // 2, 16)
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                hidden,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode=pad_mode,
            ),
            nn.GroupNorm(num_groups=_valid_gn_groups(hidden), num_channels=hidden),
            nn.SiLU(),
            nn.Conv2d(
                hidden,
                stem_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode=pad_mode,
            ),
            nn.GroupNorm(num_groups=_valid_gn_groups(stem_channels), num_channels=stem_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResNetBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.block = ResnetBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            norm_type="gn",
            act_type=("silu", "gelu"),
            padding_mode=validate_padding_mode(padding_mode),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SpatialPadConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding_mode: str = "zeros",
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.padding_mode = validate_padding_mode(padding_mode)
        self.causal = bool(causal)
        if isinstance(kernel_size, int):
            kernel = (kernel_size, kernel_size, kernel_size)
        else:
            kernel = kernel_size
        kt, kh, kw = kernel
        temporal_pad_left = kt - 1 if self.causal else 0
        self.full_pad = (kw // 2, kw // 2, kh // 2, kh // 2, temporal_pad_left, 0)
        temporal_conv_padding = 0 if self.causal else kt // 2
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=(temporal_conv_padding, 0, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if any(self.full_pad):
            pad_mode = self.padding_mode
            if pad_mode == "reflect":
                pad_t = max(self.full_pad[4], self.full_pad[5])
                pad_h = max(self.full_pad[2], self.full_pad[3])
                pad_w = max(self.full_pad[0], self.full_pad[1])
                if pad_t >= int(x.shape[-3]) or pad_h >= int(x.shape[-2]) or pad_w >= int(x.shape[-1]):
                    # reflect padding requires pad < input size per dimension
                    pad_mode = "replicate"
            if pad_mode == "zeros":
                x = F.pad(x, self.full_pad)
            else:
                x = F.pad(x, self.full_pad, mode=pad_mode)
        return self.conv(x)


class WanFactorizedCausalConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding_mode: str = "zeros",
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.padding_mode = validate_padding_mode(padding_mode)
        self.causal = bool(causal)

        if isinstance(kernel_size, int):
            kt, kh, kw = (kernel_size, kernel_size, kernel_size)
        else:
            kt, kh, kw = kernel_size
        if isinstance(stride, int):
            st, sh, sw = (stride, stride, stride)
        else:
            st, sh, sw = stride

        self.spatial_pad = (kw // 2, kw // 2, kh // 2, kh // 2)
        self.spatial_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kh, kw),
            stride=(sh, sw),
            padding=0,
        )
        self.temporal_conv = SpatialPadConv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(kt, 1, 1),
            stride=(st, 1, 1),
            padding_mode=padding_mode,
            causal=causal,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, t, _, _ = x.shape
        x = rearrange(x, "b c t h w -> (b t) c h w")
        if any(self.spatial_pad):
            pad_mode = self.padding_mode
            if pad_mode == "reflect":
                pad_h = max(self.spatial_pad[2], self.spatial_pad[3])
                pad_w = max(self.spatial_pad[0], self.spatial_pad[1])
                if pad_h >= int(x.shape[-2]) or pad_w >= int(x.shape[-1]):
                    pad_mode = "replicate"
            if pad_mode == "zeros":
                x = F.pad(x, self.spatial_pad)
            else:
                x = F.pad(x, self.spatial_pad, mode=pad_mode)
        x = self.spatial_conv(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", b=b, t=t)
        return self.temporal_conv(x)


def build_3d_conv_layer(
    *,
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int, int],
    stride: int | tuple[int, int, int] = 1,
    padding_mode: str = "zeros",
    causal: bool = False,
    conv_style: str = "full3d",
) -> nn.Module:
    style = str(conv_style).lower()
    if style == "full3d":
        return SpatialPadConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding_mode=padding_mode,
            causal=causal,
        )
    if style == "wan_factorized":
        return WanFactorizedCausalConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding_mode=padding_mode,
            causal=causal,
        )
    raise ValueError(f"conv_style must be 'full3d' or 'wan_factorized', got {conv_style}")


class ResNetBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k_size: int = 3,
        dropout: float = 0.0,
        padding_mode: str = "zeros",
        causal: bool = False,
        conv_style: str = "full3d",
    ) -> None:
        super().__init__()
        padding_mode = validate_padding_mode(padding_mode)
        k_size = _validate_odd_kernel_size(k_size, "ResNetBlock3D.k_size")
        self.norm1 = nn.GroupNorm(num_groups=_valid_gn_groups(in_channels), num_channels=in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = build_3d_conv_layer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=k_size,
            padding_mode=padding_mode,
            causal=causal,
            conv_style=conv_style,
        )
        self.norm2 = nn.GroupNorm(num_groups=_valid_gn_groups(out_channels), num_channels=out_channels)
        self.act2 = nn.SiLU()
        self.dropout = nn.Dropout3d(p=dropout) if dropout > 0 else nn.Identity()
        self.conv2 = build_3d_conv_layer(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=k_size,
            padding_mode=padding_mode,
            causal=causal,
            conv_style=conv_style,
        )
        self.skip = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(self.act1(self.norm1(x)))
        x = self.conv2(self.dropout(self.act2(self.norm2(x))))
        return x + residual


class FiLMCondition2D(nn.Module):
    def __init__(self, feat_channels: int, cond_channels: int) -> None:
        super().__init__()
        self.to_scale_shift = nn.Conv2d(cond_channels, feat_channels * 2, kernel_size=1)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if cond.ndim != 4:
            raise ValueError(f"2D FiLM cond must be [B,C,H,W], got {tuple(cond.shape)}")
        cond = F.interpolate(cond, size=x.shape[-2:], mode="bilinear", align_corners=False)
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=1)
        return x * (1.0 + scale) + shift


class FiLMCondition3D(nn.Module):
    def __init__(self, feat_channels: int, cond_channels: int) -> None:
        super().__init__()
        self.to_scale_shift = nn.Conv3d(cond_channels, feat_channels * 2, kernel_size=1)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if cond.ndim != 5:
            raise ValueError(f"3D FiLM cond must be [B,C,T,H,W], got {tuple(cond.shape)}")
        cond = F.interpolate(cond, size=x.shape[-3:], mode="trilinear", align_corners=False)
        scale, shift = self.to_scale_shift(cond).chunk(2, dim=1)
        return x * (1.0 + scale) + shift


class UpsampleConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int,
        upsample_mode: str,
        padding_mode: str,
        k_size: int = 3,
    ) -> None:
        super().__init__()
        self.upsample_mode = upsample_mode
        kernel_size = _validate_odd_kernel_size(k_size, "UpsampleConv2d.k_size")
        padding = kernel_size // 2

        if upsample_mode == "nearest_conv":
            self.resize = nn.Upsample(scale_factor=scale_factor, mode="nearest")
            self.proj = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode=padding_mode,
            )
            return

        if upsample_mode == "linear_conv":
            self.resize = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=False)
            self.proj = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                padding_mode=padding_mode,
            )
            return

        if upsample_mode == "pixelshuffle":
            if scale_factor != 2:
                raise ValueError(f"pixelshuffle requires scale_factor=2, got {scale_factor}")
            self.resize = nn.Identity()
            self.proj = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * (scale_factor**2),
                    kernel_size=kernel_size,
                    padding=padding,
                    padding_mode=padding_mode,
                ),
                nn.PixelShuffle(scale_factor),
            )
            return

        if upsample_mode == "conv_transpose":
            if scale_factor != 2:
                raise ValueError(f"conv_transpose requires scale_factor=2, got {scale_factor}")
            self.resize = nn.Identity()
            self.proj = nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
            )
            return

        raise ValueError(f"Unsupported decoder upsample mode: {upsample_mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.resize(x))


class UpsampleConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_t: int,
        scale_hw: int,
        upsample_mode: str,
        padding_mode: str,
        causal: bool = False,
        conv_style: str = "full3d",
        k_size: int = 3,
    ) -> None:
        super().__init__()
        self.upsample_mode = upsample_mode
        kernel_size = _validate_odd_kernel_size(k_size, "UpsampleConv3d.k_size")

        if upsample_mode == "nearest_conv":
            self.resize = nn.Upsample(scale_factor=(scale_t, scale_hw, scale_hw), mode="nearest")
            self.proj = build_3d_conv_layer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding_mode=padding_mode,
                causal=causal,
                conv_style=conv_style,
            )
            self.shuffle = None
            return

        if upsample_mode == "linear_conv":
            self.resize = nn.Upsample(scale_factor=(scale_t, scale_hw, scale_hw), mode="trilinear", align_corners=False)
            self.proj = build_3d_conv_layer(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding_mode=padding_mode,
                causal=causal,
                conv_style=conv_style,
            )
            self.shuffle = None
            return

        if upsample_mode == "pixelshuffle":
            self.resize = nn.Identity() if scale_t == 1 else nn.Upsample(scale_factor=(scale_t, 1, 1), mode="nearest")
            proj_channels = out_channels if scale_hw == 1 else out_channels * (scale_hw**2)
            self.proj = build_3d_conv_layer(
                in_channels=in_channels,
                out_channels=proj_channels,
                kernel_size=kernel_size,
                padding_mode=padding_mode,
                causal=causal,
                conv_style=conv_style,
            )
            if scale_hw == 1:
                self.shuffle = None
                return
            if scale_hw != 2:
                raise ValueError(f"3D pixelshuffle requires scale_hw in {{1, 2}}, got {scale_hw}")
            self.shuffle = nn.PixelShuffle(scale_hw)
            return

        if upsample_mode == "conv_transpose":
            kernel_t, stride_t, pad_t = _conv_transpose_step_spec(scale_t)
            kernel_hw, stride_hw, pad_hw = _conv_transpose_step_spec(scale_hw)
            self.resize = nn.Identity()
            self.proj = nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=(kernel_t, kernel_hw, kernel_hw),
                stride=(stride_t, stride_hw, stride_hw),
                padding=(pad_t, pad_hw, pad_hw),
            )
            self.shuffle = None
            return

        raise ValueError(f"Unsupported decoder upsample mode: {upsample_mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resize(x)
        x = self.proj(x)
        if self.upsample_mode != "pixelshuffle" or self.shuffle is None:
            return x

        b, c, t, h, w = x.shape
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.shuffle(x)
        return rearrange(x, "(b t) c h w -> b c t h w", b=b, t=t)


def _conv_transpose_step_spec(scale: int) -> tuple[int, int, int]:
    if scale == 1:
        return 1, 1, 0
    if scale == 2:
        return 4, 2, 1
    raise ValueError(f"conv_transpose only supports scale 1 or 2, got {scale}")


def build_decoder_upsample_layer(
    *,
    spatial_dims: Literal[2, 3],
    in_channels: int,
    out_channels: int,
    scale_hw: int,
    upsample_mode: str,
    padding_mode: str,
    scale_t: int = 1,
    causal: bool = False,
    conv_style: str = "full3d",
    k_size: int = 3,
) -> nn.Module:
    upsample_mode = str(upsample_mode).lower()
    padding_mode = validate_padding_mode(padding_mode)
    if spatial_dims == 2:
        if scale_t != 1:
            raise ValueError(f"2D upsample layer does not support temporal scaling, got scale_t={scale_t}")
        return UpsampleConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            scale_factor=scale_hw,
            upsample_mode=upsample_mode,
            padding_mode=padding_mode,
            k_size=k_size,
        )
    if spatial_dims == 3:
        return UpsampleConv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            scale_t=scale_t,
            scale_hw=scale_hw,
            upsample_mode=upsample_mode,
            padding_mode=padding_mode,
            causal=causal,
            conv_style=conv_style,
            k_size=k_size,
        )
    raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")


class ResNetDecoder2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        base_channels: int = 256,
        dropout: float = 0.0,
        activation_checkpoint: bool = False,
        padding_mode: str = "zeros",
        upsample_mode: str = "nearest_conv",
        cond_channels: int | None = None,
        use_film: bool = False,
        k_size: int = 3,
    ) -> None:
        super().__init__()
        padding_mode = validate_padding_mode(padding_mode)
        k_size = _validate_odd_kernel_size(k_size, "ResNetDecoder2D.k_size")
        conv_padding = k_size // 2
        self.use_film = bool(use_film)
        if self.use_film and (cond_channels is None or cond_channels <= 0):
            raise ValueError(f"use_film=True requires positive cond_channels, got {cond_channels}")

        self.input_proj = nn.Sequential(
            nn.Conv2d(
                in_channels,
                base_channels,
                kernel_size=k_size,
                padding=conv_padding,
                padding_mode=padding_mode,
            ),
            nn.GroupNorm(num_groups=_valid_gn_groups(base_channels), num_channels=base_channels),
            nn.SiLU(),
            ResNetBlock2D(
                in_channels=base_channels,
                out_channels=base_channels,
                dropout=dropout,
                padding_mode=padding_mode,
            ),
        )
        self.input_film = FiLMCondition2D(base_channels, int(cond_channels)) if self.use_film else None

        up_layers: list[nn.Module] = []
        up_films: list[nn.Module] = []
        c = base_channels
        n_upsample = int(math.log2(patch_size)) if patch_size > 1 else 0
        for _ in range(n_upsample):
            c_next = max(c // 2, 64)
            up_layer = nn.Sequential(
                build_decoder_upsample_layer(
                    spatial_dims=2,
                    in_channels=c,
                    out_channels=c_next,
                    scale_hw=2,
                    upsample_mode=upsample_mode,
                    padding_mode=padding_mode,
                    k_size=k_size,
                ),
                nn.GroupNorm(num_groups=_valid_gn_groups(c_next), num_channels=c_next),
                nn.SiLU(),
                ResNetBlock2D(
                    in_channels=c_next,
                    out_channels=c_next,
                    dropout=dropout,
                    padding_mode=padding_mode,
                ),
            )
            if activation_checkpoint:
                up_layer = CheckpointWrapper(up_layer)
            up_layers.append(up_layer)
            if self.use_film:
                up_films.append(FiLMCondition2D(c_next, int(cond_channels)))
            c = c_next
        self.up_layers = nn.ModuleList(up_layers)
        self.up_films = nn.ModuleList(up_films)
        self.out_head = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, target_hw: tuple[int, int], cond: torch.Tensor | None = None) -> torch.Tensor:
        x = self.input_proj(x)
        if self.use_film:
            if cond is None:
                raise ValueError("ResNetDecoder2D requires cond when use_film=True.")
            x = self.input_film(x, cond)

        for idx, layer in enumerate(self.up_layers):
            x = layer(x)
            if self.use_film:
                x = self.up_films[idx](x, cond)
        if x.shape[-2:] != target_hw:
            x = F.interpolate(x, size=target_hw, mode="bilinear", align_corners=False)
        return self.out_head(x)


class ResNetDecoder3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        patch_size: int,
        frame_patch_size: int = 1,
        base_channels: int = 256,
        dropout: float = 0.0,
        activation_checkpoint: bool = False,
        padding_mode: str = "zeros",
        upsample_mode: str = "nearest_conv",
        cond_channels: int | None = None,
        use_film: bool = False,
        causal: bool = False,
        conv_style: str = "full3d",
        k_size: int = 7,
    ) -> None:
        super().__init__()
        padding_mode = validate_padding_mode(padding_mode)
        k_size = _validate_odd_kernel_size(k_size, "ResNetDecoder3D.k_size")
        self.use_film = bool(use_film)
        if self.use_film and (cond_channels is None or cond_channels <= 0):
            raise ValueError(f"use_film=True requires positive cond_channels, got {cond_channels}")

        self.input_proj = nn.Sequential(
            build_3d_conv_layer(
                in_channels=in_channels,
                out_channels=base_channels,
                kernel_size=k_size,
                padding_mode=padding_mode,
                causal=causal,
                conv_style=conv_style,
            ),
            nn.GroupNorm(num_groups=_valid_gn_groups(base_channels), num_channels=base_channels),
            nn.SiLU(),
            ResNetBlock3D(
                in_channels=base_channels,
                out_channels=base_channels,
                k_size=k_size,
                dropout=dropout,
                padding_mode=padding_mode,
                causal=causal,
                conv_style=conv_style,
            ),
        )
        self.input_film = FiLMCondition3D(base_channels, int(cond_channels)) if self.use_film else None

        spatial_steps = _resolve_upsample_steps(patch_size, "patch_size")
        temporal_steps = _resolve_upsample_steps(frame_patch_size, "frame_patch_size")
        n_upsample = max(spatial_steps, temporal_steps)

        up_layers: list[nn.Module] = []
        up_films: list[nn.Module] = []
        c = base_channels
        for i in range(n_upsample):
            scale_t = 2 if i < temporal_steps else 1
            scale_hw = 2 if i < spatial_steps else 1
            c_next = max(c // 2, 64)
            up_layer = nn.Sequential(
                build_decoder_upsample_layer(
                    spatial_dims=3,
                    in_channels=c,
                    out_channels=c_next,
                    scale_t=scale_t,
                    scale_hw=scale_hw,
                    upsample_mode=upsample_mode,
                    padding_mode=padding_mode,
                    causal=causal,
                    conv_style=conv_style,
                    k_size=k_size,
                ),
                nn.GroupNorm(num_groups=_valid_gn_groups(c_next), num_channels=c_next),
                nn.SiLU(),
                ResNetBlock3D(
                    in_channels=c_next,
                    out_channels=c_next,
                    k_size=k_size,
                    dropout=dropout,
                    padding_mode=padding_mode,
                    causal=causal,
                    conv_style=conv_style,
                ),
            )
            if activation_checkpoint:
                up_layer = CheckpointWrapper(up_layer)
            up_layers.append(up_layer)
            if self.use_film:
                up_films.append(FiLMCondition3D(c_next, int(cond_channels)))
            c = c_next
        self.up_layers = nn.ModuleList(up_layers)
        self.up_films = nn.ModuleList(up_films)
        self.out_head = nn.Conv3d(c, out_channels, kernel_size=1)

    def forward(
        self, x: torch.Tensor, target_thw: tuple[int, int, int], cond: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.input_proj(x)
        if self.use_film:
            if cond is None:
                raise ValueError("ResNetDecoder3D requires cond when use_film=True.")
            x = self.input_film(x, cond)

        for idx, layer in enumerate(self.up_layers):
            x = layer(x)
            if self.use_film:
                x = self.up_films[idx](x, cond)
        if x.shape[-3:] != target_thw:
            x = F.interpolate(x, size=target_thw, mode="trilinear", align_corners=False)
        return self.out_head(x)
