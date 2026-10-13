import torch
import torch.nn as nn
from timm.layers import create_norm_layer


def _valid_gn_groups(channels: int, max_groups: int = 32) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _build_norm_layer(
    channels: int,
    *,
    norm_layer: str,
    norm_max_groups: int,
) -> nn.Module:
    normalized_name = norm_layer.strip().lower()
    if normalized_name == "":
        return nn.Identity()
    if normalized_name == "groupnorm":
        groups = _valid_gn_groups(channels=channels, max_groups=norm_max_groups)
        return create_norm_layer("groupnorm", channels, num_groups=groups)
    return create_norm_layer(norm_layer, channels)


class _ConvNormAct3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: tuple[int, int, int],
        stride: tuple[int, int, int],
        padding: tuple[int, int, int],
        norm_layer: str,
        norm_max_groups: int,
        act_negative_slope: float,
    ) -> None:
        super().__init__()
        use_bias = norm_layer.strip() == ""
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=use_bias,
        )
        self.norm = _build_norm_layer(
            channels=out_channels,
            norm_layer=norm_layer,
            norm_max_groups=norm_max_groups,
        )
        self.act = nn.LeakyReLU(negative_slope=act_negative_slope, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class PatchGAN3D(nn.Module):
    def __init__(
        self,
        context_channels: int,
        target_channels: int,
        base_channels: int = 64,
        channel_multipliers: tuple[int, ...] = (1, 2, 4, 8),
        norm_layer: str = "groupnorm",
        norm_max_groups: int = 32,
        act_negative_slope: float = 0.2,
    ) -> None:
        super().__init__()
        if context_channels <= 0:
            raise ValueError(f"context_channels must be > 0, got {context_channels}")
        if target_channels <= 0:
            raise ValueError(f"target_channels must be > 0, got {target_channels}")
        if base_channels <= 0:
            raise ValueError(f"base_channels must be > 0, got {base_channels}")
        if len(channel_multipliers) <= 0:
            raise ValueError("channel_multipliers must be non-empty.")
        if any(multiplier <= 0 for multiplier in channel_multipliers):
            raise ValueError(f"channel_multipliers must all be > 0, got {channel_multipliers}")
        if norm_max_groups <= 0:
            raise ValueError(f"norm_max_groups must be > 0, got {norm_max_groups}")
        if act_negative_slope <= 0:
            raise ValueError(f"act_negative_slope must be > 0, got {act_negative_slope}")

        self.context_channels = int(context_channels)
        self.target_channels = int(target_channels)
        self.input_channels = self.context_channels + self.target_channels

        blocks: list[nn.Module] = []
        in_channels = self.input_channels
        for idx, multiplier in enumerate(channel_multipliers):
            out_channels = int(base_channels * multiplier)
            stride = (1, 2, 2) if idx < 2 else (2, 2, 2)
            blocks.append(
                _ConvNormAct3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=(3, 4, 4),
                    stride=stride,
                    padding=(1, 1, 1),
                    norm_layer=norm_layer,
                    norm_max_groups=norm_max_groups,
                    act_negative_slope=act_negative_slope,
                )
            )
            in_channels = out_channels

        blocks.append(
            _ConvNormAct3D(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=(3, 3, 3),
                stride=(1, 1, 1),
                padding=(1, 1, 1),
                norm_layer=norm_layer,
                norm_max_groups=norm_max_groups,
                act_negative_slope=act_negative_slope,
            )
        )
        self.features = nn.Sequential(*blocks)
        self.head = nn.Conv3d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
            bias=True,
        )

    @staticmethod
    def _check_bcthw(x: torch.Tensor, name: str) -> None:
        if x.ndim != 5:
            raise ValueError(f"{name} must be [B,C,T,H,W], got shape={tuple(x.shape)}")

    def forward(self, context: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        self._check_bcthw(context, "context")
        self._check_bcthw(target, "target")
        if int(context.shape[0]) != int(target.shape[0]):
            raise ValueError(f"context/target batch mismatch: {tuple(context.shape)} vs {tuple(target.shape)}")
        if tuple(context.shape[2:]) != tuple(target.shape[2:]):
            raise ValueError(
                "context/target shape mismatch on [T,H,W]: "
                f"context={tuple(context.shape)}, target={tuple(target.shape)}"
            )
        if int(context.shape[1]) != self.context_channels:
            raise ValueError(
                f"context channels mismatch: expected {self.context_channels}, got {int(context.shape[1])}"
            )
        if int(target.shape[1]) != self.target_channels:
            raise ValueError(f"target channels mismatch: expected {self.target_channels}, got {int(target.shape[1])}")

        x = torch.cat([context, target], dim=1)
        features = self.features(x)
        logits = self.head(features)
        return logits
