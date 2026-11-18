from typing import Literal, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.activations import ACT2FN

from src.networks.modules.norms import Normalize


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int | None = None,
        dropout: float,
        use_residual_factor: bool = False,
        act_type: tuple = ("gelu", "gelu"),
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        padding_mode = kwargs.get("padding_mode", "zeros")
        norm_type = kwargs.get("norm_type", "gn")
        gn_norm_groups = kwargs.get("num_groups", 32)

        self.norm1 = Normalize(
            in_channels, num_groups=gn_norm_groups, norm_type=norm_type
        )
        self.act1 = ACT2FN[act_type[0]]
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode=padding_mode,
        )
        self.norm2 = Normalize(
            out_channels, num_groups=gn_norm_groups, norm_type=norm_type
        )
        self.act2 = ACT2FN[act_type[1]]
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode=padding_mode,
        )
        self.nin_shortcut = (
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
            )
            if in_channels != out_channels
            else nn.Identity()
        )
        self.act_checkpoint = kwargs.get("act_checkpoint", False)
        self.use_residual_factor = use_residual_factor
        if use_residual_factor:
            self.residual_factor = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward_fn(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        h = self.norm1(h)
        h = self.act1(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.conv2(h)

        x = self.nin_shortcut(x)
        if self.use_residual_factor:
            h = h * self.residual_factor

        return x + h

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if self.act_checkpoint and self.training:
            return checkpoint(self.forward_fn, x, use_reentrant=True)
        return self.forward_fn(x)
