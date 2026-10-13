
from typing import Literal

import torch
import torch.nn as nn
from einops import rearrange
from timm.layers import DropPath
from torch.utils.checkpoint import checkpoint

from src.networks.modules.norms import Normalize

# * --- Convnext blocks --- #


class ConvNeXtBlock(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim,
        out_dim,
        norm_type="gn",
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        act_checkpoint=False,
        padding_mode="zeros",
        num_groups=32,
    ):
        super().__init__()

        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=7, padding=3, groups=dim, padding_mode=padding_mode
        )
        self.norm = Normalize(dim, norm_type=norm_type, num_groups=num_groups)

        self.pwconv1 = nn.Conv2d(dim, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden_dim, out_dim, kernel_size=1)

        # Layer Scaling
        self.gamma = (
            nn.Parameter(
                layer_scale_init_value * torch.ones((out_dim)), requires_grad=True
            )
            if layer_scale_init_value > 0
            else None
        )
        if dim != out_dim:
            self.nin_shortcut = nn.Conv2d(dim, out_dim, kernel_size=1, stride=1)
        else:
            self.nin_shortcut = nn.Identity()

        # Stochastic Depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.act_checkpoint = act_checkpoint

    def forward_fn(self, x):
        input = x

        x = self.dwconv(x)
        x = self.norm(x)

        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        # Layer Scaling
        if self.gamma is not None:
            x = self.gamma.view(-1, 1, 1) * x

        x = self.nin_shortcut(input) + self.drop_path(x)

        return x

    def forward(self, x):
        if self.act_checkpoint and self.training:
            return checkpoint(self.forward_fn, x, use_reentrant=True)  # type: ignore
        return self.forward_fn(x)
