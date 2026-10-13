from functools import partial
from inspect import signature
from typing import Literal, Optional, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.networks.modules.norms import Normalize
from src.utils.logging import log_print

# * --- Activation Functions --- #


def val2tuple(x: list | tuple | Any, min_len: int = 1) -> tuple:
    if isinstance(x, (list, tuple)):
        x = list(x)
    else:
        x = [x] * min_len

    return tuple(x)


# register activation function here
REGISTERED_ACT_DICT: dict[str, type | partial] = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "hswish": nn.Hardswish,
    "silu": nn.SiLU,
    "gelu": partial(nn.GELU, approximate="tanh"),
}


def build_act(name: str, **kwargs) -> Optional[nn.Module]:
    if name in REGISTERED_ACT_DICT:
        act_cls = REGISTERED_ACT_DICT[name]
        kwargs = extract_needed_kwargs(kwargs, act_cls)
        return act_cls(**kwargs)
    else:
        return None


# * --- Convolutional Layers --- #


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        use_bias: bool = False,
        dropout: float = 0,
        norm: str = "bn2d",
        act_func: str = "relu",
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()

        padding = get_same_padding(kernel_size)
        padding *= dilation

        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
            padding_mode=padding_mode,
        )
        self.norm = Normalize(in_channels=out_channels, norm_type=norm)
        self.act = build_act(act_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x



# * --- Residaul block builder --- #


class ResidualBlock(nn.Module):
    def __init__(
        self,
        main: Optional[nn.Module],
        shortcut: Optional[nn.Module],
        post_act=None,
        pre_norm: Optional[nn.Module] = None,
    ):
        super().__init__()

        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = build_act(post_act)

    def forward_main(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm is None:
            return self.main(x)
        else:
            return self.main(self.pre_norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x)
        else:
            res = self.forward_main(x) + self.shortcut(x)
            if self.post_act:
                res = self.post_act(res)
        return res


# TODO: this version causes the norm is high
class GLUMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        use_bias=False,
        norm=(None, None, "ln2d"),
        act_func=("silu", "silu", None),
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act_func = val2tuple(act_func, 3)

        mid_channels = (
            round(in_channels * expand_ratio) if mid_channels is None else mid_channels
        )

        self.glu_act = build_act(act_func[1], inplace=False)
        self.inverted_conv = ConvLayer(
            in_channels,
            mid_channels * 2,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.depth_conv = ConvLayer(
            mid_channels * 2,
            mid_channels * 2,
            kernel_size,
            stride=stride,
            groups=mid_channels * 2,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=None,
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            use_bias=use_bias[2],
            norm=norm[2],
            act_func=act_func[2],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)

        x, gate = torch.chunk(x, 2, dim=1)
        gate = self.glu_act(gate)
        x = x * gate

        x = self.point_conv(x)
        return x



# * --- Upsample and Downsample --- #

def get_same_padding(
    kernel_size: Union[int, tuple[int, ...]],
):
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2


def resample_norm_keep(x, x_resampled):
    return x_resampled * torch.norm(x) / torch.norm(x_resampled)


class UpsampleRepeatConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        padding_mode: str = "zeros",
        norm_type: str | None = None,
        norm_keep: bool = False,
    ):
        super().__init__()
        self.norm_keep = norm_keep
        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode=padding_mode,
        )
        # self.norm = Normalize(
        #     in_channels=in_channels, num_groups=32, norm_type=norm_type
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
        x_resp = self.conv(x)
        if self.norm_keep:
            x_resp = resample_norm_keep(x, x_resp)
        return x_resp


class DownsamplePadConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        padding_mode: str = "constant",
        padding_in_conv: bool = False,
        norm_type: Optional[str] = None,
        norm_keep: bool = False,
    ):
        # Zihan NOTE: using pad (left and right) align the center of the pixel when downsampling
        # but (may?) cause the boundary artifact when upsampling

        super().__init__()
        self.padding_mode = padding_mode
        self.padding_in_conv = padding_in_conv
        self.norm_keep = norm_keep

        self.conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=2,
            padding=0 if not padding_in_conv else 1,
            padding_mode=padding_mode
            if padding_in_conv
            else "zeros",  # 'zeros' as default
        )

        # self.norm = Normalize(
        #     in_channels=in_channels, num_groups=32, norm_type=norm_type
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LDM VAE also use the unsymmetric padding
        if not self.padding_in_conv:  # cosmos manually pad
            # to align on the center of the downsampled image pixels
            pad = (0, 1, 0, 1)  # lower and righter pads, why? inductive bias?
            if self.padding_mode not in ("constant", "zeros"):
                x = F.pad(x, pad, mode=self.padding_mode)
            else:
                x = F.pad(x, pad, mode="constant", value=0)

        x_resp = self.conv(x)
        if self.norm_keep:
            x_resp = resample_norm_keep(x, x_resp)
        return x_resp


class ConvPixelUnshuffleDownSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
        padding_mode: str = "zeros",
        norm_keep: bool = False,
    ):
        super().__init__()
        self.factor = factor
        self.norm_keep = norm_keep
        out_ratio = factor**2
        assert out_channels % out_ratio == 0
        self.conv = nn.Conv2d(
            in_channels,
            out_channels // out_ratio,
            kernel_size,
            padding=get_same_padding(kernel_size),
            padding_mode=padding_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = F.pixel_unshuffle(x, self.factor)
        if self.norm_keep:
            x = resample_norm_keep(x, x)
        return x


class PixelUnshuffleChannelAveragingDownSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
        # group_size: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        assert in_channels * factor**2 % out_channels == 0
        self.group_size = in_channels * factor**2 // out_channels
        # hidden = out_channels * group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pixel_unshuffle(
            x, self.factor
        )  # c * factor ** 2 -> hidden = out_c * group_size
        B, C, H, W = x.shape
        x = x.view(B, self.out_channels, self.group_size, H, W)
        x = x.mean(dim=2)
        return x


class ConvPixelShuffleUpSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
        padding_mode: str = "zeros",
        norm_type: Optional[str] = None,
    ):
        super().__init__()
        self.factor = factor
        out_ratio = factor**2
        self.norm = Normalize(
            in_channels=in_channels, num_groups=32, norm_type=norm_type
        )
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * out_ratio,
            kernel_size,
            padding=get_same_padding(kernel_size),
            padding_mode=padding_mode,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(self.norm(x))
        x = F.pixel_shuffle(x, self.factor)
        return x


class InterpolateConvUpSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        factor: int,
        mode: str = "nearest",
        padding_mode: str = "zeros",
        norm_type: Optional[str] = None,
        norm_keep: bool = False,
    ) -> None:
        super().__init__()
        self.factor = factor
        self.mode = mode
        self.norm_keep = norm_keep
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            1,
            padding=get_same_padding(kernel_size),
            padding_mode=padding_mode,
        )
        # self.norm = Normalize(
        #     in_channels=in_channels, num_groups=32, norm_type=self.norm_type
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, scale_factor=self.factor, mode=self.mode)
        x_resp = self.conv(x)
        if self.norm_keep:
            x_resp = resample_norm_keep(x, x_resp)
        return x_resp


class ChannelDuplicatingPixelUnshuffleUpSampleLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor = factor
        assert out_channels * factor**2 % in_channels == 0
        self.repeats = out_channels * factor**2 // in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = F.pixel_shuffle(x, self.factor)
        return x


# * --- Upsample and downsample entries --- #


def build_upsample_block(
    block_type: str,
    in_channels: int,
    out_channels: int,
    shortcut: Optional[str],
    padding_mode: str = "zeros",
    norm_type: str | None = None,  # deprecated
    norm_keep: bool = False,
) -> nn.Module:
    log_print(
        f"[build_upsample_block] block_type: {block_type}, "
        f"in_channels: {in_channels}, "
        f"out_channels: {out_channels}, "
        f"shortcut: {shortcut}, "
        f"padding_mode: {padding_mode}, "
        f"norm keep: {norm_keep}",
        "debug",
    )

    if block_type == "ConvPixelShuffle":
        block = ConvPixelShuffleUpSampleLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            factor=2,
            padding_mode=padding_mode,
        )
    elif block_type == "RepeatConv":
        block = UpsampleRepeatConv(
            in_channels, padding_mode=padding_mode, norm_keep=norm_keep
        )
    elif block_type == "InterpolateConv":
        block = InterpolateConvUpSampleLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            factor=2,
            padding_mode=padding_mode,
            mode="nearest",
            norm_keep=norm_keep,
        )
    else:
        raise ValueError(f"block_type {block_type} is not supported for upsampling")

    if shortcut is None:
        pass
    elif shortcut == "duplicating":
        shortcut_block = ChannelDuplicatingPixelUnshuffleUpSampleLayer(
            in_channels=in_channels, out_channels=out_channels, factor=2
        )
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for upsample")
    return block


def build_downsample_block(
    block_type: str,
    in_channels: int,
    out_channels: int,
    shortcut: Optional[str],
    padding_mode: str = "zeros",
    *,
    padconv_use_manually_pad: bool = True,  # for the compatibility with cosmos checkpoints
    norm_type: str | None = None,
    norm_keep: bool = False,
) -> nn.Module:
    log_print(
        f"[build_downsample_block] block_type: {block_type}, "
        f"in_channels: {in_channels}, "
        f"out_channels: {out_channels}, "
        f"shortcut: {shortcut}, "
        f"padding_mode: {padding_mode} "
        f"padconv_use_manually_pad: {padconv_use_manually_pad}, "
        f"norm keep: {norm_keep}, ",
        "debug",
    )

    if block_type == "Conv":
        block = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            padding_mode=padding_mode,
            bias=True,
        )
    elif block_type == "PadConv":
        block = DownsamplePadConv(
            in_channels=in_channels,
            padding_in_conv=not padconv_use_manually_pad,
            padding_mode=padding_mode,
            norm_keep=norm_keep,
        )
    elif block_type == "ConvPixelUnshuffle":
        block = ConvPixelUnshuffleDownSampleLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            factor=2,
            padding_mode=padding_mode,
        )
    else:
        raise ValueError(f"block_type {block_type} is not supported for downsampling")

    if shortcut is None:
        pass
    elif shortcut == "averaging":
        shortcut_block = PixelUnshuffleChannelAveragingDownSampleLayer(
            in_channels=in_channels, out_channels=out_channels, factor=2
        )
        block = ResidualBlock(block, shortcut_block)
    else:
        raise ValueError(f"shortcut {shortcut} is not supported for downsample")
    return block

