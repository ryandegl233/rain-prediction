import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LayerNorm(nn.Module):
    def __init__(self, channels: int, channels_first: bool = False) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.channels_first = channels_first

    def forward(self, x: Tensor) -> Tensor:
        if not self.channels_first:
            return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-6)
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).pow(2).mean(1, keepdim=True)
        return (x - mean) / torch.sqrt(variance + 1e-6) * self.weight[:, None, None] + self.bias[:, None, None]


class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, transpose: bool) -> None:
        super().__init__()
        if transpose:
            self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels * 4, 3, padding=1), nn.PixelShuffle(2))
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.norm = LayerNorm(out_channels, channels_first=True)
        self.act = nn.SiLU(inplace=True)
        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.trunc_normal_(layer.weight, std=0.02)
                nn.init.zeros_(layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))


class ConvSC(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, transpose: bool = False) -> None:
        super().__init__()
        self.conv = BasicConv2d(in_channels, out_channels, stride, transpose and stride != 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class TimeEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(64, 256)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(256, 64)

    def forward(self, x: Tensor) -> Tensor:
        frequency = torch.exp(torch.arange(32, device=x.device, dtype=torch.float32) * (-math.log(10000) / 31))
        phase = x[:, None] * frequency[None, :]
        embedding = torch.cat((phase.sin(), phase.cos()), dim=-1)
        return self.linear2(self.gelu(self.linear1(embedding)))


class LargeKernelAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.conv_spatial = nn.Conv2d(channels, channels, 7, padding=9, groups=channels, dilation=3)
        self.conv1 = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        return x * self.conv1(self.conv_spatial(self.conv0(x)))


class TemporalBlock(nn.Module):
    def __init__(self, channels: int, bottleneck: bool = False) -> None:
        super().__init__()
        self.mlp = nn.Sequential(nn.GELU(), nn.Linear(128, channels))
        self.dwconv = (
            LargeKernelAttention(channels)
            if bottleneck
            else nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        )
        self.norm = LayerNorm(channels)
        self.pwconv1 = nn.Linear(channels, 4 * channels)
        self.act = nn.SiLU(inplace=True)
        self.pwconv2 = nn.Linear(4 * channels, channels)
        self.gamma = nn.Parameter(torch.full((channels,), 1e-6))
        self.drop_path = nn.Identity()
        if bottleneck:
            self.res_conv = nn.Conv2d(channels, channels, 1)

    def forward(self, x: Tensor, day: Tensor, hour: Tensor) -> Tensor:
        residual = self.res_conv(x) if hasattr(self, "res_conv") else x
        time = self.mlp(torch.cat((day, hour), dim=1))[:, :, None, None]
        value = (self.dwconv(x) + time).permute(0, 2, 3, 1)
        value = self.pwconv2(self.act(self.pwconv1(self.norm(value))))
        return residual + self.drop_path((self.gamma * value).permute(0, 3, 1, 2))


class Encoder(nn.Module):
    def __init__(self, channels: int, hidden: int, depth: int) -> None:
        super().__init__()
        strides = ([1, 2] * depth)[:depth]
        self.enc = nn.Sequential(
            ConvSC(channels, hidden, strides[0]),
            *(ConvSC(hidden, hidden, stride) for stride in strides[1:]),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        skip = self.enc[0](x)
        value = skip
        for layer in self.enc[1:]:
            value = layer(value)
        return value, skip


class Decoder(nn.Module):
    def __init__(self, hidden: int, channels: int, depth: int) -> None:
        super().__init__()
        strides = list(reversed(([1, 2] * depth)[:depth]))
        self.dec = nn.Sequential(
            *(ConvSC(hidden, hidden, stride, transpose=True) for stride in strides[:-1]),
            ConvSC(2 * hidden, hidden, strides[-1], transpose=True),
        )
        self.readout = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        for layer in self.dec[:-1]:
            x = layer(x)
        return self.readout(self.dec[-1](torch.cat((x, skip), dim=1)))


class Predictor(nn.Module):
    def __init__(self, channels: int, depth: int) -> None:
        super().__init__()
        self.st_block = nn.ModuleList(
            [TemporalBlock(channels, bottleneck=True)] + [TemporalBlock(channels) for _ in range(depth - 1)]
        )

    def forward(self, x: Tensor, day: Tensor, hour: Tensor) -> Tensor:
        batch, frames, channels, height, width = x.shape
        value = x.reshape(batch, frames * channels, height, width)
        for block in self.st_block:
            value = block(value, day, hour)
        return value.reshape(batch, frames, channels, height, width)


class SatelliteForecaster(nn.Module):
    def __init__(self, channels: int, frames: int, hidden: int, spatial_depth: int, temporal_depth: int) -> None:
        super().__init__()
        self.time_mlp = TimeEmbedding()
        self.time_mlp2 = TimeEmbedding()
        self.enc = Encoder(channels, hidden, spatial_depth)
        self.hid = Predictor(frames * hidden, temporal_depth)
        self.dec = Decoder(hidden, channels, spatial_depth)

    def forward(self, x: Tensor, day: Tensor, hour: Tensor) -> Tensor:
        batch, frames, channels, height, width = x.shape
        day_embedding = self.time_mlp(day)
        hour_embedding = self.time_mlp2(hour)
        encoded, skip = self.enc(x.reshape(batch * frames, channels, height, width))
        hidden, latent_height, latent_width = encoded.shape[1:]
        encoded = encoded.reshape(batch, frames, hidden, latent_height, latent_width)
        predicted = self.hid(encoded, day_embedding, hour_embedding)
        decoded = self.dec(predicted.reshape(batch * frames, hidden, latent_height, latent_width), skip)
        return decoded.reshape(batch, frames, channels, height, width)
