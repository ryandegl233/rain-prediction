import torch
from torch import Tensor, nn

from .npm_port import SatelliteForecaster, SatelliteToRain


class NPMSatelliteOnly(nn.Module):
    def __init__(self, options: dict) -> None:
        super().__init__()
        known_options = {
            "image_size",
            "context_frames",
            "forecast_frames",
            "in_channels",
            "hidden",
            "spatial_depth",
            "temporal_depth",
            "translator_width",
            "translator_downsamples",
            "translator_residual_blocks",
            "translator_norm",
        }
        if unknown := set(options) - known_options:
            raise ValueError(f"Unknown NPM options: {sorted(unknown)}")
        self.image_size = int(options.get("image_size", 448))
        self.context_frames = int(options.get("context_frames", 4))
        self.in_channels = int(options.get("in_channels", 10))
        spatial_depth = int(options.get("spatial_depth", 4))
        temporal_depth = int(options.get("temporal_depth", 8))
        hidden = int(options.get("hidden", 64))
        translator_width = int(options.get("translator_width", 30))
        translator_downsamples = int(options.get("translator_downsamples", 4))
        translator_residual_blocks = int(options.get("translator_residual_blocks", 9))
        translator_norm = options.get("translator_norm", "BatchNorm2d")
        if self.context_frames != 4 or int(options.get("forecast_frames", 4)) != 4:
            raise ValueError("The NPM satellite-only comparison requires exactly four input and four output frames.")
        if self.in_channels != 10:
            raise ValueError("The NPM satellite-only comparison requires all ten satellite channels.")
        if min(hidden, temporal_depth, translator_width, translator_residual_blocks) < 1:
            raise ValueError("NPM widths and depths must be positive.")
        if spatial_depth < 2 or spatial_depth % 2 or translator_downsamples < 1:
            raise ValueError("NPM requires an even spatial depth >= 2 and at least one translator downsample.")
        divisor = max(2 ** (spatial_depth // 2), 2**translator_downsamples)
        if self.image_size % divisor or self.image_size // 2**translator_downsamples < 2:
            raise ValueError("Image size must fit the encoder strides and translator reflection padding.")
        if translator_norm not in {"BatchNorm2d", "InstanceNorm2d"}:
            raise ValueError("translator_norm must be BatchNorm2d or InstanceNorm2d.")

        self.forecaster = SatelliteForecaster(
            self.in_channels, self.context_frames, hidden, spatial_depth, temporal_depth
        )
        self.translator = SatelliteToRain(
            self.in_channels,
            translator_width,
            translator_downsamples,
            translator_residual_blocks,
            translator_norm == "BatchNorm2d",
        )

    def forward(self, satellite: Tensor, time: Tensor | None = None) -> dict[str, Tensor]:
        if satellite.ndim != 5 or tuple(satellite.shape[1:]) != (
            self.in_channels,
            self.context_frames,
            self.image_size,
            self.image_size,
        ):
            raise ValueError("NPM expects satellite [B,10,4,image_size,image_size].")
        if time is None or tuple(time.shape) != (satellite.shape[0], self.context_frames, 2):
            raise ValueError("NPM requires observed time [B,4,2] containing calendar day of year and fractional hour.")
        time = time.to(device=satellite.device, dtype=torch.float32)
        if not bool(torch.isfinite(time).all()):
            raise ValueError("NPM calendar time must be finite.")
        if bool(((time[..., 0] < 1) | (time[..., 0] > 366) | (time[..., 1] < 0) | (time[..., 1] >= 24)).any()):
            raise ValueError("NPM calendar time requires day 1..366 and hour in [0,24).")

        predicted = self.forecaster(satellite.permute(0, 2, 1, 3, 4), time[:, -1, 0], time[:, -1, 1])
        batch, frames, channels, height, width = predicted.shape
        translated = self.translator(predicted.reshape(batch * frames, channels, height, width))
        rain = translated.reshape(batch, frames, 1, height, width)
        return {"rain": rain.permute(0, 2, 1, 3, 4), "satellite": predicted.permute(0, 2, 1, 3, 4)}


def build_npm(options: dict) -> nn.Module:
    return NPMSatelliteOnly(options)
