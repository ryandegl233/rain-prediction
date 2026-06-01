"""
Multimodal spatial enhancement frontend for rain prediction.

The module is intentionally independent from a specific RainPred baseline.  It
keeps radar, satellite, and rain as separate shallow branches, then uses a
shared residual trunk to learn interpolation-basis plus residual compensation.
"""

from dataclasses import dataclass
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


TensorDict = dict[str, torch.Tensor]


def _valid_gn_groups(channels: int, max_groups: int = 8) -> int:
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _as_bcthw(x: torch.Tensor, *, name: str) -> torch.Tensor:
    if x.ndim == 4:
        return x.unsqueeze(2)
    if x.ndim != 5:
        raise ValueError(f"{name} must be [B,C,T,H,W] or [B,C,H,W], got {tuple(x.shape)}")
    return x


def _flatten_time(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int, int]]:
    b, c, t, h, w = x.shape
    return x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w), (b, c, t, h, w)


def _unflatten_time(x: torch.Tensor, shape: tuple[int, int, int, int, int]) -> torch.Tensor:
    b, _c, t, _h, _w = shape
    _, c, h, w = x.shape
    return x.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()


def resize_bcthw(
    x: torch.Tensor,
    *,
    size: tuple[int, int] | None = None,
    scale_factor: float | None = None,
    mode: str = "bilinear",
) -> torch.Tensor:
    x = _as_bcthw(x, name="x")
    flat, shape = _flatten_time(x)
    align_corners = False if mode in {"bilinear", "bicubic"} else None
    resized = F.interpolate(flat, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)
    return _unflatten_time(resized, shape)


class ResidualBlock2D(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        groups = _valid_gn_groups(channels)
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Dropout2d(float(dropout)) if dropout > 0 else nn.Identity(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ModalityStem2D(nn.Module):
    def __init__(self, in_channels: int, feature_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_valid_gn_groups(feature_channels), feature_channels),
            nn.SiLU(),
            ResidualBlock2D(feature_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass(frozen=True)
class SpatialFrontendOutput:
    radar: torch.Tensor
    satellite: torch.Tensor
    rain: torch.Tensor
    radar_base: torch.Tensor
    satellite_base: torch.Tensor
    rain_base: torch.Tensor
    radar_residual: torch.Tensor
    satellite_residual: torch.Tensor
    rain_residual: torch.Tensor

    def enhanced(self) -> TensorDict:
        return {"radar": self.radar, "satellite": self.satellite, "rain": self.rain}

    def bases(self) -> TensorDict:
        return {"radar": self.radar_base, "satellite": self.satellite_base, "rain": self.rain_base}

    def residuals(self) -> TensorDict:
        return {"radar": self.radar_residual, "satellite": self.satellite_residual, "rain": self.rain_residual}


class MultimodalSpatialEnhancementFrontend(nn.Module):
    """
    Unified frontend with modality-specific shallow stems.

    Input tensors follow the project convention [B,C,T,H,W].  The frontend first
    builds an interpolation basis at the target spatial size, then predicts small
    residual corrections for each modality.
    """

    def __init__(
        self,
        radar_channels: int = 1,
        satellite_channels: int = 10,
        rain_channels: int = 1,
        feature_channels: int = 32,
        shared_depth: int = 4,
        scale_factor: float = 1.0,
        output_size: tuple[int, int] | None = None,
        residual_scale: float = 0.1,
        upsample_mode: str = "bilinear",
        dropout: float = 0.0,
        clamp_rain_min: float | None = None,
        temporal_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        if scale_factor <= 0:
            raise ValueError(f"scale_factor must be > 0, got {scale_factor}")
        if output_size is not None and (len(output_size) != 2 or output_size[0] <= 0 or output_size[1] <= 0):
            raise ValueError(f"output_size must be positive (H, W), got {output_size}")
        if feature_channels <= 0:
            raise ValueError(f"feature_channels must be > 0, got {feature_channels}")
        if temporal_chunk_size is not None and temporal_chunk_size <= 0:
            raise ValueError(f"temporal_chunk_size must be > 0 or None, got {temporal_chunk_size}")

        self.radar_channels = int(radar_channels)
        self.satellite_channels = int(satellite_channels)
        self.rain_channels = int(rain_channels)
        self.scale_factor = float(scale_factor)
        self.output_size = tuple(int(v) for v in output_size) if output_size is not None else None
        self.residual_scale = float(residual_scale)
        self.upsample_mode = str(upsample_mode)
        self.clamp_rain_min = clamp_rain_min
        self.temporal_chunk_size = None if temporal_chunk_size is None else int(temporal_chunk_size)

        self.radar_stem = ModalityStem2D(self.radar_channels, feature_channels)
        self.satellite_stem = ModalityStem2D(self.satellite_channels, feature_channels)
        self.rain_stem = ModalityStem2D(self.rain_channels, feature_channels)

        shared_channels = feature_channels * 3
        blocks: list[nn.Module] = [nn.Conv2d(shared_channels, shared_channels, kernel_size=3, padding=1)]
        for _ in range(int(shared_depth)):
            blocks.append(ResidualBlock2D(shared_channels, dropout=dropout))
        self.shared_trunk = nn.Sequential(*blocks)

        self.radar_head = nn.Conv2d(shared_channels, self.radar_channels, kernel_size=3, padding=1)
        self.satellite_head = nn.Conv2d(shared_channels, self.satellite_channels, kernel_size=3, padding=1)
        self.rain_head = nn.Conv2d(shared_channels, self.rain_channels, kernel_size=3, padding=1)

        for head in (self.radar_head, self.satellite_head, self.rain_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _target_size(self, radar: torch.Tensor) -> tuple[int, int] | None:
        if self.output_size is not None:
            return self.output_size
        if self.scale_factor == 1.0:
            return None
        h, w = int(radar.shape[-2]), int(radar.shape[-1])
        return max(1, round(h * self.scale_factor)), max(1, round(w * self.scale_factor))

    def _resize_input(self, x: torch.Tensor, *, target_size: tuple[int, int] | None) -> torch.Tensor:
        if target_size is None:
            return _as_bcthw(x, name="x")
        return resize_bcthw(x, size=target_size, mode=self.upsample_mode)

    def _forward_chunk(
        self,
        radar: torch.Tensor,
        satellite: torch.Tensor,
        rain: torch.Tensor,
        *,
        target_size: tuple[int, int] | None,
    ) -> SpatialFrontendOutput:
        radar_base = self._resize_input(radar, target_size=target_size)
        satellite_base = self._resize_input(satellite, target_size=target_size)
        rain_base = self._resize_input(rain, target_size=target_size)

        radar_flat, shape = _flatten_time(radar_base)
        satellite_flat, _ = _flatten_time(satellite_base)
        rain_flat, _ = _flatten_time(rain_base)

        features = torch.cat(
            [
                self.radar_stem(radar_flat),
                self.satellite_stem(satellite_flat),
                self.rain_stem(rain_flat),
            ],
            dim=1,
        )
        shared = self.shared_trunk(features)
        radar_residual = _unflatten_time(self.radar_head(shared), shape)
        satellite_residual = _unflatten_time(self.satellite_head(shared), shape)
        rain_residual = _unflatten_time(self.rain_head(shared), shape)

        radar_out = radar_base + self.residual_scale * radar_residual
        satellite_out = satellite_base + self.residual_scale * satellite_residual
        rain_out = rain_base + self.residual_scale * rain_residual
        if self.clamp_rain_min is not None:
            rain_out = rain_out.clamp_min(float(self.clamp_rain_min))

        return SpatialFrontendOutput(
            radar=radar_out,
            satellite=satellite_out,
            rain=rain_out,
            radar_base=radar_base,
            satellite_base=satellite_base,
            rain_base=rain_base,
            radar_residual=radar_residual,
            satellite_residual=satellite_residual,
            rain_residual=rain_residual,
        )

    @staticmethod
    def _cat_outputs(chunks: list[SpatialFrontendOutput]) -> SpatialFrontendOutput:
        return SpatialFrontendOutput(
            radar=torch.cat([chunk.radar for chunk in chunks], dim=2),
            satellite=torch.cat([chunk.satellite for chunk in chunks], dim=2),
            rain=torch.cat([chunk.rain for chunk in chunks], dim=2),
            radar_base=torch.cat([chunk.radar_base for chunk in chunks], dim=2),
            satellite_base=torch.cat([chunk.satellite_base for chunk in chunks], dim=2),
            rain_base=torch.cat([chunk.rain_base for chunk in chunks], dim=2),
            radar_residual=torch.cat([chunk.radar_residual for chunk in chunks], dim=2),
            satellite_residual=torch.cat([chunk.satellite_residual for chunk in chunks], dim=2),
            rain_residual=torch.cat([chunk.rain_residual for chunk in chunks], dim=2),
        )

    def forward(
        self,
        radar: torch.Tensor,
        satellite: torch.Tensor,
        rain: torch.Tensor,
        *,
        return_dict: bool = True,
    ) -> SpatialFrontendOutput | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        radar = _as_bcthw(radar, name="radar")
        satellite = _as_bcthw(satellite, name="satellite")
        rain = _as_bcthw(rain, name="rain")
        if radar.shape[0] != satellite.shape[0] or radar.shape[0] != rain.shape[0]:
            raise ValueError("radar/satellite/rain batch sizes must match")
        if radar.shape[2:] != satellite.shape[2:] or radar.shape[2:] != rain.shape[2:]:
            raise ValueError(
                "radar/satellite/rain temporal and spatial shapes must match, "
                f"got radar={tuple(radar.shape)}, satellite={tuple(satellite.shape)}, rain={tuple(rain.shape)}"
            )
        if radar.shape[1] != self.radar_channels:
            raise ValueError(f"radar channel mismatch: expected {self.radar_channels}, got {radar.shape[1]}")
        if satellite.shape[1] != self.satellite_channels:
            raise ValueError(
                f"satellite channel mismatch: expected {self.satellite_channels}, got {satellite.shape[1]}"
            )
        if rain.shape[1] != self.rain_channels:
            raise ValueError(f"rain channel mismatch: expected {self.rain_channels}, got {rain.shape[1]}")

        target_size = self._target_size(radar)
        chunk_size = self.temporal_chunk_size
        if chunk_size is None or chunk_size >= int(radar.shape[2]):
            output = self._forward_chunk(radar, satellite, rain, target_size=target_size)
        else:
            chunks: list[SpatialFrontendOutput] = []
            for start in range(0, int(radar.shape[2]), chunk_size):
                end = min(start + chunk_size, int(radar.shape[2]))
                chunks.append(
                    self._forward_chunk(
                        radar[:, :, start:end],
                        satellite[:, :, start:end],
                        rain[:, :, start:end],
                        target_size=target_size,
                    )
                )
            output = self._cat_outputs(chunks)

        if not return_dict:
            return output.radar, output.satellite, output.rain
        return output


def _gradient_xy(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    dx = x[..., :, 1:] - x[..., :, :-1]
    dy = x[..., 1:, :] - x[..., :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    return dx, dy


def _channel_mean(x: torch.Tensor) -> torch.Tensor:
    return x.mean(dim=1, keepdim=True) if x.shape[1] > 1 else x


def build_spatial_guide(
    radar: torch.Tensor,
    satellite: torch.Tensor,
    rain: torch.Tensor,
    *,
    weights: Mapping[str, float] | None = None,
) -> torch.Tensor:
    weights = weights or {"radar": 1.0, "satellite": 1.0, "rain": 1.0}
    parts: list[torch.Tensor] = []
    total = 0.0
    for name, tensor in (("radar", radar), ("satellite", satellite), ("rain", rain)):
        weight = float(weights.get(name, 0.0))
        if weight <= 0:
            continue
        parts.append(weight * _channel_mean(_as_bcthw(tensor, name=name)))
        total += weight
    if not parts or total <= 0:
        raise ValueError("at least one positive spatial guide weight is required")
    return torch.stack(parts, dim=0).sum(dim=0) / total


def spectral_degradation_loss(enhanced: torch.Tensor, low_ref: torch.Tensor) -> torch.Tensor:
    enhanced = _as_bcthw(enhanced, name="enhanced")
    low_ref = _as_bcthw(low_ref, name="low_ref")
    degraded = resize_bcthw(enhanced, size=(int(low_ref.shape[-2]), int(low_ref.shape[-1])), mode="area")
    return F.mse_loss(degraded, low_ref)


def spatial_gradient_loss(enhanced: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
    enhanced = _channel_mean(_as_bcthw(enhanced, name="enhanced"))
    guide = _channel_mean(_as_bcthw(guide, name="guide"))
    if enhanced.shape[-2:] != guide.shape[-2:]:
        guide = resize_bcthw(guide, size=(int(enhanced.shape[-2]), int(enhanced.shape[-1])), mode="bilinear")
    ex, ey = _gradient_xy(enhanced)
    gx, gy = _gradient_xy(guide)
    return F.l1_loss(ex, gx) + F.l1_loss(ey, gy)


def residual_energy_loss(residuals: Mapping[str, torch.Tensor]) -> torch.Tensor:
    values = [value.pow(2).mean() for value in residuals.values()]
    if not values:
        raise ValueError("residuals must not be empty")
    return torch.stack(values).mean()


def frontend_unsupervised_loss(
    output: SpatialFrontendOutput,
    low_inputs: Mapping[str, torch.Tensor],
    *,
    spectral_weight: float = 1.0,
    spatial_weight: float = 0.1,
    residual_weight: float = 1.0e-4,
    guide_weights: Mapping[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    spectral = (
        spectral_degradation_loss(output.radar, low_inputs["radar"])
        + spectral_degradation_loss(output.satellite, low_inputs["satellite"])
        + spectral_degradation_loss(output.rain, low_inputs["rain"])
    ) / 3.0
    guide = build_spatial_guide(output.radar_base, output.satellite_base, output.rain_base, weights=guide_weights)
    spatial = (
        spatial_gradient_loss(output.radar, guide)
        + spatial_gradient_loss(output.satellite, guide)
        + spatial_gradient_loss(output.rain, guide)
    ) / 3.0
    residual = residual_energy_loss(output.residuals())
    total = float(spectral_weight) * spectral + float(spatial_weight) * spatial + float(residual_weight) * residual
    logs = {
        "loss/frontend_total": total.detach(),
        "loss/frontend_spectral": spectral.detach(),
        "loss/frontend_spatial": spatial.detach(),
        "loss/frontend_residual": residual.detach(),
    }
    return total, logs


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0, eps: float = 1.0e-8) -> torch.Tensor:
    mse = F.mse_loss(pred, target)
    return 20.0 * torch.log10(torch.as_tensor(float(data_range), device=pred.device, dtype=pred.dtype)) - 10.0 * torch.log10(
        mse.clamp_min(eps)
    )


def ssim_global(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0, eps: float = 1.0e-8) -> torch.Tensor:
    pred = _as_bcthw(pred, name="pred")
    target = _as_bcthw(target, name="target")
    dims = tuple(range(1, pred.ndim))
    c1 = (0.01 * float(data_range)) ** 2
    c2 = (0.03 * float(data_range)) ** 2
    mu_x = pred.mean(dim=dims)
    mu_y = target.mean(dim=dims)
    var_x = pred.var(dim=dims, unbiased=False)
    var_y = target.var(dim=dims, unbiased=False)
    cov = ((pred - mu_x.view(-1, 1, 1, 1, 1)) * (target - mu_y.view(-1, 1, 1, 1, 1))).mean(dim=dims)
    score = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x.pow(2) + mu_y.pow(2) + c1) * (var_x + var_y + c2) + eps)
    return score.mean()
