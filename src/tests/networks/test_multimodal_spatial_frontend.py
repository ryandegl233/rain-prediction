import torch

from src.networks.spatial_rain_upsample.upsampler import (
    MultimodalSpatialEnhancementFrontend,
    frontend_unsupervised_loss,
    psnr,
    resize_bcthw,
    spectral_degradation_loss,
    ssim_global,
)


def _make_inputs(batch: int = 2, frames: int = 3, size: int = 16):
    radar = torch.rand(batch, 1, frames, size, size)
    satellite = torch.rand(batch, 10, frames, size, size)
    rain = torch.rand(batch, 1, frames, size, size)
    return radar, satellite, rain


def test_frontend_outputs_requested_spatial_size() -> None:
    radar, satellite, rain = _make_inputs(size=16)
    model = MultimodalSpatialEnhancementFrontend(
        feature_channels=8,
        shared_depth=1,
        scale_factor=2,
    )

    out = model(radar=radar, satellite=satellite, rain=rain)

    assert out.radar.shape == (2, 1, 3, 32, 32)
    assert out.satellite.shape == (2, 10, 3, 32, 32)
    assert out.rain.shape == (2, 1, 3, 32, 32)


def test_frontend_v1_outputs_1024_from_448() -> None:
    radar, satellite, rain = _make_inputs(batch=1, frames=1, size=448)
    model = MultimodalSpatialEnhancementFrontend(
        feature_channels=1,
        shared_depth=0,
        output_size=(1024, 1024),
    )

    with torch.no_grad():
        out = model(radar=radar, satellite=satellite, rain=rain)

    assert out.radar.shape == (1, 1, 1, 1024, 1024)
    assert out.satellite.shape == (1, 10, 1, 1024, 1024)
    assert out.rain.shape == (1, 1, 1, 1024, 1024)


def test_zero_initialized_heads_match_interpolation_basis() -> None:
    radar, satellite, rain = _make_inputs(size=12)
    model = MultimodalSpatialEnhancementFrontend(
        feature_channels=8,
        shared_depth=1,
        output_size=(20, 20),
        residual_scale=0.25,
    )

    out = model(radar=radar, satellite=satellite, rain=rain)

    assert torch.allclose(out.radar, resize_bcthw(radar, size=(20, 20)))
    assert torch.allclose(out.satellite, resize_bcthw(satellite, size=(20, 20)))
    assert torch.allclose(out.rain, resize_bcthw(rain, size=(20, 20)))


def test_frontend_unsupervised_loss_backward_runs() -> None:
    radar, satellite, rain = _make_inputs(size=16)
    model = MultimodalSpatialEnhancementFrontend(
        feature_channels=8,
        shared_depth=1,
        scale_factor=2,
    )

    out = model(radar=radar, satellite=satellite, rain=rain)
    loss, logs = frontend_unsupervised_loss(
        out,
        {"radar": radar, "satellite": satellite, "rain": rain},
        spectral_weight=1.0,
        spatial_weight=0.1,
        residual_weight=1.0e-4,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert set(logs) == {
        "loss/frontend_total",
        "loss/frontend_spectral",
        "loss/frontend_spatial",
        "loss/frontend_residual",
    }
    assert any(param.grad is not None for param in model.parameters() if param.requires_grad)


def test_degradation_loss_downsamples_to_low_reference_shape() -> None:
    low = torch.rand(1, 1, 1, 448, 448)
    high = resize_bcthw(low, size=(1024, 1024))

    loss = spectral_degradation_loss(high, low)
    degraded = resize_bcthw(high, size=(448, 448), mode="area")

    assert degraded.shape == low.shape
    assert torch.isfinite(loss)


def test_frontend_metrics_are_finite() -> None:
    x = torch.rand(2, 1, 3, 16, 16)
    y = x.clone()

    assert torch.isfinite(psnr(x, y))
    assert torch.isfinite(ssim_global(x, y))
    assert ssim_global(x, y) > 0.99
