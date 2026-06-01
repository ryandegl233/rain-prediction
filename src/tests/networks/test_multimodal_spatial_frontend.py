import torch

from src.networks.spatial_rain_upsample.upsampler import (
    MultimodalSpatialEnhancementFrontend,
    SpatialFrontendOutput,
    build_spatial_guide,
    frontend_unsupervised_loss,
    psnr,
    resize_bcthw,
    spectral_degradation_loss,
    spatial_gradient_loss,
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
        temporal_chunk_size=1,
    )

    out = model(radar=radar, satellite=satellite, rain=rain)

    assert out.radar.shape == (2, 1, 3, 32, 32)
    assert out.satellite.shape == (2, 10, 3, 32, 32)
    assert out.rain.shape == (2, 1, 3, 32, 32)


def test_temporal_chunk_size_one_matches_full_time_output_shapes() -> None:
    radar, satellite, rain = _make_inputs(batch=1, frames=3, size=10)
    base_model = MultimodalSpatialEnhancementFrontend(
        feature_channels=4,
        shared_depth=1,
        output_size=(18, 18),
        temporal_chunk_size=None,
    )
    chunked_model = MultimodalSpatialEnhancementFrontend(
        feature_channels=4,
        shared_depth=1,
        output_size=(18, 18),
        temporal_chunk_size=1,
    )
    chunked_model.load_state_dict(base_model.state_dict())
    base_model.eval()
    chunked_model.eval()

    with torch.no_grad():
        full = base_model(radar=radar, satellite=satellite, rain=rain)
        chunked = chunked_model(radar=radar, satellite=satellite, rain=rain)

    assert chunked.radar.shape == full.radar.shape == (1, 1, 3, 18, 18)
    assert chunked.satellite.shape == full.satellite.shape == (1, 10, 3, 18, 18)
    assert chunked.rain.shape == full.rain.shape == (1, 1, 3, 18, 18)
    assert torch.allclose(chunked.radar, full.radar)
    assert torch.allclose(chunked.satellite, full.satellite)
    assert torch.allclose(chunked.rain, full.rain)


def test_frontend_v1_outputs_1024_from_256() -> None:
    radar, satellite, rain = _make_inputs(batch=1, frames=1, size=256)
    model = MultimodalSpatialEnhancementFrontend(
        feature_channels=1,
        shared_depth=0,
        output_size=(1024, 1024),
        temporal_chunk_size=1,
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


def test_rain_output_uses_multimodal_gate_without_direct_fused_residual() -> None:
    torch.manual_seed(7)
    _radar, _satellite, rain = _make_inputs(batch=1, frames=1, size=8)
    radar_a = torch.rand(1, 1, 1, 8, 8)
    radar_b = torch.rand(1, 1, 1, 8, 8)
    satellite_a = torch.rand(1, 10, 1, 8, 8)
    satellite_b = torch.rand(1, 10, 1, 8, 8)
    model = MultimodalSpatialEnhancementFrontend(
        feature_channels=4,
        shared_depth=0,
        output_size=(16, 16),
        temporal_chunk_size=1,
    )
    with torch.no_grad():
        model.rain_head.weight.fill_(0.1)
        model.rain_gate_head.weight.fill_(0.1)
        model.rain_gate_head.bias.zero_()

    out_a = model(radar=radar_a, satellite=satellite_a, rain=rain)
    out_b = model(radar=radar_b, satellite=satellite_b, rain=rain)

    assert out_a.rain_gate is not None
    assert out_b.rain_gate is not None
    assert not torch.allclose(out_a.rain_gate, out_b.rain_gate)
    assert not torch.allclose(out_a.rain, out_b.rain)


def test_chunked_zero_initialized_heads_match_interpolation_basis() -> None:
    radar, satellite, rain = _make_inputs(batch=1, frames=4, size=12)
    model = MultimodalSpatialEnhancementFrontend(
        feature_channels=4,
        shared_depth=1,
        output_size=(20, 20),
        residual_scale=0.25,
        temporal_chunk_size=1,
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
        temporal_chunk_size=1,
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


def test_rain_spatial_loss_uses_rain_only_guide() -> None:
    radar_base = torch.rand(1, 1, 1, 8, 8)
    satellite_base = torch.rand(1, 10, 1, 8, 8)
    rain_base = torch.rand(1, 1, 1, 8, 8)
    residual = torch.zeros_like(rain_base)
    output = SpatialFrontendOutput(
        radar=radar_base,
        satellite=satellite_base,
        rain=rain_base,
        radar_base=radar_base,
        satellite_base=satellite_base,
        rain_base=rain_base,
        radar_residual=residual,
        satellite_residual=torch.zeros_like(satellite_base),
        rain_residual=residual,
    )

    loss, _logs = frontend_unsupervised_loss(
        output,
        {"radar": radar_base, "satellite": satellite_base, "rain": rain_base},
        spectral_weight=0.0,
        spatial_weight=1.0,
        residual_weight=0.0,
        guide_weights={"radar": 1.0, "satellite": 1.0, "rain": 1.0},
    )
    shared_guide = build_spatial_guide(radar_base, satellite_base, rain_base)
    expected = (
        spatial_gradient_loss(output.radar, shared_guide)
        + spatial_gradient_loss(output.satellite, shared_guide)
        + spatial_gradient_loss(output.rain, rain_base)
    ) / 3.0

    assert torch.allclose(loss, expected)


def test_degradation_loss_downsamples_to_low_reference_shape() -> None:
    low = torch.rand(1, 1, 1, 256, 256)
    high = resize_bcthw(low, size=(1024, 1024))

    loss = spectral_degradation_loss(high, low)
    degraded = resize_bcthw(high, size=(256, 256), mode="area")

    assert degraded.shape == low.shape
    assert torch.isfinite(loss)


def test_frontend_metrics_are_finite() -> None:
    x = torch.rand(2, 1, 3, 16, 16)
    y = x.clone()

    assert torch.isfinite(psnr(x, y))
    assert torch.isfinite(ssim_global(x, y))
    assert ssim_global(x, y) > 0.99
