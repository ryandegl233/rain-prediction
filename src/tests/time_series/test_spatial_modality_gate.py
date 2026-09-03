import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.networks.time_series.causal_patch_transformer_next_frame import RainCausalPatchTransformerNextFrame


def build_model(**overrides: object) -> RainCausalPatchTransformerNextFrame:
    settings = {
        "in_channels": 12,
        "radar_out_channels": 1,
        "satellite_out_channels": 10,
        "rain_out_channels": 1,
        "input_size": 16,
        "patch_size": 4,
        "stem_channels": 16,
        "dim": 32,
        "depth": 1,
        "num_heads": 4,
        "decoder_base_channels": 16,
        "dropout": 0.0,
        "drop_path": 0.0,
        "max_frames": 8,
    }
    settings.update(overrides)
    return RainCausalPatchTransformerNextFrame(**settings)


def test_neutral_gate_preserves_complete_predictions() -> None:
    torch.manual_seed(7)
    baseline = build_model().eval()
    gated = build_model(spatial_modality_gate_enabled=True).eval()
    result = gated.load_state_dict(baseline.state_dict(), strict=False)
    assert not result.unexpected_keys
    assert result.missing_keys
    assert all(key.startswith("spatial_modality_gate.") for key in result.missing_keys)
    x = torch.randn(1, 12, 4, 16, 16)
    with torch.no_grad():
        original = baseline(x, return_modality_dict=True)
        updated = gated(x, return_modality_dict=True)
    for name in ("radar", "satellite", "rain"):
        torch.testing.assert_close(updated[name], original[name], rtol=0, atol=0)


def test_gate_preserves_initialization_and_encoded_tokens() -> None:
    torch.manual_seed(11)
    baseline = build_model().eval()
    torch.manual_seed(11)
    gated = build_model(spatial_modality_gate_enabled=True).eval()
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(gated.state_dict()[key], value, rtol=0, atol=0)
    x = torch.randn(1, 12, 4, 16, 16)
    with torch.no_grad():
        torch.testing.assert_close(gated._encode_tokens(x)[0], baseline._encode_tokens(x)[0], rtol=0, atol=0)


@pytest.mark.parametrize("encoder_type", ["patch", "resnet"])
def test_disabled_gate_strictly_loads_baseline_weights(encoder_type: str) -> None:
    settings = {"encoder_type": encoder_type, "encoder_spatial_downsample_stages": 2}
    baseline = build_model(**settings).eval()
    disabled = build_model(spatial_modality_gate_enabled=False, **settings).eval()
    assert disabled.spatial_modality_gate is None
    assert not any(key.startswith("spatial_modality_gate.") for key in disabled.state_dict())
    disabled.load_state_dict(baseline.state_dict(), strict=True)
    x = torch.randn(1, 12, 4, 16, 16)
    with torch.no_grad():
        torch.testing.assert_close(disabled(x)["rain"], baseline(x)["rain"], rtol=0, atol=0)


def test_non_neutral_fusion_keeps_rain_and_bias_unscaled() -> None:
    model = build_model(spatial_modality_gate_enabled=True).eval()
    with torch.no_grad():
        model.patch_embed.weight[:, :1].fill_(0.125)
        model.patch_embed.weight[:, 1:11].fill_(0.25)
        model.patch_embed.weight[:, 11:].fill_(0.5)
        model.patch_embed.bias.fill_(7.0)
        model.spatial_modality_gate.net[-1].bias.copy_(torch.tensor([math.atanh(0.5), math.atanh(-0.5)]))
    x = torch.empty(1, 12, 4, 16, 16)
    x[:, :1] = 2.0
    x[:, 1:11] = 3.0
    x[:, 11:] = 4.0
    fused_inputs: list[torch.Tensor] = []

    def capture_fused_input(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        fused_inputs.append(inputs[0].detach().clone())

    handle = model.stem.register_forward_pre_hook(capture_fused_input)
    try:
        with torch.no_grad():
            model._encode_tokens(x)
            x[:, 11:] = 5.0
            model._encode_tokens(x)
    finally:
        handle.remove()
    torch.testing.assert_close(fused_inputs[0], torch.full((4, 16, 4, 4), 105.0), rtol=1e-6, atol=1e-5)
    torch.testing.assert_close(fused_inputs[1], torch.full((4, 16, 4, 4), 113.0), rtol=1e-6, atol=1e-5)


def test_gates_have_spatial_variation_without_cross_time_mixing() -> None:
    model = build_model(spatial_modality_gate_enabled=True).eval()
    gate = model.spatial_modality_gate
    with torch.no_grad():
        for layer in gate.net:
            if isinstance(layer, nn.Conv2d):
                layer.weight.fill_(0.01)
                layer.bias.zero_()
        gate.net[-1].weight[1].neg_()
    radar = torch.zeros(2, 16, 4, 4, 4)
    radar[:, :, :, 1, 1] = 2.0
    satellite = torch.ones_like(radar)
    rain = torch.zeros_like(radar)
    changed_rain = rain.clone()
    changed_rain[:, :, -1] = 10.0
    with torch.no_grad():
        gates = gate(radar, satellite, rain)
        changed = gate(radar, satellite, changed_rain)
    for before, after in zip(gates, changed, strict=True):
        assert before.shape == (2, 1, 4, 4, 4)
        assert torch.isfinite(before).all()
        assert ((before > 0) & (before < 2)).all()
        assert before[0, 0, 0].std() > 0
        torch.testing.assert_close(before[:, :, :-1], after[:, :, :-1], rtol=0, atol=0)
        assert not torch.equal(before[:, :, -1], after[:, :, -1])
    assert not torch.equal(gates[0], gates[1])


@pytest.mark.parametrize("return_modality_dict", [True, False])
@pytest.mark.parametrize("available", [None, [False, True, False], [True, False, True]])
def test_neutral_gate_preserves_ar_and_mask_tokens(
    return_modality_dict: bool, available: list[bool] | None
) -> None:
    baseline = build_model().eval()
    gated = build_model(spatial_modality_gate_enabled=True).eval()
    gated.load_state_dict(baseline.state_dict(), strict=False)
    context = torch.randn(1, 12, 4, 16, 16)
    target = torch.randn(1, 12, 2, 16, 16)
    availability = None if available is None else torch.tensor([available])
    with torch.no_grad():
        expected = baseline.forward_ar(
            context, target, context_modality_available=availability, return_modality_dict=return_modality_dict
        )
        actual = gated.forward_ar(
            context, target, context_modality_available=availability, return_modality_dict=return_modality_dict
        )
    if return_modality_dict:
        assert set(actual) == {"radar", "satellite", "rain"}
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    else:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_gate_and_original_patch_weights_receive_gradients() -> None:
    torch.manual_seed(23)
    model = build_model(spatial_modality_gate_enabled=True).eval()
    x = torch.randn(1, 12, 4, 16, 16)
    head = model.spatial_modality_gate.net[-1]
    optimizer = torch.optim.SGD(head.parameters(), lr=0.1)
    prediction = model(x)
    sum(value.square().mean() for value in prediction.values()).backward()
    for parameter in (model.patch_embed.weight, head.weight, head.bias):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    optimizer.step()
    model.zero_grad(set_to_none=True)
    prediction = model(x)
    sum(value.square().mean() for value in prediction.values()).backward()
    for layer in (model.spatial_modality_gate.net[0], model.spatial_modality_gate.net[2]):
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()
        assert layer.weight.grad.abs().sum() > 0


def test_gated_weights_strictly_roundtrip(tmp_path: Path) -> None:
    model = build_model(spatial_modality_gate_enabled=True).eval()
    with torch.no_grad():
        model.spatial_modality_gate.net[-1].weight.fill_(0.05)
    checkpoint = tmp_path / "gated.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = build_model(spatial_modality_gate_enabled=True).eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
    x = torch.randn(1, 12, 4, 16, 16)
    with torch.no_grad():
        expected, actual = model(x), restored(x)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"encoder_type": "resnet"}, "encoder_type"),
        ({"encoder_type": "unknown"}, "encoder_type"),
        ({"frame_patch_size": 2}, "frame_patch_size"),
        ({"spatial_modality_gate_hidden_channels": 0}, "hidden_channels"),
        ({"spatial_modality_gate_hidden_channels": -1}, "hidden_channels"),
        ({"radar_out_channels": 0}, "channels"),
        ({"satellite_out_channels": -1}, "channels"),
        ({"rain_out_channels": 0}, "channels"),
        ({"in_channels": 13}, "channels"),
    ],
)
def test_invalid_gate_configuration_fails_early(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_model(spatial_modality_gate_enabled=True, **overrides)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU unavailable; GPU path not validated")
@pytest.mark.parametrize("autocast_enabled", [False, True], ids=["fp32", "bf16"])
def test_cuda_neutral_gate_forward_backward(autocast_enabled: bool) -> None:
    if autocast_enabled and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device lacks bf16 support; bf16 path not validated")
    baseline = build_model(activation_checkpoint=True).cuda().eval()
    gated = build_model(spatial_modality_gate_enabled=True, activation_checkpoint=True).cuda().eval()
    gated.load_state_dict(baseline.state_dict(), strict=False)
    x = torch.randn(1, 12, 4, 16, 16, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        with torch.no_grad():
            expected = baseline(x)
        actual = gated(x)
        loss = sum(value.float().square().mean() for value in actual.values())
    for name in expected:
        assert actual[name].is_cuda
        assert torch.isfinite(actual[name]).all()
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    loss.backward()
    for parameter in (gated.patch_embed.weight, gated.spatial_modality_gate.net[-1].weight):
        assert parameter.grad is not None
        assert parameter.grad.is_cuda
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
