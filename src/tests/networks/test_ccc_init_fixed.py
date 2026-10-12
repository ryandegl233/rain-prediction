import pytest
import torch
from torch import nn

from src.networks.time_series.causal_patch_transformer_ccc_init_fixed import RainCausalPatchTransformerCCCInitFixed


def make_model(enabled: bool = True) -> RainCausalPatchTransformerCCCInitFixed:
    return RainCausalPatchTransformerCCCInitFixed(
        in_channels=3, radar_out_channels=1, satellite_out_channels=1, rain_out_channels=1,
        input_size=16, patch_size=4, stem_channels=16, dim=16, depth=1, num_heads=4,
        decoder_base_channels=16, dropout=0, drop_path=0,
        cross_modal_adapter_enabled=True, consensus_calibration_enabled=enabled,
        consensus_calibration_bottleneck=8,
    )


def test_initial_output_matches_no_ccc_with_nonzero_outer_gate() -> None:
    torch.manual_seed(2025)
    baseline = make_model(False).eval()
    fixed = make_model().eval()
    missing, unexpected = fixed.load_state_dict(baseline.state_dict(), strict=False)
    assert missing and all("consensus_calibration" in name for name in missing)
    assert not unexpected
    nn.init.constant_(baseline.cross_modal_adapter.gate, 0.2)
    nn.init.constant_(fixed.cross_modal_adapter.gate, 0.2)
    inputs = torch.randn(1, 3, 3, 16, 16)
    with torch.no_grad():
        original = baseline.forward_ar(inputs[:, :, :2], inputs[:, :, 2:], predict_frames=1)
        corrected = fixed.forward_ar(inputs[:, :, :2], inputs[:, :, 2:], predict_frames=1)
    if isinstance(original, dict):
        for key in original:
            torch.testing.assert_close(corrected[key], original[key], rtol=0, atol=0)
    else:
        torch.testing.assert_close(corrected, original, rtol=0, atol=0)


def test_conditional_branch_learns_in_two_backward_passes() -> None:
    torch.manual_seed(2025)
    conditioner = make_model().cross_modal_adapter
    nn.init.constant_(conditioner.gate, 0.2)
    calibration = conditioner.consensus_calibration
    optimizer = torch.optim.SGD(conditioner.parameters(), lr=0.1)
    context = torch.randn(1, 3, 2, 16, 16)
    query = torch.randn(1, 1, 16, 16)
    target = torch.randn_like(query)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = (conditioner(query, context) - target).square().mean()
        loss.backward()
        for network in calibration.conditional:
            assert torch.isfinite(network.net[-1].weight.grad).all()
            assert network.net[-1].weight.grad.count_nonzero() > 0
            if step == 1:
                assert network.net[1].weight.grad.count_nonzero() > 0
        if step == 1:
            assert calibration.condition_scale.grad.count_nonzero() > 0
        optimizer.step()


def test_checkpoint_loading_preserves_learned_scale() -> None:
    model = make_model()
    nn.init.constant_(model.cross_modal_adapter.consensus_calibration.condition_scale, 0.07)
    restored = make_model()
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(
        restored.cross_modal_adapter.consensus_calibration.condition_scale,
        torch.full((3,), 0.07),
    )


@pytest.mark.parametrize("scale", [0, -0.1, float("nan"), float("inf")])
def test_invalid_initial_scale_is_rejected(scale: float) -> None:
    with pytest.raises(ValueError, match="ccc_condition_scale_init"):
        RainCausalPatchTransformerCCCInitFixed(ccc_condition_scale_init=scale)
