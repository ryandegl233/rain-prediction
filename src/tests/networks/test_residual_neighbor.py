import copy

import pytest
import torch
from torch import nn

from src.networks.time_series.causal_patch_transformer_ccc_init_fixed import RainCausalPatchTransformerCCCInitFixed
from src.networks.time_series.causal_patch_transformer_residual_neighbor import (
    PreservedPointwiseConditioner, RainCausalPatchTransformerResidualNeighbor, freeze_original_model,
)
from src.trainer.rain_trainer_ccc_frozen_ablation import frozen_digest


def make_model(radius: int) -> RainCausalPatchTransformerResidualNeighbor:
    return RainCausalPatchTransformerResidualNeighbor(
        in_channels=3, radar_out_channels=1, satellite_out_channels=1, rain_out_channels=1,
        input_size=16, patch_size=4, stem_channels=16, dim=16, depth=1, num_heads=4,
        decoder_base_channels=16, dropout=0, drop_path=0,
        cross_modal_adapter_enabled=True, consensus_calibration_enabled=False,
        correction_radius=radius, correction_chunk_size=5,
    )


@pytest.mark.parametrize("radius", [0, 1])
def test_initial_predictions_equal_original_and_ablating_branch_restores_original(radius: int) -> None:
    torch.manual_seed(12)
    model = make_model(radius).eval()
    baseline = RainCausalPatchTransformerCCCInitFixed(
        in_channels=3, radar_out_channels=1, satellite_out_channels=1, rain_out_channels=1,
        input_size=16, patch_size=4, stem_channels=16, dim=16, depth=1, num_heads=4,
        decoder_base_channels=16, dropout=0, drop_path=0,
        cross_modal_adapter_enabled=True, consensus_calibration_enabled=False,
    ).eval()
    baseline.cross_modal_adapter.gate.data.fill_(0.03)
    missing, unexpected = model.load_state_dict(baseline.state_dict(), strict=False)
    assert missing and all(key.startswith("cross_modal_adapter.correction.") for key in missing)
    assert not unexpected
    model.cross_modal_adapter.initialize_correction()
    inputs = torch.randn(1, 3, 3, 16, 16)
    with torch.no_grad():
        old = baseline.forward_ar(inputs[:, :, :2], inputs[:, :, 2:])
        new = model.forward_ar(inputs[:, :, :2], inputs[:, :, 2:])
    for key in old:
        torch.testing.assert_close(old[key], new[key], rtol=0, atol=0)
    model.cross_modal_adapter.correction.attention.out_proj.weight.data.normal_()
    model.cross_modal_adapter.correction_enabled = False
    with torch.no_grad():
        disabled = model.forward_ar(inputs[:, :, :2], inputs[:, :, 2:])
    for key in old:
        torch.testing.assert_close(old[key], disabled[key], rtol=0, atol=0)


def test_decoder_transmits_gradients_and_original_weights_never_change() -> None:
    torch.manual_seed(15)
    model = make_model(1)
    freeze_original_model(model)
    before = frozen_digest(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    inputs = torch.randn(1, 3, 3, 16, 16)
    target = torch.randn(1, 1, 1, 16, 16)
    correction = model.cross_modal_adapter.correction
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        output = model.forward_ar(inputs[:, :, :2], inputs[:, :, 2:])["rain"]
        (output - target).square().mean().backward()
        assert correction.attention.out_proj.weight.grad.count_nonzero() > 0
        if step == 1:
            assert correction.attention.in_proj_weight.grad.count_nonzero() > 0
            assert all(adapter.weight.grad.count_nonzero() > 0 for adapter in correction.adapters)
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        optimizer.step()
    assert frozen_digest(model) == before


def test_trained_checkpoint_reload_does_not_zero_correction() -> None:
    model = make_model(1).eval()
    nn.init.normal_(model.cross_modal_adapter.correction.attention.out_proj.weight, std=0.02)
    restored = make_model(1).eval()
    restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    inputs = torch.randn(1, 3, 3, 16, 16)
    with torch.no_grad():
        a, b = (m.forward_ar(inputs[:, :, :2], inputs[:, :, 2:]) for m in (model, restored))
    torch.testing.assert_close(a["rain"], b["rain"], rtol=0, atol=0)


@pytest.mark.parametrize("scale", [0.0, 1.0, float("nan")])
def test_no_double_zero_initialization(scale: float) -> None:
    with pytest.raises(ValueError):
        PreservedPointwiseConditioner(dim=8, heads=2, modality_channels=(1, 1, 1), patch_size=1,
                                     frame_patch_size=1, correction_scale_init=scale)
