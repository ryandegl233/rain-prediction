import torch
from torch import nn

from src.networks.time_series.causal_patch_transformer_ccc_init_fixed import RainCausalPatchTransformerCCCInitFixed
from src.trainer.rain_trainer_ccc_frozen_ablation import freeze_pretrained_parameters, frozen_digest


def make_model(enabled: bool) -> RainCausalPatchTransformerCCCInitFixed:
    return RainCausalPatchTransformerCCCInitFixed(
        in_channels=3, radar_out_channels=1, satellite_out_channels=1, rain_out_channels=1,
        input_size=16, patch_size=4, stem_channels=16, dim=16, depth=1, num_heads=4,
        decoder_base_channels=16, dropout=0, drop_path=0,
        cross_modal_adapter_enabled=True, local_window_refiner_enabled=True,
        local_window_size=2, consensus_calibration_enabled=enabled, consensus_calibration_bottleneck=8,
    )


def test_frozen_decoder_passes_rainfall_gradients_into_ccc() -> None:
    torch.manual_seed(2025)
    baseline = make_model(False)
    model = make_model(True)
    missing, unexpected = model.load_state_dict(baseline.state_dict(), strict=False)
    assert missing and all("consensus_calibration" in name for name in missing)
    assert not unexpected
    nn.init.constant_(model.cross_modal_adapter.gate, 0.2)
    nn.init.constant_(model.local_window_refiner.gate, 0.1)
    audit = freeze_pretrained_parameters(model)
    assert audit["parameter_counts"]["frozen"] > 0
    assert audit["parameter_counts"]["ccc"] > 0
    initial = frozen_digest(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
    inputs = torch.randn(1, 3, 3, 16, 16)
    target = torch.randn(1, 1, 1, 16, 16)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        prediction = model.forward_ar(inputs[:, :, :2], inputs[:, :, 2:], predict_frames=1, return_modality_dict=True)
        (prediction["rain"] - target).square().mean().backward()
        assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        for conditional in model.cross_modal_adapter.consensus_calibration.conditional:
            assert conditional.net[-1].weight.grad.count_nonzero() > 0
            if step == 1:
                assert conditional.net[1].weight.grad.count_nonzero() > 0
        optimizer.step()
    assert frozen_digest(model) == initial


def test_both_arms_have_equal_initial_forecasts_and_shared_scope() -> None:
    torch.manual_seed(2025)
    a, b = make_model(False).eval(), make_model(True).eval()
    b.load_state_dict(a.state_dict(), strict=False)
    for model in (a, b):
        nn.init.constant_(model.cross_modal_adapter.gate, 0.2)
        nn.init.constant_(model.local_window_refiner.gate, 0.1)
    scope_a, scope_b = freeze_pretrained_parameters(a), freeze_pretrained_parameters(b)
    assert set(scope_a["trainable_names"]) == {
        name for name in scope_b["trainable_names"] if ".consensus_calibration." not in name
    }
    assert frozen_digest(a) == frozen_digest(b)
    inputs = torch.randn(1, 3, 3, 16, 16)
    with torch.no_grad():
        first = a.forward_ar(inputs[:, :, :2], inputs[:, :, 2:], predict_frames=1, return_modality_dict=True)
        second = b.forward_ar(inputs[:, :, :2], inputs[:, :, 2:], predict_frames=1, return_modality_dict=True)
    for name in first:
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
