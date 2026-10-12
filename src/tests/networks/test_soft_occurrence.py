import pytest
import torch

from src.networks.time_series.causal_patch_transformer_ccc_init_fixed import RainCausalPatchTransformerCCCInitFixed
from src.networks.time_series.causal_patch_transformer_soft_occurrence import (
    RainCausalPatchTransformerSoftOccurrence,
    freeze_for_occurrence_training,
)
from src.trainer.soft_occurrence_objective import resolve_gate_strength, soft_occurrence_loss


def make_model() -> RainCausalPatchTransformerSoftOccurrence:
    return RainCausalPatchTransformerSoftOccurrence(
        in_channels=3,
        radar_out_channels=1,
        satellite_out_channels=1,
        rain_out_channels=1,
        input_size=16,
        patch_size=4,
        stem_channels=16,
        dim=16,
        depth=1,
        num_heads=4,
        decoder_base_channels=16,
        dropout=0,
        drop_path=0,
        cross_modal_adapter_enabled=True,
        consensus_calibration_enabled=False,
        occurrence_gate_gamma=2.0,
    )


def test_zero_strength_exactly_preserves_residual_prediction() -> None:
    torch.manual_seed(7)
    baseline = RainCausalPatchTransformerCCCInitFixed(
        in_channels=3,
        radar_out_channels=1,
        satellite_out_channels=1,
        rain_out_channels=1,
        input_size=16,
        patch_size=4,
        stem_channels=16,
        dim=16,
        depth=1,
        num_heads=4,
        decoder_base_channels=16,
        dropout=0,
        drop_path=0,
        cross_modal_adapter_enabled=True,
        consensus_calibration_enabled=False,
    ).eval()
    model = make_model().eval()
    missing, unexpected = model.load_state_dict(baseline.state_dict(), strict=False)
    assert missing and all(key.startswith("occurrence_head.") or key == "soft_gate_strength" for key in missing)
    assert not unexpected
    context = torch.randn(1, 3, 2, 16, 16)
    seed = torch.randn(1, 3, 1, 16, 16)
    with torch.no_grad():
        old = baseline.forward_ar(context, seed)["rain"] + seed[:, -1:]
        new = model.forward_ar(context, seed)["rain"]
    torch.testing.assert_close(old, new, rtol=0, atol=0)


def test_full_soft_gate_uses_probability_power() -> None:
    model = make_model().eval()
    model.occurrence_head.projection.bias.data.fill_(torch.logit(torch.tensor(0.25)))
    model.set_soft_gate_strength(1.0)
    inputs = torch.randn(1, 3, 3, 16, 16)
    with torch.no_grad():
        output = model.forward_ar(inputs[:, :, :2], inputs[:, :, 2:])
    torch.testing.assert_close(output["rain"], output["rain_ungated"] * 0.25**2, rtol=1e-5, atol=1e-6)


def test_only_occurrence_head_is_trainable() -> None:
    model = make_model()
    scope = freeze_for_occurrence_training(model)
    assert scope["parameter_counts"]["occurrence"] > 0
    assert all(name.startswith("occurrence_head.") for name in scope["trainable_names"])


def test_false_positive_weight_changes_only_dry_term_priority() -> None:
    logits = torch.zeros(1, 1, 1, 2, 2)
    target = torch.tensor([[[[[0.0, 0.0], [0.02, 0.2]]]]])
    low, _ = soft_occurrence_loss(
        logits,
        target,
        threshold_mm_min=0.01,
        false_positive_weight=1.0,
        strong_threshold_mm_min=0.1,
        strong_weight=0.0,
        pod_floor=0.0,
        pod_floor_weight=0.0,
    )
    high_logits = logits.clone()
    high_logits[..., :1, :] = 1.0
    low_weight, _ = soft_occurrence_loss(
        high_logits,
        target,
        threshold_mm_min=0.01,
        false_positive_weight=1.0,
        strong_threshold_mm_min=0.1,
        strong_weight=0.0,
        pod_floor=0.0,
        pod_floor_weight=0.0,
    )
    high_weight, _ = soft_occurrence_loss(
        high_logits,
        target,
        threshold_mm_min=0.01,
        false_positive_weight=4.0,
        strong_threshold_mm_min=0.1,
        strong_weight=0.0,
        pod_floor=0.0,
        pod_floor_weight=0.0,
    )
    assert torch.isfinite(low)
    assert high_weight > low_weight


@pytest.mark.parametrize("step, expected", [(0, 0.0), (50, 0.0), (75, 0.5), (100, 1.0), (200, 1.0)])
def test_gate_schedule(step: int, expected: float) -> None:
    assert resolve_gate_strength(step, 50, 50) == expected
