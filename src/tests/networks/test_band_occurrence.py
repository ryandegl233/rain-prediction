from contextlib import nullcontext

from omegaconf import OmegaConf
import pytest
import torch

from src.networks.time_series.causal_patch_transformer_band_occurrence import (
    RainCausalPatchTransformerBandOccurrence,
)
from src.trainer.rain_trainer_ts_next_frame import RainTSNextFrameTrainer


def make_model(max_suppression: float = 0.5) -> RainCausalPatchTransformerBandOccurrence:
    return RainCausalPatchTransformerBandOccurrence(
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
        occurrence_gate_gamma=1.0,
        occurrence_band_lower_mm_min=0.01,
        occurrence_band_upper_mm_min=0.1,
        occurrence_max_suppression=max_suppression,
    )


def test_band_gate_changes_only_weak_rain_and_exports_ungated_rollout(monkeypatch) -> None:
    model = make_model(max_suppression=0.5)
    rain = torch.tensor([[[[[0.005, 0.01, 0.055, 0.099, 0.1, 0.4]]]]])
    probability = torch.full_like(rain, 0.2)
    parent_output = {
        "rain": rain * probability,
        "rain_ungated": rain,
        "occurrence_probability": probability,
    }
    monkeypatch.setattr(
        "src.networks.time_series.causal_patch_transformer_soft_occurrence."
        "RainCausalPatchTransformerSoftOccurrence.forward_ar",
        lambda _self, *args, **kwargs: {key: value.clone() for key, value in parent_output.items()},
    )
    model.set_soft_gate_strength(1.0)
    output = model.forward_ar(None, None)

    unchanged = torch.tensor([True, False, False, False, True, True]).view_as(rain)
    torch.testing.assert_close(output["rain"][unchanged], rain[unchanged], rtol=0, atol=0)
    assert torch.all(output["rain"][~unchanged] < rain[~unchanged])
    torch.testing.assert_close(output["rain_rollout"], rain, rtol=0, atol=0)
    for threshold in (0.1, 0.2, 0.5):
        assert torch.equal(output["rain"] >= threshold, rain >= threshold)


def test_zero_gate_strength_is_exact_identity(monkeypatch) -> None:
    model = make_model()
    rain = torch.linspace(0, 0.2, 16).reshape(1, 1, 1, 4, 4)
    parent_output = {
        "rain": rain,
        "rain_ungated": rain,
        "occurrence_probability": torch.full_like(rain, 0.1),
    }
    monkeypatch.setattr(
        "src.networks.time_series.causal_patch_transformer_soft_occurrence."
        "RainCausalPatchTransformerSoftOccurrence.forward_ar",
        lambda _self, *args, **kwargs: {key: value.clone() for key, value in parent_output.items()},
    )
    model.set_soft_gate_strength(0.0)
    output = model.forward_ar(None, None)
    torch.testing.assert_close(output["rain"], rain, rtol=0, atol=0)


def test_invalid_band_configuration_is_rejected() -> None:
    try:
        make_model(max_suppression=1.1)
    except ValueError as error:
        assert "suppression" in str(error)
    else:
        raise AssertionError("Expected invalid suppression to fail")


def test_rollout_feedback_uses_ungated_rain() -> None:
    class DummyModel:
        frame_patch_size = 1

        def __init__(self) -> None:
            self.rain_history = []

        def forward_ar(self, context_x, target_x, **kwargs):
            self.rain_history.append(float(context_x[:, -1:, -1:].mean()))
            shape = (int(context_x.shape[0]), 1, int(kwargs["predict_frames"]), *context_x.shape[-2:])
            zeros = context_x.new_zeros(shape)
            return {
                "radar": zeros,
                "satellite": zeros,
                "rain": context_x.new_full(shape, 0.05),
                "rain_rollout": context_x.new_full(shape, 0.2),
                "occurrence_logits": context_x.new_full(shape, 0.4),
            }

    class DummyAccelerator:
        def __init__(self, model) -> None:
            self.model = model

        def unwrap_model(self, model):
            return model

        def autocast(self):
            return nullcontext()

    trainer = object.__new__(RainTSNextFrameTrainer)
    trainer.model = DummyModel()
    trainer.accelerator = DummyAccelerator(trainer.model)
    trainer.train_cfg = OmegaConf.create({
        "strict_target_isolation": False,
        "loss": {"rain_residual": {"enabled": False}},
    })
    trainer.radar_c = trainer.satellite_c = trainer.rain_c = 1
    context = torch.zeros(1, 3, 1, 2, 2)
    context[:, -1:] = 0.7

    output = trainer._rollout_predict_with_settings(
        context=context,
        total_future_frames=2,
        mode="frame",
        rollout_block_size=1,
        detach_history=True,
    )

    torch.testing.assert_close(output["rain"], torch.full_like(output["rain"], 0.05))
    torch.testing.assert_close(output["occurrence_logits"], torch.full_like(output["rain"], 0.4))
    assert trainer.model.rain_history == pytest.approx([0.7, 0.2])
