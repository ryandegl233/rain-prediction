from contextlib import nullcontext

from omegaconf import OmegaConf
import torch

from src.networks.time_series.causal_patch_transformer_band_occurrence import (
    RainCausalPatchTransformerBandOccurrence,
)
from src.networks.time_series.causal_patch_transformer_explicit_kv import (
    RainCausalPatchTransformerExplicitKV,
    freeze_for_explicit_kv,
)
from src.trainer.rain_trainer_explicit_kv import ExplicitKVTrainer


def model_kwargs() -> dict:
    return {
        "in_channels": 3,
        "radar_out_channels": 1,
        "satellite_out_channels": 1,
        "rain_out_channels": 1,
        "input_size": 16,
        "patch_size": 4,
        "stem_channels": 16,
        "dim": 16,
        "depth": 1,
        "num_heads": 4,
        "decoder_base_channels": 16,
        "dropout": 0.0,
        "drop_path": 0.0,
        "cross_modal_adapter_enabled": True,
        "consensus_calibration_enabled": False,
        "occurrence_gate_gamma": 1.0,
        "occurrence_band_lower_mm_min": 0.01,
        "occurrence_band_upper_mm_min": 0.1,
        "occurrence_max_suppression": 0.15,
    }


def make_explicit() -> RainCausalPatchTransformerExplicitKV:
    return RainCausalPatchTransformerExplicitKV(
        **model_kwargs(),
        correction_kv_heads=2,
        correction_query_chunk_size=5,
        correction_recompute=False,
    )


def test_zero_initialized_explicit_kv_is_exact_baseline() -> None:
    torch.manual_seed(24)
    baseline = RainCausalPatchTransformerBandOccurrence(**model_kwargs()).eval()
    baseline.set_soft_gate_strength(1.0)
    model = make_explicit().eval()
    missing, unexpected = model.load_state_dict(baseline.state_dict(), strict=False)
    assert missing and all(key.startswith("cross_modal_adapter.correction.") for key in missing)
    assert not unexpected
    model.cross_modal_adapter.initialize_correction()
    context = torch.randn(1, 3, 2, 16, 16)
    seed = torch.randn(1, 3, 1, 16, 16)
    with torch.no_grad():
        expected = baseline.forward_ar(context, seed)
        actual = model.forward_ar(
            context,
            seed,
            context_generated_mask=torch.tensor([[False, True]]),
        )
    for key in ("radar", "satellite", "rain", "rain_ungated", "occurrence_logits"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_only_explicit_kv_trains_and_source_identity_receives_gradient() -> None:
    torch.manual_seed(25)
    model = make_explicit()
    freeze_for_explicit_kv(model)
    optimizer = torch.optim.AdamW([value for value in model.parameters() if value.requires_grad], lr=1e-3)
    context = torch.randn(1, 3, 2, 16, 16)
    seed = torch.randn(1, 3, 1, 16, 16)
    target = torch.randn(1, 1, 1, 16, 16)
    correction = model.cross_modal_adapter.correction
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        prediction = model.forward_ar(
            context,
            seed,
            context_generated_mask=torch.tensor([[False, True]]),
        )["rain"]
        (prediction - target).square().mean().backward()
        assert correction.output.weight.grad.count_nonzero() > 0
        if step == 1:
            assert correction.source_embedding.grad.count_nonzero() > 0
            assert correction.modality_embedding.grad.count_nonzero() > 0
            assert correction.time_bias.grad.count_nonzero() > 0
            assert correction.spatial_bias.grad.count_nonzero() > 0
        assert all(value.grad is None for value in model.parameters() if not value.requires_grad)
        optimizer.step()


def test_rollout_marks_generated_history_without_using_gated_rain() -> None:
    class DummyModel:
        def __init__(self) -> None:
            self.masks = []

        def forward_ar(self, context_x, target_x, context_generated_mask, **kwargs):
            self.masks.append(context_generated_mask.clone())
            shape = (context_x.shape[0], 1, 1, *context_x.shape[-2:])
            zeros = context_x.new_zeros(shape)
            return {
                "radar": zeros,
                "satellite": zeros,
                "rain": context_x.new_full(shape, 0.02),
                "rain_rollout": context_x.new_full(shape, 0.08),
                "rain_delta": zeros,
                "occurrence_logits": zeros,
            }

    class DummyAccelerator:
        def autocast(self):
            return nullcontext()

    trainer = object.__new__(ExplicitKVTrainer)
    trainer.model = DummyModel()
    trainer.accelerator = DummyAccelerator()
    trainer.train_cfg = OmegaConf.create({"strict_target_isolation": False})
    trainer.radar_c = trainer.satellite_c = trainer.rain_c = 1
    context = torch.zeros(1, 3, 2, 4, 4)
    output = trainer._rollout_predict_with_settings(
        context=context,
        total_future_frames=3,
        mode="frame",
        rollout_block_size=1,
        detach_history=True,
        future_modalities={
            "radar": torch.randn(1, 1, 3, 4, 4),
            "satellite": torch.randn(1, 1, 3, 4, 4),
        },
        future_rain=torch.randn(1, 1, 3, 4, 4),
    )
    assert [mask.tolist() for mask in trainer.model.masks] == [
        [[False, False]],
        [[False, False, True]],
        [[False, False, True, True]],
    ]
    torch.testing.assert_close(output["rain"], torch.full_like(output["rain"], 0.02))
