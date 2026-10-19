import pytest
import torch

from src.rain_priority.gate import TemporalRainGate
from src.rain_priority.model import RainPriorityModel
from src.tests.time_series.test_spatial_modality_gate import build_baseline_model


def small_model(**overrides: object) -> RainPriorityModel:
    settings = dict(
        in_channels=12,
        radar_out_channels=1,
        satellite_out_channels=10,
        rain_out_channels=1,
        input_size=16,
        patch_size=4,
        stem_channels=16,
        dim=32,
        depth=1,
        num_heads=4,
        decoder_base_channels=16,
        dropout=0.0,
        drop_path=0.0,
        max_frames=8,
        spatial_modality_gate_enabled=True,
        spatial_modality_gate_hidden_channels=8,
    )
    settings.update(overrides)
    return RainPriorityModel(**settings)


def test_neutral_temporal_gate_keeps_original_3d_predictions() -> None:
    baseline = build_baseline_model().eval()
    model = small_model().eval()
    result = model.load_state_dict(baseline.state_dict(), strict=False)
    assert not result.unexpected_keys
    assert all(k.startswith("spatial_modality_gate.") for k in result.missing_keys)
    x = torch.randn(1, 12, 8, 16, 16)
    with torch.no_grad():
        expected = baseline(x, predict_frames=7)
        actual = model(x, predict_frames=7, gate_source=torch.ones(1, 3, 8), gate_steps=torch.arange(8)[None])
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert type(model.rain_decoder) is type(baseline.rain_decoder)


def test_gate_prefix_is_unchanged_by_later_inputs_and_metadata() -> None:
    gate = TemporalRainGate(4, 8)
    torch.nn.init.normal_(gate.net[-1].weight)
    features = [torch.randn(2, 4, 5, 3, 3) for _ in range(3)]
    source = torch.ones(2, 3, 5, dtype=torch.long)
    steps = torch.arange(5).expand(2, -1)
    before = gate(*features, source, steps)
    for z in features:
        z[:, :, 3:] += 10
    source[:, :, 3:] = 2
    after = gate(*features, source, steps)
    for a, b in zip(before, after, strict=True):
        torch.testing.assert_close(a[:, :, :3], b[:, :, :3], rtol=0, atol=0)
    assert not torch.equal(before[0][:, :, 3:], after[0][:, :, 3:])


def test_duplicate_seed_uses_previous_physical_frame_and_gaps_are_invalid() -> None:
    gate = TemporalRainGate(2, 4)
    a = torch.tensor([1.0, -1.0]).reshape(1, 2, 1, 1, 1)
    b = -a
    z = torch.cat([a, b, b, a], dim=2)
    source = torch.ones(1, 3, 4, dtype=torch.long)
    steps = torch.tensor([[-1, 0, 0, 2]])
    features = gate.condition_features(z, z, z, source, steps)
    torch.testing.assert_close(features[:, 6:12, 1], features[:, 6:12, 2])
    assert torch.count_nonzero(features[:, 6:12, 3]) == 0
    assert torch.count_nonzero(features[:, -4:-1, 3]) == 0
    source[:, 0, 0] = 0
    masked = gate.condition_features(z, z, z, source, steps)
    assert torch.count_nonzero(masked[:, 6:8, 2]) == 0


def test_gate_rejects_invalid_source_and_reverse_time_indices() -> None:
    gate = TemporalRainGate(2, 4)
    z = torch.randn(1, 2, 2, 2, 2)
    with pytest.raises(ValueError):
        gate(z, z, z, torch.full((1, 3, 2), 3), torch.tensor([[0, 1]]))
    with pytest.raises(ValueError):
        gate(z, z, z, torch.ones(1, 3, 2), torch.tensor([[1, 0]]))


def test_checkpointed_model_backward_reaches_gate_and_original_weights() -> None:
    model = small_model(activation_checkpoint=True).train()
    output = model(
        torch.randn(1, 12, 8, 16, 16),
        predict_frames=7,
        gate_source=torch.ones(1, 3, 8),
        gate_steps=torch.arange(8)[None],
    )
    loss = output["rain"].float().square().mean()
    grads = torch.autograd.grad(loss, [model.spatial_modality_gate.net[-1].weight, model.patch_embed.weight])
    assert all(torch.isfinite(g).all() and torch.count_nonzero(g) for g in grads)


def test_model_save_reload_and_metadata_reach_forward_ar() -> None:
    model = small_model().eval()
    torch.nn.init.normal_(model.spatial_modality_gate.net[-1].weight, std=0.1)
    restored = small_model().eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    x = torch.randn(1, 12, 4, 16, 16)
    metadata = dict(gate_source=torch.ones(1, 3, 5, dtype=torch.long), gate_steps=torch.tensor([[-3, -2, -1, 0, 0]]))
    with torch.no_grad():
        first, stats = model.forward_ar(x, x[:, :, -1:], return_gate_stats=True, **metadata)
        second = restored.forward_ar(x, x[:, :, -1:], **metadata)
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    assert "gate/radar/mean" in stats
    assert stats["gate/radar/source_truth/count"] > 0
    assert stats["gate/radar/source_generated/count"] == 0
    assert (
        stats["gate/radar/input_wet/count"] + stats["gate/radar/input_dry/count"]
        == stats["gate/radar/source_truth/count"]
    )
    assert all(not value.requires_grad for value in stats.values())


def test_temporal_model_requires_explicit_input_provenance() -> None:
    model = small_model().eval()
    x = torch.randn(1, 12, 4, 16, 16)
    with pytest.raises(ValueError, match="explicit.*gate_source.*gate_steps"):
        model.forward_ar(x, x[:, :, -1:])
    with pytest.raises(ValueError, match="explicit.*gate_source.*gate_steps"):
        model(x, gate_source=torch.ones(1, 3, 4))
