import copy

import pytest
import torch
from src.networks.time_series.causal_patch_transformer_next_frame import _CrossModalConditioner
from src.networks.time_series.causal_patch_transformer_ccc_init_fixed import RainCausalPatchTransformerCCCInitFixed
from src.networks.time_series.causal_patch_transformer_neighborhood import (
    NeighborhoodCrossModalConditioner, RainCausalPatchTransformerNeighborhood, neighborhood_indices,
)


def make_adapter(radius: int, chunk: int = 3, recompute: bool = False) -> NeighborhoodCrossModalConditioner:
    layer = NeighborhoodCrossModalConditioner(
        dim=8, heads=2, modality_channels=(1, 1, 1), patch_size=1, frame_patch_size=1,
        radius=radius, query_chunk_size=chunk, recompute=recompute,
    )
    layer.gate.data.fill_(0.2)
    return layer


def test_radius_zero_matches_original_outputs_and_gradients() -> None:
    torch.manual_seed(3)
    original = _CrossModalConditioner(dim=8, heads=2, modality_channels=(1, 1, 1), patch_size=1, frame_patch_size=1)
    original.gate.data.fill_(0.2)
    candidate = make_adapter(0)
    candidate.load_state_dict(original.state_dict(), strict=True)
    q, context = torch.randn(2, 2, 12, 8), torch.randn(2, 3, 3, 3, 4)
    weight = torch.randn_like(q)
    results = []
    for layer in (original, candidate):
        query, history = q.clone().requires_grad_(), context.clone().requires_grad_()
        output = layer(query, history)
        (output * weight).sum().backward()
        results.append((output, query.grad, history.grad))
    for a, b in zip(*results, strict=True):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    for a, b in zip(original.parameters(), candidate.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, atol=3e-6, rtol=2e-5)


def test_corner_mask_no_wraparound() -> None:
    indices, valid = neighborhood_indices(3, 4, 1, torch.device("cpu"))
    assert indices[0][valid[0]].tolist() == [0, 1, 4, 5]
    assert indices[-1][valid[-1]].tolist() == [6, 7, 10, 11]
    assert valid.sum(dim=1).tolist() == [4, 6, 6, 4, 6, 9, 9, 6, 4, 6, 6, 4]


def test_neighbor_influence_excludes_distant_pixels() -> None:
    torch.manual_seed(5)
    layer = make_adapter(1)
    q = torch.randn(1, 1, 25, 8)
    context = torch.randn(1, 3, 2, 5, 5, requires_grad=True)
    layer(q, context)[0, 0, 0].square().sum().backward()
    assert context.grad[:, :, :, :2, :2].abs().sum() > 0
    assert context.grad[:, :, :, 2:, :].count_nonzero() == 0
    assert context.grad[:, :, :, :, 2:].count_nonzero() == 0


def test_chunking_recomputation_matches_outputs_and_backward() -> None:
    torch.manual_seed(7)
    plain = make_adapter(1, 100, False)
    chunked = copy.deepcopy(plain)
    chunked.query_chunk_size, chunked.recompute = 3, True
    q, context = torch.randn(2, 3, 12, 8), torch.randn(2, 3, 4, 3, 4)
    outputs = []
    for layer in (plain, chunked):
        output = layer(q, context)
        output.square().mean().backward()
        outputs.append(output)
    torch.testing.assert_close(*outputs)
    for a, b in zip(plain.parameters(), chunked.parameters(), strict=True):
        assert b.grad is not None and torch.isfinite(b.grad).all()
        torch.testing.assert_close(a.grad, b.grad, atol=1e-7, rtol=2e-5)


def test_chunk_attention_matches_explicit_masked_softmax() -> None:
    layer = make_adapter(1)
    q = torch.randn(1, 6, 2, 2, 4)
    k, v = torch.randn(1, 6, 9, 2, 4), torch.randn(1, 6, 9, 2, 4)
    indices, valid = neighborhood_indices(2, 3, 1, q.device)
    actual = layer._attend_chunk(q, k, v, indices, valid)
    expected = []
    for position in range(6):
        key = k[0, indices[position][valid[position]]].flatten(0, 1).permute(1, 0, 2)
        value = v[0, indices[position][valid[position]]].flatten(0, 1).permute(1, 0, 2)
        query = q[0, position].permute(1, 0, 2)
        probabilities = (query @ key.transpose(-1, -2) / 2).softmax(-1)
        expected.append((probabilities @ value).permute(1, 0, 2).flatten(-2))
    torch.testing.assert_close(actual, torch.stack(expected)[None], atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("radius,chunk", [(2, 32), (1, 0)])
def test_invalid_settings(radius: int, chunk: int) -> None:
    with pytest.raises(ValueError):
        make_adapter(radius, chunk)


def test_full_model_loads_baseline_and_cross_reads_only_context() -> None:
    kwargs = dict(
        in_channels=3, radar_out_channels=1, satellite_out_channels=1, rain_out_channels=1,
        input_size=16, patch_size=4, stem_channels=16, dim=16, depth=1, num_heads=4,
        decoder_base_channels=16, dropout=0, drop_path=0,
        cross_modal_adapter_enabled=True, consensus_calibration_enabled=False,
    )
    original = RainCausalPatchTransformerCCCInitFixed(**kwargs).eval()
    candidate = RainCausalPatchTransformerNeighborhood(neighborhood_radius=0, **kwargs).eval()
    candidate.load_state_dict(original.state_dict(), strict=True)
    original.cross_modal_adapter.gate.data.fill_(0.2)
    candidate.cross_modal_adapter.gate.data.fill_(0.2)
    inputs = torch.randn(1, 3, 4, 16, 16)
    captured = []

    def capture(module: torch.nn.Module, args: tuple, keywords: dict) -> None:
        captured.append(keywords["context_x"].detach().clone())

    hook = candidate.cross_modal_adapter.register_forward_pre_hook(capture, with_kwargs=True)
    with torch.no_grad():
        a = original.forward_ar(inputs[:, :, :2], inputs[:, :, 2:], predict_frames=2, strict_target_isolation=True)
        b = candidate.forward_ar(inputs[:, :, :2], inputs[:, :, 2:], predict_frames=2, strict_target_isolation=True)
    hook.remove()
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=2e-6, rtol=2e-5)
    assert len(captured) == 1
    torch.testing.assert_close(captured[0], inputs[:, :, :2], atol=0, rtol=0)
