import pytest
import torch

from src.rain_local_fusion.fusion import RainLocalResidual
from src.rain_local_fusion.model import RainLocalFusionModel
from src.rain_priority.model import RainPriorityModel


@pytest.fixture(autouse=True)
def single_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def settings() -> dict[str, object]:
    return dict(
        input_size=8,
        patch_size=2,
        stem_channels=8,
        dim=16,
        depth=1,
        num_heads=2,
        decoder_base_channels=8,
        decoder_type="3d",
        decoder_k_size=3,
        decoder_pad_mode="zeros",
        stem_pad_mode="zeros",
        max_frames=8,
        spatial_modality_gate_enabled=False,
        activation_checkpoint=True,
        dropout=0.0,
        drop_path=0.0,
    )


def metadata(t: int) -> dict[str, torch.Tensor]:
    return dict(gate_source=torch.ones(1, 3, t, dtype=torch.long), gate_steps=torch.arange(t)[None])


@pytest.mark.parametrize("kernel", [1, 3])
def test_operator_neutral_and_missing_auxiliary_mask(kernel: int) -> None:
    branch = RainLocalResidual(4, 6, hidden_channels=3, kernel_size=kernel)
    q = torch.randn(2, 4, 3, 5, 5)
    radar, satellite = torch.randn(2, 6, 3, 5, 5), torch.randn(2, 6, 3, 5, 5)
    available = torch.ones(2, 2, 3, dtype=torch.bool)
    assert torch.count_nonzero(branch(q, radar, satellite, available)) == 0
    torch.nn.init.normal_(branch.output.weight, std=0.1)
    torch.nn.init.constant_(branch.output.bias, 0.5)
    available[:, 0] = False
    before = branch(q, radar, satellite, available)
    after = branch(q, torch.full_like(radar, float("nan")), satellite, available)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    available[:] = False
    assert torch.count_nonzero(branch(q, radar, satellite, available)) == 0


def test_operator_has_local_support_without_temporal_or_wraparound_mixing() -> None:
    branch = RainLocalResidual(1, 1, hidden_channels=1, kernel_size=3)
    with torch.no_grad():
        for name, param in branch.named_parameters():
            param.fill_(0 if name.endswith("bias") else 1)
    q = torch.zeros(1, 1, 3, 7, 7)
    radar = torch.zeros_like(q)
    radar[:, :, 1, 3, 3] = 1
    out = branch(q, radar, q, torch.ones(1, 2, 3, dtype=torch.bool))
    assert torch.count_nonzero(out[:, :, 0]) == torch.count_nonzero(out[:, :, 2]) == 0
    assert torch.count_nonzero(out[:, :, 1]) == 9
    radar.zero_()
    radar[:, :, 1, 0, 0] = 1
    out = branch(q, radar, q, torch.ones(1, 2, 3, dtype=torch.bool))
    assert torch.count_nonzero(out[..., -1, :]) == torch.count_nonzero(out[..., :, -1]) == 0


@pytest.mark.parametrize("enabled,kernel", [(False, 3), (True, 1), (True, 3)])
@pytest.mark.parametrize("frames,predict", [(8, 7), (5, 1)])
def test_initialization_preserves_baseline_parameters_rng_and_predictions(
    enabled: bool, kernel: int, frames: int, predict: int
) -> None:
    torch.manual_seed(42)
    baseline = RainPriorityModel(**settings()).eval()
    rng = torch.get_rng_state()
    torch.manual_seed(42)
    model = RainLocalFusionModel(
        **settings(),
        local_fusion_enabled=enabled,
        local_fusion_kernel_size=kernel,
        local_fusion_hidden_channels=4,
    ).eval()
    assert torch.equal(torch.get_rng_state(), rng)
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    x = torch.randn(1, 12, frames, 8, 8)
    with torch.no_grad():
        expected = baseline(x, predict_frames=predict, **metadata(frames))
        actual = model(x, predict_frames=predict, **metadata(frames))
        rain_only = model(x, predict_frames=predict, return_modality_dict=False, **metadata(frames))
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    torch.testing.assert_close(rain_only, expected["rain"], rtol=0, atol=0)
    assert type(model.rain_decoder) is type(baseline.rain_decoder)


def test_only_rain_head_changes_each_decoder_runs_once_and_branch_receives_rain_gradients() -> None:
    model = RainLocalFusionModel(**settings(), local_fusion_hidden_channels=4).train()
    x = torch.randn(1, 12, 8, 8, 8)
    baseline = model(x, predict_frames=7, **metadata(8))
    with torch.no_grad():
        model.local_fusion.output.weight.normal_(std=0.03)
    calls = []
    handles = [
        head.register_forward_hook(lambda module, args, out, name=name: calls.append(name))
        for name, head in (
            ("rain", model.rain_decoder),
            ("radar", model.radar_decoder),
            ("satellite", model.satellite_decoder),
        )
    ]
    actual, stats = model(x, predict_frames=7, return_gate_stats=True, **metadata(8))
    for handle in handles:
        handle.remove()
    assert sorted(calls) == ["radar", "rain", "satellite"]
    for name in ("radar", "satellite"):
        torch.testing.assert_close(actual[name], baseline[name], rtol=0, atol=0)
    assert not torch.equal(actual["rain"], baseline["rain"])
    params = list(model.local_fusion.parameters())
    aux = actual["radar"].square().mean() + actual["satellite"].square().mean()
    grads = torch.autograd.grad(aux, params, allow_unused=True, retain_graph=True)
    assert all(grad is None for grad in grads)
    grads = torch.autograd.grad(actual["rain"].square().mean(), params)
    assert all(torch.isfinite(grad).all() and torch.count_nonzero(grad) for grad in grads)
    assert stats["fusion/delta_rms"] > 0
    assert all(not value.requires_grad for value in stats.values())


@pytest.mark.parametrize("frames,predict", [(8, 7), (5, 1), (8, 1)])
def test_branch_uses_matching_seed_features_and_sources(frames: int, predict: int) -> None:
    model = RainLocalFusionModel(**settings(), local_fusion_hidden_channels=4).eval()
    x = torch.randn(1, 12, frames, 8, 8)
    source = metadata(frames)
    source["gate_source"][:, 0, -1] = 0
    received = []
    handle = model.local_fusion.register_forward_pre_hook(lambda module, args: received.append(args))
    model(x, predict_frames=predict, **source)
    handle.remove()
    query, radar, satellite, available = received[0]
    assert query.shape[2] == predict
    torch.testing.assert_close(available, source["gate_source"][:, :2, -predict:] != 0)
    seed = x[:, :, -predict:]
    expected = torch.nn.functional.conv3d(seed[:, :1], model.patch_embed.weight[:, :1], stride=model.patch_embed.stride)
    expected = torch.where(available[:, :1, :, None, None], expected, 0)
    torch.testing.assert_close(radar, expected)
    expected = torch.nn.functional.conv3d(
        seed[:, 1:11], model.patch_embed.weight[:, 1:11], stride=model.patch_embed.stride
    )
    torch.testing.assert_close(satellite, expected)


def test_checkpoint_and_forward_ar_roundtrip() -> None:
    model = RainLocalFusionModel(**settings(), local_fusion_hidden_channels=4).eval()
    torch.nn.init.normal_(model.local_fusion.output.weight, std=0.01)
    restored = RainLocalFusionModel(**settings(), local_fusion_hidden_channels=4).eval()
    restored.load_state_dict(model.state_dict(), strict=True)
    x = torch.randn(1, 12, 4, 8, 8)
    with torch.no_grad():
        first = model.forward_ar(x, x[:, :, -1:], **metadata(5))
        second = restored.forward_ar(x, x[:, :, -1:], **metadata(5))
    for name in first:
        torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
    other = RainLocalFusionModel(**settings(), local_fusion_hidden_channels=4, local_fusion_kernel_size=1)
    with pytest.raises(RuntimeError):
        other.load_state_dict(model.state_dict(), strict=True)


def test_neutral_branch_matches_shared_gradients_before_global_clipping() -> None:
    torch.manual_seed(19)
    baseline = RainPriorityModel(**settings()).train()
    torch.manual_seed(19)
    model = RainLocalFusionModel(**settings(), local_fusion_hidden_channels=4).train()
    x = torch.randn(1, 12, 8, 8, 8)
    expected = baseline(x, predict_frames=7, **metadata(8))
    actual = model(x, predict_frames=7, **metadata(8))
    sum(value.square().mean() for value in expected.values()).backward()
    sum(value.square().mean() for value in actual.values()).backward()
    actual_params = dict(model.named_parameters())
    differences, references = [], []
    for name, parameter in baseline.named_parameters():
        if parameter.grad is not None:
            grad = actual_params[name].grad
            torch.testing.assert_close(grad, parameter.grad, rtol=1e-5, atol=1e-7)
            differences.append((grad - parameter.grad).reshape(-1))
            references.append(parameter.grad.reshape(-1))
    assert torch.cat(differences).norm() / torch.cat(references).norm() < 1e-6
    assert torch.count_nonzero(model.local_fusion.output.weight.grad)


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
def test_neutral_bf16_preserves_decoder_input_layout(device: str) -> None:
    torch.manual_seed(42)
    baseline = RainPriorityModel(**settings()).to(device).eval()
    torch.manual_seed(42)
    model = RainLocalFusionModel(**settings(), local_fusion_hidden_channels=4).to(device).eval()
    x = torch.randn(2, 12, 8, 8, 8, device=device)
    source = torch.ones(2, 3, 8, device=device)
    layouts = []
    handles = [
        net.rain_decoder.register_forward_pre_hook(lambda module, args: layouts.append(args[0].stride()))
        for net in (baseline, model)
    ]
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16):
        expected = baseline(x, predict_frames=7, gate_source=source)
        actual = model(x, predict_frames=7, gate_source=source)
    for handle in handles:
        handle.remove()
    assert layouts[0] == layouts[1]
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_unsupported_settings_and_missing_source_fail_explicitly() -> None:
    for override in (
        {"spatial_modality_gate_enabled": True},
        {"decoder_type": "2d"},
        {"decoder_condition_mode": "film"},
    ):
        with pytest.raises(ValueError):
            RainLocalFusionModel(**(settings() | override))
    with pytest.raises(ValueError):
        RainLocalResidual(4, 4, kernel_size=5)
    model = RainLocalFusionModel(**settings())
    x = torch.randn(1, 12, 5, 8, 8)
    with pytest.raises(ValueError, match="gate_source"):
        model(x)
    with pytest.raises(ValueError, match="gate_source"):
        model(x, gate_source=torch.full((1, 3, 5), 3))
