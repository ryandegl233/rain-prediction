import inspect

import pytest
import torch
from torch import Tensor, nn

from src.baselines_stage3.deep.npm import build_npm


@pytest.fixture
def npm_tiny() -> nn.Module:
    return build_npm(
        {
            "image_size": 16,
            "hidden": 4,
            "spatial_depth": 2,
            "temporal_depth": 2,
            "translator_width": 2,
            "translator_downsamples": 2,
            "translator_residual_blocks": 1,
        }
    )


@pytest.fixture
def observed_time() -> Tensor:
    return torch.tensor([[[120.0, 10.0], [120.0, 10.1667], [120.0, 10.3333], [120.0, 10.5]]])


def test_npm_train_step_updates_both_stages(npm_tiny: nn.Module, observed_time: Tensor) -> None:
    torch.manual_seed(29)
    satellite = torch.randn(1, 10, 4, 16, 16)
    optimizer = torch.optim.Adam(npm_tiny.parameters(), lr=1e-3)
    original_forecast_weight = npm_tiny.forecaster.enc.enc[0].conv.conv.weight.detach().clone()
    original_translator_weight = npm_tiny.translator.model[1].weight.detach().clone()
    result = npm_tiny(satellite, observed_time)
    assert result["rain"].shape == (1, 1, 4, 16, 16)
    assert result["satellite"].shape == (1, 10, 4, 16, 16)
    assert all(torch.isfinite(value).all() for value in result.values())

    rain_target = torch.rand_like(result["rain"]) * 5
    satellite_target = torch.rand_like(result["satellite"])
    loss = 2 * (result["rain"] - rain_target).square().mean()
    loss = loss + (result["satellite"] - satellite_target).square().mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in npm_tiny.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in npm_tiny.parameters())
    optimizer.step()
    assert not torch.equal(original_forecast_weight, npm_tiny.forecaster.enc.enc[0].conv.conv.weight)
    assert not torch.equal(original_translator_weight, npm_tiny.translator.model[1].weight)


def test_npm_rain_uses_predicted_satellite_and_backpropagates(npm_tiny: nn.Module, observed_time: Tensor) -> None:
    captured: list[Tensor] = []

    def capture_translation(module: nn.Module, arguments: tuple[Tensor, ...]) -> None:
        captured.append(arguments[0].detach().clone())

    handle = npm_tiny.translator.register_forward_pre_hook(capture_translation)
    result = npm_tiny(torch.randn(1, 10, 4, 16, 16), observed_time)
    handle.remove()
    torch.testing.assert_close(captured[0], result["satellite"].permute(0, 2, 1, 3, 4).reshape(4, 10, 16, 16))
    result["rain"].square().mean().backward()
    gradient = npm_tiny.forecaster.dec.readout.weight.grad
    assert gradient is not None and bool(gradient.abs().sum() > 0)


def test_npm_only_observed_last_calendar_time_conditions_forecast(npm_tiny: nn.Module, observed_time: Tensor) -> None:
    npm_tiny.eval()
    satellite = torch.randn(1, 10, 4, 16, 16)
    earlier_time = observed_time.clone()
    earlier_time[:, :3, 0] = 10
    with torch.no_grad():
        expected = npm_tiny(satellite, observed_time)
        actual = npm_tiny(satellite, earlier_time)
    torch.testing.assert_close(actual["rain"], expected["rain"])
    assert list(inspect.signature(npm_tiny.forward).parameters) == ["satellite", "time"]


def test_npm_refuses_missing_time_and_historical_rain(npm_tiny: nn.Module, observed_time: Tensor) -> None:
    with pytest.raises(ValueError, match="observed time"):
        npm_tiny(torch.randn(1, 10, 4, 16, 16))
    with pytest.raises(ValueError, match="expects satellite"):
        npm_tiny(torch.randn(1, 11, 4, 16, 16), observed_time)
    bad_time = observed_time.clone()
    bad_time[..., 1] = 24
    with pytest.raises(ValueError, match="calendar time"):
        npm_tiny(torch.randn(1, 10, 4, 16, 16), bad_time)


def test_npm_linear_rain_output_keeps_native_label_range(npm_tiny: nn.Module, observed_time: Tensor) -> None:
    npm_tiny.eval()
    with torch.no_grad():
        npm_tiny.translator.model[-1].weight.zero_()
        npm_tiny.translator.model[-1].bias.fill_(5)
        result = npm_tiny(torch.randn(1, 10, 4, 16, 16), observed_time)
    torch.testing.assert_close(result["rain"], torch.full_like(result["rain"], 5))


def test_npm_default_pipeline_parameter_budget() -> None:
    with torch.device("meta"):
        model = build_npm({})
    counts = {
        name: sum(parameter.numel() for parameter in stage.parameters()) for name, stage in model.named_children()
    }
    assert counts == {"forecaster": 5_306_122, "translator": 40_125_031}
    total = sum(counts.values())
    assert 0.9 * 44_162_136 <= total <= 1.1 * 44_162_136


@pytest.mark.parametrize(
    "options", [{"context_frames": 6}, {"in_channels": 11}, {"image_size": 450}, {"generator_width": 8}]
)
def test_npm_rejects_wrong_comparison_protocol(options: dict) -> None:
    with pytest.raises(ValueError):
        build_npm(options)
