import pytest
import torch

from src.tools.check_rain_learning import forecast_metrics, sequence_mean


def test_forecast_metrics_separate_absolute_and_temporal_error() -> None:
    anchor = torch.tensor([1.0]).reshape(1, 1, 1, 1, 1)
    target = torch.tensor([2.0, 3.0, 4.0]).reshape(1, 1, 3, 1, 1)
    prediction = torch.tensor([1.0, 2.0, 3.0]).reshape(1, 1, 3, 1, 1)
    metrics = forecast_metrics(prediction, target, anchor)
    assert metrics["mse_by_lead"] == [1.0, 1.0, 1.0]
    assert metrics["delta_mse_by_lead"] == [1.0, 0.0, 0.0]
    assert metrics["persistence_mse_by_lead"] == [1.0, 4.0, 9.0]


def test_sequence_loss_uses_segment_weights_not_frame_count() -> None:
    values = torch.tensor([2.0, 6.0, 6.0, 6.0]).reshape(1, 1, 4, 1, 1)
    aux = {"sequence_context_frames": 1, "sequence_context_weight": 1.0, "sequence_future_weight": 1.0}
    assert float(sequence_mean(values, aux)) == 4.0


def test_raw_delta_with_detached_history_has_temporal_value_but_distinct_gradient() -> None:
    previous = torch.tensor(2.0, requires_grad=True)
    raw_delta = torch.tensor(0.5, requires_grad=True)
    current = previous.detach() + raw_delta
    direct_diff = current - previous
    assert float(direct_diff.detach()) == float(raw_delta.detach())
    raw_grad = torch.autograd.grad(raw_delta.square(), previous, allow_unused=True)[0]
    temporal_grad = torch.autograd.grad(direct_diff.square(), previous)[0]
    assert raw_grad is None
    assert float(temporal_grad) == pytest.approx(-1.0)
