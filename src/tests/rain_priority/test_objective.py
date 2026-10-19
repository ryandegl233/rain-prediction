import copy

import pytest
import torch
from accelerate import Accelerator

from src.rain_priority.controller import HorizonController
from src.rain_priority.loss import modality_losses
from src.rain_priority.optimization import backward_rain_priority


def test_sequence_segments_and_horizons_have_hand_computed_weights() -> None:
    targets = {m: torch.zeros(1, c, 7, 1, 1) for m, c in (("rain", 1), ("radar", 1), ("satellite", 10))}
    pred = {m: torch.tensor([1, 1, 1, 2, 2, 2, 2.0]).reshape(1, 1, 7, 1, 1).expand_as(v) for m, v in targets.items()}
    future = {m: v[:, :, :4] for m, v in targets.items()}
    rolled = {m: torch.tensor([1, 2, 3, 4.0]).reshape(1, 1, 4, 1, 1).expand_as(v) for m, v in future.items()}
    losses, errors = modality_losses(pred, targets, rolled, future, torch.tensor([0.1, 0.2, 0.3, 0.4]), 0.2)
    assert losses["rain"].item() == pytest.approx(4.5)
    assert losses["radar"].item() == pytest.approx(4.0)
    assert losses["satellite"].item() == pytest.approx(4.0)
    torch.testing.assert_close(errors, torch.tensor([1.0, 4.0, 9.0, 16.0]))


def test_absolute_rain_loss_corrects_biased_predicted_history() -> None:
    delta = torch.tensor(-0.1, requires_grad=True)
    pred = {m: torch.zeros(1, 1, 7, 1, 1) for m in ("rain", "radar", "satellite")}
    target = {m: torch.zeros_like(v) for m, v in pred.items()}
    rolled = {m: torch.zeros(1, 1, 4, 1, 1) for m in pred}
    rolled["rain"] = (delta + 0.2).expand(1, 1, 4, 1, 1)
    future = {m: torch.zeros_like(v) for m, v in rolled.items()}
    losses, _ = modality_losses(pred, target, rolled, future, torch.full((4,), 0.25), 0.2)
    losses["rain"].backward()
    assert delta.grad.item() > 0


def test_controller_calibrates_late_warmup_and_resumes_mid_window() -> None:
    controller = HorizonController(warmup_steps=4, update_every=2, ema_decay=0.0)
    for step, error in enumerate((100.0, 100.0, 2.0, 2.0), start=1):
        controller.observe(torch.full((4,), error * 3), 3, step)
    torch.testing.assert_close(controller.reference, torch.full((4,), 2.0))
    controller.observe(torch.tensor([1.0, 1.0, 2.0, 2.0]) * 3, 3, 5)
    resumed = HorizonController(warmup_steps=4, update_every=2, ema_decay=0.0)
    resumed.load_state_dict(copy.deepcopy(controller.state_dict()))
    for obj in (controller, resumed):
        obj.observe(torch.tensor([1.0, 1.0, 2.0, 2.0]) * 3, 3, 6)
        torch.testing.assert_close(obj.weights, torch.tensor([5 / 24, 5 / 24, 7 / 24, 7 / 24]))
    torch.testing.assert_close(resumed.weights, controller.weights, rtol=0, atol=0)


def test_controller_uses_element_counts_and_rejects_invalid_statistics() -> None:
    obj = HorizonController(warmup_steps=2, update_every=2)
    obj.observe(torch.ones(4), 1, 1)
    obj.observe(torch.full((4,), 9.0), 3, 2)
    torch.testing.assert_close(obj.reference, torch.full((4,), 2.5))
    saved = copy.deepcopy(obj.state_dict())
    with pytest.raises(ValueError):
        obj.observe(torch.full((4,), float("nan")), 1, 3)
    assert obj.step == saved["step"]
    with pytest.raises(ValueError):
        obj.observe(torch.ones(4), 0, 3)
    with pytest.raises(ValueError):
        obj.observe(torch.ones(4), 1, 4)


def test_zero_reference_remains_finite_and_weights_have_fixed_budget() -> None:
    obj = HorizonController(warmup_steps=2, update_every=2, ema_decay=0.0)
    for step in (1, 2):
        obj.observe(torch.zeros(4), 1, step)
    for step in (3, 4):
        obj.observe(torch.tensor([0.0, 1.0, 1e5, 3.0]), 1, step)
    torch.testing.assert_close(obj.weights, torch.full((4,), 0.25))
    assert torch.isfinite(obj.weights).all()
    assert obj.neutral_reference_count == 4


@pytest.mark.parametrize("auxiliary_weight", [0.1, 100.0])
def test_gate_gradient_ignores_auxiliary_and_preserves_accumulation(auxiliary_weight: float) -> None:
    accelerator = Accelerator(cpu=True, gradient_accumulation_steps=2)
    gate = torch.nn.Parameter(torch.tensor(1.0))
    shared = torch.nn.Parameter(torch.tensor(2.0))
    for x in (1.0, 3.0):
        output = gate * shared * x
        rain = output.square()
        auxiliary = auxiliary_weight * (output - 1).square()
        backward_rain_priority(accelerator, rain + auxiliary, rain, [gate])
    assert gate.grad.item() == pytest.approx(40.0)
    assert shared.grad.item() == pytest.approx(20.0 + 16.0 * auxiliary_weight)


def test_controller_disabled_keeps_uniform_weights() -> None:
    obj = HorizonController(enabled=False, warmup_steps=2, update_every=2)
    for step in range(1, 7):
        obj.observe(torch.tensor([1.0, 1.0, 1.0, 100.0]), 1, step)
    torch.testing.assert_close(obj.weights, torch.full((4,), 0.25))
