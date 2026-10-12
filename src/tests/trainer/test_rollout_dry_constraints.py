import copy

import pytest
import torch
import yaml

from src.trainer.rollout_dry_constraints import assess_candidate, bounded_auxiliary, dry_rollout_constraint


@pytest.fixture
def config() -> dict:
    with open("src/config/ts_rain_train/rain_trainer_rollout_dry.yaml") as handle:
        return yaml.safe_load(handle)["train"]


def test_dry_and_wet_gradient_directions(config: dict) -> None:
    truth = torch.tensor([0.0, 0.02, 0.6]).view(1, 1, 1, 1, 3).expand(1, 1, 4, 1, 3)
    pred = torch.tensor([0.025, 0.0, 0.2]).view(1, 1, 1, 1, 3).expand_as(truth).clone().requires_grad_()
    loss, terms = dry_rollout_constraint(pred, truth, config["rollout_dry_constraint"])
    loss.backward()
    assert (pred.grad[..., 0] > 0).all()
    assert (pred.grad[..., 1:] < 0).all()
    assert pred.grad[0, 0, 3, 0, 0] / pred.grad[0, 0, 0, 0, 0] == pytest.approx(1.5)
    assert all(torch.isfinite(value) for value in terms.values())


@pytest.mark.parametrize("rain", [0.0, 0.02, 0.6])
def test_perfect_forecast_and_empty_masks(config: dict, rain: float) -> None:
    target = torch.full((1, 1, 4, 2, 2), rain)
    prediction = target.clone().requires_grad_()
    loss, _ = dry_rollout_constraint(prediction, target, config["rollout_dry_constraint"])
    loss.backward()
    assert loss.item() == 0
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))


def test_budget_preserves_auxiliary_gradient(config: dict) -> None:
    base = torch.tensor(0.01, requires_grad=True)
    auxiliary = torch.tensor(100.0, requires_grad=True)
    extra, ramp, scale = bounded_auxiliary(base, auxiliary, 99, config["rollout_dry_constraint"])
    assert extra.item() == pytest.approx(0.002)
    assert ramp == 1 and 0 < scale < 1
    (base + extra).backward()
    assert base.grad.item() == 1
    assert auxiliary.grad.item() > 0


def make_report() -> dict:
    return {
        "deterministic": {"by_lead": [
            {"lead_index": lead, **{f"threshold@{threshold}": {"csi": 0.4, "pod": 0.6, "far": 0.5}
                                     for threshold in ("0.01", "0.1", "0.2", "0.5")}}
            for lead in range(4)]},
        "evolution": {str(lead): {"core_0.5_mae": 0.2, "core_0.5_intensity_ratio": 0.8}
                      for lead in range(1, 5)},
    }


def test_selection_rejects_lower_far_from_missing_rain(config: dict) -> None:
    reference = make_report()
    current = copy.deepcopy(reference)
    for lead in current["deterministic"]["by_lead"]:
        lead["threshold@0.01"]["far"] = 0.4
    assert assess_candidate(current, reference, config["constraint_selection"])["eligible"]
    current["deterministic"]["by_lead"][3]["threshold@0.01"]["pod"] = 0.5
    assert not assess_candidate(current, reference, config["constraint_selection"])["eligible"]
    current["deterministic"]["by_lead"][3]["threshold@0.01"]["pod"] = 0.6
    current["evolution"]["4"]["core_0.5_intensity_ratio"] = 0.6
    assert not assess_candidate(current, reference, config["constraint_selection"])["eligible"]
    assert not assess_candidate(reference, reference, config["constraint_selection"])["eligible"]


def test_selection_rejects_nan(config: dict) -> None:
    reference = make_report()
    current = copy.deepcopy(reference)
    for lead in current["deterministic"]["by_lead"]:
        lead["threshold@0.01"]["far"] = 0.4
    current["deterministic"]["by_lead"][3]["threshold@0.5"]["csi"] = float("nan")
    assert not assess_candidate(current, reference, config["constraint_selection"])["eligible"]
