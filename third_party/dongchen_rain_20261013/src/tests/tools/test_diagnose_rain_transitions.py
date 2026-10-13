from pathlib import Path

import numpy as np
import pytest

from src.tools.diagnose_rain_transitions import evaluate_predictions, region_metrics, transition_masks


def test_transition_partition_and_threshold_boundary() -> None:
    previous = np.array([0.0, 0.0, 0.1, 0.1, np.nan])
    target = np.array([0.0, 0.1, 0.0, 0.1, 0.1])
    masks = transition_masks(previous, target, 0.1)
    assert [int(mask.sum()) for mask in masks.values()] == [1, 1, 1, 1]
    np.testing.assert_array_equal(sum(masks.values()), [1, 1, 1, 1, 0])


def test_persistence_exposes_onset_and_cessation_errors() -> None:
    previous = np.array([0.0, 0.0, 0.5, 0.5])
    target = np.array([0.0, 0.5, 0.0, 0.75])
    metrics = region_metrics(previous, target, previous, previous, 0.1)
    assert metrics["onset"]["bias"] == -0.5
    assert metrics["cessation"]["bias"] == 0.5
    assert metrics["persistent_wet"]["target_abs_delta"] == 0.25
    assert metrics["dry"]["mse"] == 0


def test_empty_regions_are_not_reported_as_perfect_predictions() -> None:
    values = np.zeros(3)
    metrics = region_metrics(values, values, values, values, 0.1)
    assert metrics["onset"]["count"] == 0
    assert metrics["onset"]["mae"] is None


def test_invalid_prediction_is_rejected() -> None:
    values = np.zeros(1)
    with pytest.raises(ValueError, match="non-finite"):
        region_metrics(values, values, np.array([np.nan]), values, 0.1)


def test_multistep_deltas_use_prediction_history_and_regions_use_gt(tmp_path: Path) -> None:
    target = np.array([[[[0.5]], [[0.0]]]])
    prediction = np.array([[[[0.2]], [[0.2]]]])
    path = tmp_path / "predictions.npz"
    np.savez(path, target=target, prediction=prediction, anchor=np.zeros((1, 1, 1)))
    metrics = evaluate_predictions(path, [0.1])["0.1"]
    second = metrics["prediction_by_lead"][1]["cessation"]
    assert second["count"] == 1
    assert second["delta_mae"] == 0.5
    assert second["bias"] == 0.2
    assert metrics["fixed_anchor_persistence_by_lead"][1]["cessation"]["mae"] == 0
