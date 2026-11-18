import pytest
import torch

from src.residual_diffusion_stage.metrics import (
    DeterministicMetricAccumulator,
    EnsembleMetricAccumulator,
    center_ensemble_residuals,
)


def test_deterministic_metrics_match_pixel_contingency_table() -> None:
    accumulator = DeterministicMetricAccumulator([0.5], lead_times=1, device=torch.device("cpu"))
    prediction = torch.tensor([[[[[0.7, 0.6], [0.2, 0.1]]]]])
    target = torch.tensor([[[[[0.8, 0.1], [0.7, 0.0]]]]])
    accumulator.update(prediction, target)
    event = accumulator.result()["overall"]["threshold@0.5"]
    assert event["csi"] == pytest.approx(1 / 3)
    assert event["pod"] == pytest.approx(0.5)
    assert event["far"] == pytest.approx(0.5)
    assert event["f1"] == pytest.approx(0.5)


def test_centered_ensemble_has_zero_member_mean_and_preserves_variance() -> None:
    residuals = torch.randn(2, 8, 1, 2, 4, 4)
    centered = center_ensemble_residuals(residuals, variance_correction=True)
    torch.testing.assert_close(centered.mean(dim=1), torch.zeros_like(centered[:, 0]), atol=1.0e-6, rtol=0)
    original_variance = residuals.var(dim=1, unbiased=True).mean()
    centered_variance = centered.var(dim=1, unbiased=True).mean()
    assert centered_variance.item() == pytest.approx(original_variance.item(), rel=0.2)


def test_perfect_ensemble_has_zero_crps_and_brier_score() -> None:
    target = torch.tensor([[[[[0.0, 0.2], [0.7, 0.1]]]]])
    ensemble = target[:, None].repeat(1, 4, 1, 1, 1, 1)
    reference = torch.zeros_like(target)
    accumulator = EnsembleMetricAccumulator([0.1, 0.5], 1, torch.device("cpu"), reliability_bins=5)
    accumulator.update(ensemble, target, reference)
    result = accumulator.result()["overall"]
    assert result["crps"] == 0.0
    assert result["threshold@0.1"]["brier"] == 0.0
    assert result["threshold@0.5"]["brier"] == 0.0
