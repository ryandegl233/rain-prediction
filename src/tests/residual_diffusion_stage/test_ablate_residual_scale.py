import pytest
import torch

from src.residual_diffusion_stage.ablate_residual_scale import ScaleMetricAccumulator, summarize_ablation


def test_scale_metric_accumulator_computes_expected_event_metrics() -> None:
    accumulator = ScaleMetricAccumulator([0.0], [0.5], lead_times=1, device=torch.device("cpu"))
    prediction = torch.tensor([[[[[0.7, 0.6], [0.2, 0.1]]]]])
    target = torch.tensor([[[[[0.8, 0.1], [0.7, 0.0]]]]])
    accumulator.update(0, prediction, target)
    overall = accumulator.result()[0]["overall"]
    event = overall["threshold@0.5"]
    assert event["csi"] == 1 / 3
    assert event["pod"] == 0.5
    assert event["far"] == 0.5
    assert overall["mae"] == pytest.approx(0.3)


def test_scale_metric_accumulator_keeps_lead_times_separate() -> None:
    accumulator = ScaleMetricAccumulator([0.0], [0.5], lead_times=2, device=torch.device("cpu"))
    prediction = torch.tensor([[[[[0.0]], [[1.0]]]]])
    target = torch.tensor([[[[[0.0]], [[0.0]]]]])
    accumulator.update(0, prediction, target)
    by_lead = accumulator.result()[0]["by_lead"]
    assert by_lead[0]["mae"] == 0.0
    assert by_lead[1]["mae"] == 1.0


def test_summary_rejects_csi_gain_when_mae_is_worse_than_coarse() -> None:
    rows = [
        {
            "scale": 0.0,
            "overall": {"mae": 0.10, "rmse": 0.2, "bias": 0.01, "mean_csi": 0.40, "threshold@0.1": _event(0.4)},
        },
        {
            "scale": 0.2,
            "overall": {"mae": 0.09, "rmse": 0.18, "bias": 0.00, "mean_csi": 0.42, "threshold@0.1": _event(0.42)},
        },
        {
            "scale": 1.0,
            "overall": {"mae": 0.20, "rmse": 0.3, "bias": 0.03, "mean_csi": 0.70, "threshold@0.1": _event(0.70)},
        },
    ]
    summary = summarize_ablation(rows, [0.1])
    assert summary["recommended_scale"] == 0.2
    assert summary["best_scale_by_metric"]["mean_csi"] == 1.0


def _event(csi: float) -> dict[str, float]:
    return {"csi": csi, "pod": csi, "far": 1.0 - csi, "hss": csi}
