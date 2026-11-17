import json
from pathlib import Path

import pytest
import torch

from src.baselines.classical import Persistence
from src.baselines.evaluation import evaluate_forecasts, write_report


def samples() -> dict[str, torch.Tensor]:
    result = {
        f"{name}_{period}": torch.zeros(3, channels, 4, 4, 4)
        for name, channels in (("radar", 1), ("satellite", 10), ("rain", 1))
        for period in ("past", "future")
    }
    result["rain_past"][0, :, -1] = 0.4
    result["rain_future"][0, :, :] = 0.4
    result["rain_future"][1, :, :] = 0.4
    stamps = torch.arange(8, dtype=torch.float64)[None] * 1800 + torch.arange(3)[:, None] * 36000
    result["time_past_timestamp"] = stamps[:, :4]
    result["time_future_timestamp"] = stamps[:, 4:]
    return result


def test_metrics_same_for_partial_batch_and_single_batch(tmp_path: Path) -> None:
    data = samples()
    split = [{key: value[start:stop] for key, value in data.items()} for start, stop in ((0, 2), (2, 3))]
    models = {"persistence": Persistence()}
    combined = evaluate_forecasts(models, [data], [("rain",)], [0.1, 0.3, 0.5])
    separated = evaluate_forecasts(models, split, [("rain",)], [0.1, 0.3, 0.5])
    assert combined["window_sha256"] == separated["window_sha256"]
    assert combined["samples"] == separated["samples"] == 3
    metrics = separated["results"]["rain"]["persistence"]["metrics"]
    assert metrics == combined["results"]["rain"]["persistence"]["metrics"]
    assert metrics["all/csi@0.3"] == 0.5
    assert metrics["all/csi@0.5"] is None
    write_report(separated, tmp_path)
    assert json.loads((tmp_path / "metrics.json").read_text()) == separated
    assert (tmp_path / "metrics.csv").is_file()
    assert (tmp_path / "windows.json").is_file()


def test_no_rain_combinations_are_skipped() -> None:
    result = evaluate_forecasts({"persistence": Persistence()}, [samples()], [("satellite",)], [0.3])
    entry = result["results"]["satellite"]["persistence"]
    assert entry["status"] == "skipped"
    assert "rain" in entry["reason"]


def test_empty_evaluation_is_an_error() -> None:
    with pytest.raises(ValueError, match="empty"):
        evaluate_forecasts({"persistence": Persistence()}, [], [("rain",)], [0.3])


def test_future_observations_cannot_change_predictions() -> None:
    data = samples()
    first = evaluate_forecasts({"persistence": Persistence()}, [data], [("rain",)], [0.3])
    data["radar_future"].fill_(float("nan"))
    data["satellite_future"].fill_(float("nan"))
    second = evaluate_forecasts({"persistence": Persistence()}, [data], [("rain",)], [0.3])
    assert first["results"] == second["results"]
