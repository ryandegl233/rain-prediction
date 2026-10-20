import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from src.tools.select_sichuan_cases import frame_metrics, load_cases, main, select_frames


def test_metrics_use_raw_threshold_counts_and_handle_empty_events() -> None:
    prediction = np.array([[0.7, 0.7], [0.0, 0.0]])
    truth = np.array([[0.7, 0.0], [0.7, 0.0]])
    result = frame_metrics(prediction, truth)
    assert result["csi@0.5"] == pytest.approx(1 / 3)
    assert result["pod@0.5"] == 0.5
    assert result["far@0.5"] == 0.5
    assert result["mse"] == pytest.approx(0.245)
    assert frame_metrics(np.zeros((2, 2)), np.zeros((2, 2)))["csi@0.5"] is None
    with pytest.raises(ValueError, match="finite"):
        frame_metrics(np.array([[np.nan]]), np.array([[0.0]]))


def test_selection_excludes_empty_events_and_uses_distinct_valid_times() -> None:
    rows = [
        {"case": "a", "valid_time": "a", "csi@0.5": 0.8, "csi@0.3": 0.9, "mse": 0.1},
        {"case": "b", "valid_time": "a", "csi@0.5": 0.9, "csi@0.3": 0.9, "mse": 0.1},
        {"case": "c", "valid_time": "b", "csi@0.5": 0.7, "csi@0.3": 0.9, "mse": 0.1},
        {"case": "d", "valid_time": "c", "csi@0.5": None, "csi@0.3": None, "mse": 0.0},
    ]
    selected = select_frames(rows, 3)
    assert [row["case"] for row in selected] == ["b", "c"]
    assert rows[0]["case"] == "a"
    assert select_frames([rows[-1]], 3) == []


def write_case(path: Path, prediction: np.ndarray) -> Path:
    path.mkdir()
    times = ["2025-08-10T23:30:00", "2025-08-11T00:00:00"]
    history_times = [f"2025-08-10T{t}:00" for t in ("21:30", "22:00", "22:30", "23:00")]
    np.savez_compressed(
        path / "forecast_raw.npz",
        prediction_rain=prediction,
        truth_rain=np.ones((2, 4, 4)),
        history_rain=np.zeros((4, 4, 4)),
        history_times=history_times,
        future_times=times,
    )
    (path / "case.json").write_text(
        json.dumps(
            {
                "step": 80000,
                "checkpoint": "/runs/test_run/checkpoint-00080000/model.safetensors",
                "history_times": history_times,
                "future_times": times,
                "geo_bounds": [97.3, 108.4, 26.1, 34.25],
                "origin": "upper",
            }
        ),
        encoding="utf-8",
    )
    (path / "sichuan_boundary.geojson").write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[98, 27], [107, 27], [107, 33], [98, 33], [98, 27]]],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_comparison_rejects_different_truth(tmp_path: Path) -> None:
    first = write_case(tmp_path / "a", np.ones((2, 4, 4)))
    second = write_case(tmp_path / "b", np.ones((2, 4, 4)))
    with np.load(second / "forecast_raw.npz") as archive:
        data = {key: archive[key] for key in archive.files}
    data["truth_rain"] = np.zeros((2, 4, 4))
    np.savez_compressed(second / "forecast_raw.npz", **data)
    with pytest.raises(ValueError, match="truth"):
        load_cases([first, second])


def test_cli_exports_scored_examples_without_changing_arrays(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = write_case(tmp_path / "case", np.ones((2, 4, 4)))
    raw_path = source / "forecast_raw.npz"
    before = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    output = tmp_path / "selected"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "select",
            "--cases",
            str(source),
            "--output-dir",
            str(output),
            "--top-k",
            "2",
            "--dpi",
            "40",
        ],
    )
    main()
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == before
    report = json.loads((output / "selection.json").read_text(encoding="utf-8"))
    assert report["selection_uses_ground_truth"] is True
    assert len(report["selected"]) == 2
    for name in ("all_frame_metrics.csv", "selected_comparison.png", "selected_comparison.pdf"):
        assert (output / name).stat().st_size > 0
