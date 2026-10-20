import hashlib
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from src.tools.render_sichuan_case import comparison_figure, main


def test_pair_has_only_frame_time_and_correct_gt_pred_order() -> None:
    truth = np.full((8, 8), 0.2)
    prediction = np.full((8, 8), 0.7)
    fig = comparison_figure(
        truth,
        prediction,
        {"type": "FeatureCollection", "features": []},
        [97.3, 108.4, 26.1, 34.25],
        5,
        "2025-08-11T02:00:00",
    )
    assert [ax.get_title() for ax in fig.axes] == ["GT", "Pred"]
    assert [text.get_text() for text in fig.texts] == ["frame=05 | time=2025-08-11 02:00:00"]
    assert all(not ax.axison for ax in fig.axes)
    np.testing.assert_array_equal(fig.axes[0].images[0].get_array(), truth)
    np.testing.assert_array_equal(fig.axes[1].images[0].get_array(), prediction)
    assert fig.axes[0].images[0].norm is fig.axes[1].images[0].norm
    plt.close(fig)


def test_cli_limits_by_actual_lead_and_preserves_raw_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "case"
    source.mkdir()
    origin = datetime(2025, 8, 10, 23)
    future_times = [(origin + timedelta(minutes=30 * i)).isoformat() for i in range(1, 13)]
    history_times = [(origin - timedelta(minutes=30 * i)).isoformat() for i in range(3, -1, -1)]
    rng = np.random.default_rng(42)
    np.savez_compressed(
        source / "forecast_raw.npz",
        prediction_rain=rng.random((12, 8, 8)),
        truth_rain=rng.random((12, 8, 8)),
        history_rain=rng.random((4, 8, 8)),
        history_times=history_times,
        future_times=future_times,
    )
    (source / "case.json").write_text(
        json.dumps(
            {
                "step": 50000,
                "checkpoint": "/runs/checkpoint-00050000/model.safetensors",
                "history_times": history_times,
                "future_times": future_times,
                "geo_bounds": [97.3, 108.4, 26.1, 34.25],
                "origin": "upper",
            }
        ),
        encoding="utf-8",
    )
    (source / "sichuan_boundary.geojson").write_text(
        '{"type": "FeatureCollection", "features": []}',
        encoding="utf-8",
    )
    before = hashlib.sha256((source / "forecast_raw.npz").read_bytes()).hexdigest()
    output = tmp_path / "rendered"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render",
            "--cases",
            str(source),
            "--output-dir",
            str(output),
            "--max-lead-minutes",
            "180",
            "--dpi",
            "40",
        ],
    )
    main()
    destination = output / "case"
    report = json.loads((destination / "render.json").read_text(encoding="utf-8"))
    assert [record["frame_index"] for record in report["frames"]] == list(range(6))
    assert report["frames"][-1]["valid_time"] == "2025-08-11T02:00:00"
    assert len(list(destination.glob("*.png"))) == 6
    assert len(list(destination.glob("*.pdf"))) == 6
    assert (destination / "frame_05_20250811_0200.png").is_file()
    with Image.open(destination / "comparison.gif") as image:
        assert image.n_frames == 6
    assert hashlib.sha256((source / "forecast_raw.npz").read_bytes()).hexdigest() == before
    assert report["raw_sha256"] == before
    with pytest.raises(FileExistsError, match="empty output"):
        main()
