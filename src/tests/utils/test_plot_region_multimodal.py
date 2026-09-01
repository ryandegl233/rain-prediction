import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.utils.visualization.plot_region_multimodal import (
    cap_rain_for_visualization,
    class_map_to_visual_rain,
    polish_pred_rain_for_visualization,
    save_next_frame_input_panels,
    save_region_multimodal_panels,
)


def test_cap_rain_for_visualization_keeps_rain_under_orange_threshold() -> None:
    rain = np.array([[0.0, 0.2], [1.0, 3.0]], dtype=np.float32)

    capped = cap_rain_for_visualization(rain)

    assert float(capped.max()) < 1.0
    assert float(capped[0, 1]) == float(rain[0, 1])


def test_class_map_to_visual_rain_uses_yellow_as_max_visible_class() -> None:
    class_map = np.array([[0, 1, 2], [3, 4, 8]], dtype=np.int64)

    visual_rain = class_map_to_visual_rain(class_map)

    assert float(visual_rain.max()) < 1.0
    assert float(visual_rain[1, 1]) == 0.8
    assert float(visual_rain[1, 2]) == 0.8


def test_polish_pred_rain_for_visualization_contracts_light_rain_and_keeps_heavy_core() -> None:
    pred = np.zeros((24, 24), dtype=np.float32)
    pred[2:6, 2:6] = 0.02
    pred[10:14, 10:14] = 0.2
    pred[11:13, 11:13] = 0.8
    pred[15, 15] = 0.03

    polished = polish_pred_rain_for_visualization(pred)

    assert np.count_nonzero((polished > 0.0) & (polished < 0.1)) < np.count_nonzero((pred > 0.0) & (pred < 0.1))
    assert float(polished[12, 12]) > 0.0
    assert float(polished.max()) < 1.0


def test_save_region_multimodal_panels_with_local_geojson(tmp_path: Path) -> None:
    boundary = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [0.0, 0.0],
                            [1.0, 0.0],
                            [1.0, 1.0],
                            [0.0, 1.0],
                            [0.0, 0.0],
                        ]
                    ],
                },
            }
        ],
    }
    boundary_path = tmp_path / "boundary.json"
    boundary_path.write_text(json.dumps(boundary), encoding="utf-8")

    rain = np.zeros((16, 16), dtype=np.float32)
    rain[4:10, 5:12] = 0.8
    pred = np.zeros((16, 16), dtype=np.float32)
    pred[5:11, 6:13] = 0.6
    radar = np.linspace(0.0, 1.0, 16 * 16, dtype=np.float32).reshape(1, 16, 16)
    satellite = np.zeros((10, 16, 16), dtype=np.float32)
    for channel_idx in range(10):
        satellite[channel_idx] = 0.6 + channel_idx * 0.02

    outputs = save_region_multimodal_panels(
        gt_rain=rain,
        pred_rain=pred,
        radar=radar,
        satellite=satellite,
        output_dir=tmp_path,
        sample_idx=0,
        frame_idx=2,
        geo_bounds=(0.0, 1.0, 0.0, 1.0),
        boundary_source=boundary_path,
        image_size=128,
    )

    assert sorted(outputs) == ["gt_rain", "pred_rain", "radar", "satellite"]
    for out_path in outputs.values():
        assert out_path.exists()
        assert out_path.stat().st_size > 0
        with Image.open(out_path) as image:
            assert image.size == (128, 128)


def test_save_next_frame_input_panels_with_local_geojson(tmp_path: Path) -> None:
    boundary = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [0.0, 0.0],
                            [1.0, 0.0],
                            [1.0, 1.0],
                            [0.0, 1.0],
                            [0.0, 0.0],
                        ]
                    ],
                },
            }
        ],
    }
    boundary_path = tmp_path / "boundary.json"
    boundary_path.write_text(json.dumps(boundary), encoding="utf-8")

    context_modalities = {
        "rain": np.zeros((1, 1, 4, 16, 16), dtype=np.float32),
        "radar": np.zeros((1, 1, 4, 16, 16), dtype=np.float32),
        "satellite": np.zeros((1, 10, 4, 16, 16), dtype=np.float32),
    }
    context_modalities["rain"][0, 0, :, 4:10, 5:12] = 0.8
    context_modalities["radar"][0, 0] = np.linspace(0.0, 1.0, 4 * 16 * 16, dtype=np.float32).reshape(4, 16, 16)
    context_modalities["satellite"][0, :, :] = 0.6

    outputs = save_next_frame_input_panels(
        context_modalities=context_modalities,
        output_dir=tmp_path,
        sample_idx=0,
        image_size=128,
        geo_bounds=(0.0, 1.0, 0.0, 1.0),
        boundary_source=boundary_path,
    )

    assert sorted(outputs) == ["input_radar", "input_rain", "input_satellite"]
    for paths in outputs.values():
        assert len(paths) == 4
        for out_path in paths:
            assert out_path.exists()
            assert out_path.stat().st_size > 0
            with Image.open(out_path) as image:
                assert image.size == (128, 128)
