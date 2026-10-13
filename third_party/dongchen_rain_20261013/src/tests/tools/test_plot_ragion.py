import json
from pathlib import Path

import numpy as np

from src.tools.plot_ragion import save_rain_region_image


def test_save_rain_region_image_with_local_geojson(tmp_path: Path) -> None:
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

    out_path = save_rain_region_image(
        np.array([[0.0, 0.2], [1.0, 5.0]], dtype=np.float32),
        tmp_path / "rain_region.jpg",
        geo_bounds=(0.0, 1.0, 0.0, 1.0),
        boundary_source=boundary_path,
    )

    assert out_path.exists()
    assert out_path.stat().st_size > 0
