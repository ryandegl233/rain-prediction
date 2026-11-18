#!/usr/bin/env python3
import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.geo_utils import any_modaility_time_to_local
from src.dataset.read_nc_file_mapped import radar_read, satellite_read
from src.tools.rain_station_excel_to_shard_db import ShardedRainDataImporter


@dataclass
class GridSummary:
    name: str
    shape: tuple[int, ...]
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    lon_center: float
    lat_center: float
    lon_step: float
    lat_step: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check geo alignment for radar, satellite and rain grids.")
    parser.add_argument("--data-dir", type=str, default="/home/rainpred/RainPrediction/data2/raw_2025_trainset")
    parser.add_argument("--rain-shards-name", type=str, default="rainfall_shards")
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--month", type=int, default=5)
    parser.add_argument("--interval-minutes", type=int, default=10)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--geo-bounds", type=float, nargs=4, default=[97.3, 108.4, 26.1, 34.25])
    parser.add_argument("--bands-start", type=int, default=7)
    parser.add_argument("--bands-end", type=int, default=17)
    parser.add_argument("--max-time-delta-minutes", type=int, default=10)
    parser.add_argument("--out-json", type=str, default="")
    return parser.parse_args()


def _resolve_rain_dir(data_dir: Path, rain_shards_name: str) -> Path:
    rain_shards_path = Path(rain_shards_name)
    if rain_shards_path.is_absolute() or rain_shards_path.exists():
        return rain_shards_path
    return data_dir / rain_shards_name


def _scan_radar_times(folder: Path) -> dict[datetime, Path]:
    radar_times: dict[datetime, Path] = {}
    for file in sorted(folder.rglob("*.nc")):
        time_token = ".".join(file.stem.rsplit(".", 2)[-2:])
        radar_times[any_modaility_time_to_local(time_token, "radar")] = file
    return radar_times


def _scan_satellite_times(folder: Path) -> dict[datetime, Path]:
    satellite_times: dict[datetime, Path] = {}
    for file in sorted(folder.rglob("*.nc")):
        time_token = file.stem
        if time_token.startswith("NC"):
            match = re.search(r"NC_\D+\d+_(\d+_\d+)_R21", time_token)
            if match is None:
                raise ValueError(f"satellite name {time_token} can not be converted into local time")
            time_token = match.group(1)
        satellite_times[any_modaility_time_to_local(time_token, "satellite")] = file
    return satellite_times


def _pick_sample_times(common_times: list[datetime], samples: int) -> list[datetime]:
    if not common_times:
        return []
    if len(common_times) <= samples:
        return common_times

    chosen_indices = np.linspace(0, len(common_times) - 1, samples, dtype=int)
    unique_indices: list[int] = []
    for index in chosen_indices.tolist():
        if index not in unique_indices:
            unique_indices.append(index)
    return [common_times[index] for index in unique_indices]


def _summarize_grid(name: str, data: np.ndarray, lon_mesh: np.ndarray, lat_mesh: np.ndarray) -> GridSummary:
    center_y = lon_mesh.shape[0] // 2
    center_x = lon_mesh.shape[1] // 2
    lon_step = float(lon_mesh[0, 1] - lon_mesh[0, 0]) if lon_mesh.shape[1] > 1 else 0.0
    lat_step = float(lat_mesh[1, 0] - lat_mesh[0, 0]) if lat_mesh.shape[0] > 1 else 0.0

    return GridSummary(
        name=name,
        shape=tuple(int(dim) for dim in data.shape),
        lon_min=float(np.nanmin(lon_mesh)),
        lon_max=float(np.nanmax(lon_mesh)),
        lat_min=float(np.nanmin(lat_mesh)),
        lat_max=float(np.nanmax(lat_mesh)),
        lon_center=float(lon_mesh[center_y, center_x]),
        lat_center=float(lat_mesh[center_y, center_x]),
        lon_step=lon_step,
        lat_step=lat_step,
    )


def _diff_summary(base: GridSummary, other: GridSummary) -> dict[str, float]:
    return {
        "lon_min_diff": other.lon_min - base.lon_min,
        "lon_max_diff": other.lon_max - base.lon_max,
        "lat_min_diff": other.lat_min - base.lat_min,
        "lat_max_diff": other.lat_max - base.lat_max,
        "lon_center_diff": other.lon_center - base.lon_center,
        "lat_center_diff": other.lat_center - base.lat_center,
        "lon_step_diff": other.lon_step - base.lon_step,
        "lat_step_diff": other.lat_step - base.lat_step,
    }


def _print_grid_summary(summary: GridSummary) -> None:
    print(
        f"  [{summary.name}] shape={summary.shape} "
        f"lon=({summary.lon_min:.6f}, {summary.lon_max:.6f}) "
        f"lat=({summary.lat_min:.6f}, {summary.lat_max:.6f}) "
        f"center=({summary.lon_center:.6f}, {summary.lat_center:.6f}) "
        f"step=({summary.lon_step:.6f}, {summary.lat_step:.6f})"
    )


def main() -> None:
    args = _parse_args()
    geo_bounds = tuple(float(v) for v in args.geo_bounds)
    data_dir = Path(args.data_dir)
    radar_dir = data_dir / "radar" / f"{args.year}{args.month:02d}"
    satellite_dir = data_dir / "satellite" / f"{args.year}{args.month:02d}"
    rain_dir = _resolve_rain_dir(data_dir, args.rain_shards_name)

    assert radar_dir.exists(), f"Radar directory not found: {radar_dir}"
    assert satellite_dir.exists(), f"Satellite directory not found: {satellite_dir}"
    assert rain_dir.exists(), f"Rain shards directory not found: {rain_dir}"

    rain_ds = ShardedRainDataImporter(str(rain_dir))
    radar_times = _scan_radar_times(radar_dir)
    satellite_times = _scan_satellite_times(satellite_dir)
    common_times = sorted(set(radar_times) & set(satellite_times))
    sample_times = _pick_sample_times(common_times, max(1, int(args.samples)))

    if not sample_times:
        raise RuntimeError("No common radar and satellite timestamps were found.")

    print(f"Requested geo_bounds={geo_bounds}")
    print(f"Found {len(common_times)} common timestamps, checking {len(sample_times)} samples.")

    report: dict[str, object] = {
        "requested_geo_bounds": geo_bounds,
        "year": args.year,
        "month": args.month,
        "interval_minutes": args.interval_minutes,
        "img_size": args.img_size,
        "samples": [],
    }

    for sample_time in sample_times:
        radar_result = radar_read(
            str(radar_times[sample_time]),
            grid_width=None,
            grid_height=None,
            target_proj="epsg:4326",
            interpolation_method="regular_grid",
            crop_bounds_latlon=geo_bounds,
        )
        satellite_result = satellite_read(
            str(satellite_times[sample_time]),
            grid_width=args.img_size,
            grid_height=args.img_size,
            bands_range=(args.bands_start, args.bands_end),
            target_proj="epsg:4326",
            interpolation_method="regular_grid",
            crop_bounds_latlon=geo_bounds,
            stack=True,
        )
        rain_result = rain_ds.meshgrid_rain(
            start_time=sample_time.strftime("%Y-%m-%d %H:%M:%S"),
            end_time=(sample_time + timedelta(minutes=args.interval_minutes)).strftime("%Y-%m-%d %H:%M:%S"),
            grid_width=args.img_size,
            grid_height=args.img_size,
            bounds=geo_bounds,
        )

        radar_data = radar_result.get("mapped_data")
        satellite_data = satellite_result.get("mapped_bands")
        rain_data = rain_result.get("mapped_data")

        if radar_data is None or radar_data.size == 0:
            print(f"\nTime {sample_time}: radar data is empty, skipped.")
            continue
        if satellite_data is None or satellite_data.size == 0:
            print(f"\nTime {sample_time}: satellite data is empty, skipped.")
            continue
        if rain_data is None or rain_data.size == 0:
            print(f"\nTime {sample_time}: rain data is empty, skipped.")
            continue

        radar_summary = _summarize_grid(
            "radar",
            radar_data,
            np.asarray(radar_result["lon_mesh"]),
            np.asarray(radar_result["lat_mesh"]),
        )
        satellite_summary = _summarize_grid(
            "satellite",
            satellite_data,
            np.asarray(satellite_result["lon_mesh"]),
            np.asarray(satellite_result["lat_mesh"]),
        )
        rain_summary = _summarize_grid(
            "rain",
            rain_data,
            np.asarray(rain_result["lon_mesh"]),
            np.asarray(rain_result["lat_mesh"]),
        )

        radar_vs_rain = _diff_summary(rain_summary, radar_summary)
        satellite_vs_rain = _diff_summary(rain_summary, satellite_summary)
        radar_time_delta_minutes = (
            any_modaility_time_to_local(".".join(radar_times[sample_time].stem.rsplit(".", 2)[-2:]), "radar")
            - sample_time
        ).total_seconds() / 60
        satellite_time_delta_minutes = (
            any_modaility_time_to_local(
                re.search(r"NC_\D+\d+_(\d+_\d+)_R21", satellite_times[sample_time].stem).group(1)
                if satellite_times[sample_time].stem.startswith("NC")
                else satellite_times[sample_time].stem,
                "satellite",
            )
            - sample_time
        ).total_seconds() / 60

        print(f"\nTime {sample_time.strftime('%Y-%m-%d %H:%M:%S')}")
        _print_grid_summary(rain_summary)
        _print_grid_summary(radar_summary)
        _print_grid_summary(satellite_summary)
        print(f"  [radar-rain diff] {json.dumps(radar_vs_rain, ensure_ascii=False)}")
        print(f"  [sat-rain diff] {json.dumps(satellite_vs_rain, ensure_ascii=False)}")
        print(
            "  "
            f"[time delta minutes] radar={radar_time_delta_minutes:.1f}, "
            f"satellite={satellite_time_delta_minutes:.1f}"
        )

        sample_report = {
            "time": sample_time.strftime("%Y-%m-%d %H:%M:%S"),
            "radar_file": str(radar_times[sample_time]),
            "satellite_file": str(satellite_times[sample_time]),
            "rain_statistics": rain_result.get("statistics", {}),
            "rain": asdict(rain_summary),
            "radar": asdict(radar_summary),
            "satellite": asdict(satellite_summary),
            "radar_vs_rain": radar_vs_rain,
            "satellite_vs_rain": satellite_vs_rain,
            "radar_time_delta_minutes": radar_time_delta_minutes,
            "satellite_time_delta_minutes": satellite_time_delta_minutes,
        }
        cast_samples = report["samples"]
        assert isinstance(cast_samples, list)
        cast_samples.append(sample_report)

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved report to {out_path}")


if __name__ == "__main__":
    main()
