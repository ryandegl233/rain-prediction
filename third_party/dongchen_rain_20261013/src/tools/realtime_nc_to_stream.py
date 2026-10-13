#!/usr/bin/env python3
import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
import sys

import h5py
import numpy as np
import tifffile
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset.geo_utils import any_modaility_time_to_local
from src.dataset.read_nc_file_mapped import radar_read, satellite_read


RAIN_KEY_CANDIDATES: list[str] = ["rain", "rain_interpolated", "var"]


def _parse_time(path: Path, modality: str) -> datetime:
    name = path.stem
    if re.fullmatch(r"\d{8}_\d{4}", name):
        return datetime.strptime(name, "%Y%m%d_%H%M")
    if modality == "radar":
        token = ".".join(name.rsplit(".", 2)[-2:])
        return any_modaility_time_to_local(token, "radar")
    if modality == "satellite":
        if name.startswith("NC"):
            m = re.search(r"NC_\D+\d+_(\d+_\d+)_R21", name)
            if m is None:
                raise ValueError(f"cannot parse satellite time from {name}")
            name = m.group(1)
        return any_modaility_time_to_local(name, "satellite")
    raise ValueError(f"unknown modality {modality}")


def _scan_nc(folder: Path, modality: str) -> dict[datetime, Path]:
    files = sorted(folder.rglob("*.nc"))
    return {_parse_time(file, modality): file for file in files}


def _read_rain_nc(path: Path, out_h: int, out_w: int) -> np.ndarray:
    with h5py.File(path, "r") as file:
        rain = None
        for key in RAIN_KEY_CANDIDATES:
            if key in file:
                rain = np.asarray(file[key][()])
                break
        if rain is None:
            raise KeyError(f"{path} missing rain key from {RAIN_KEY_CANDIDATES}")

        lat_key = "latitude" if "latitude" in file else ("lat" if "lat" in file else None)
        if lat_key is not None:
            lat = np.asarray(file[lat_key][()])
            if lat.ndim == 1 and lat.size >= 2 and bool(lat[0] > lat[-1]):
                rain = rain[::-1, :].copy()

    rain_t = torch.as_tensor(np.ascontiguousarray(rain), dtype=torch.float32)[None, None]
    rain_t = F.interpolate(rain_t, size=(out_h, out_w), mode="bilinear", align_corners=False)
    rain_np = rain_t[0, 0].cpu().numpy()
    return np.nan_to_num(rain_np, nan=0.0)


def _save_frame(out_dir: Path, timestamp: datetime, radar: np.ndarray, sat: np.ndarray, rain: np.ndarray) -> None:
    ts = timestamp.strftime("%Y%m%d_%H%M")
    frame_dir = out_dir / ts
    frame_dir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(frame_dir / "radar.tiff", radar.astype(np.float32), compression="zlib")
    tifffile.imwrite(frame_dir / "satellite.tiff", sat.astype(np.float32), compression="zlib")
    tifffile.imwrite(frame_dir / "rain_interpolated.tiff", rain.astype(np.float32), compression="zlib")
    meta = {
        "timestamp": ts,
        "shape": {
            "radar": list(radar.shape),
            "satellite": list(sat.shape),
            "rain": list(rain.shape),
        },
    }
    (frame_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _resize_2d(arr: np.ndarray, img_size: int) -> np.ndarray:
    t = torch.as_tensor(arr, dtype=torch.float32)[None, None]
    t = F.interpolate(t, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return t[0, 0].cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Realtime NC -> mapped TIFF stream producer (all NC mode)")
    parser.add_argument("--radar-dir", type=str, default=DEFAULT_CONFIG["radar_dir"])
    parser.add_argument("--sat-dir", type=str, default=DEFAULT_CONFIG["sat_dir"])
    parser.add_argument("--rain-dir", type=str, default=DEFAULT_CONFIG["rain_dir"])
    parser.add_argument("--out-dir", type=str, default=DEFAULT_CONFIG["out_dir"])
    parser.add_argument("--img-size", type=int, default=DEFAULT_CONFIG["img_size"])
    parser.add_argument("--geo-bounds", type=float, nargs=4, default=DEFAULT_CONFIG["geo_bounds"])
    parser.add_argument("--poll-seconds", type=int, default=DEFAULT_CONFIG["poll_seconds"])
    args = parser.parse_args()

    radar_dir = Path(args.radar_dir)
    sat_dir = Path(args.sat_dir)
    rain_dir = Path(args.rain_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    done_file = out_dir / "_done.json"
    done: set[str] = set(json.loads(done_file.read_text(encoding="utf-8")) if done_file.exists() else [])

    while True:
        radar_map = _scan_nc(radar_dir, "radar")
        sat_map = _scan_nc(sat_dir, "satellite")
        rain_map = _scan_nc(rain_dir, "radar")
        common_times = sorted(set(radar_map) & set(sat_map) & set(rain_map))

        for ts in common_times:
            key = ts.strftime("%Y%m%d_%H%M")
            if key in done:
                continue

            radar_res = radar_read(
                str(radar_map[ts]),
                grid_width=None,
                grid_height=None,
                target_proj="epsg:4326",
                interpolation_method="regular_grid",
                crop_bounds_latlon=tuple(args.geo_bounds),
            )
            radar = radar_res.get("mapped_data")
            if radar is None or radar.size == 0:
                continue
            radar = np.nan_to_num(radar, nan=0.0)
            radar = _resize_2d(radar, args.img_size)

            sat_res = satellite_read(
                str(sat_map[ts]),
                grid_width=args.img_size,
                grid_height=args.img_size,
                bands_range=(7, 17),
                target_proj="epsg:4326",
                interpolation_method="regular_grid",
                crop_bounds_latlon=tuple(args.geo_bounds),
                stack=True,
            )
            sat = sat_res.get("mapped_bands")
            if sat is None or sat.size == 0:
                continue
            sat = np.nan_to_num(sat, nan=0.0)

            rain = _read_rain_nc(path=rain_map[ts], out_h=args.img_size, out_w=args.img_size)

            _save_frame(out_dir=out_dir, timestamp=ts, radar=radar, sat=sat, rain=rain)
            done.add(key)
            done_file.write_text(json.dumps(sorted(done), ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[producer] wrote frame {key}")

        time.sleep(max(1, int(args.poll_seconds)))





# Example:
# python src/tools/realtime_nc_to_stream.py \
#   --radar-dir test_data/achn \
#   --sat-dir test_data/sat/clip \
#   --rain-dir test_data/rain \
#   --out-dir runtime_stream \
#   --img-size 256
DEFAULT_CONFIG = {
    "radar_dir": "test_data/achn",
    "sat_dir": "test_data/sat/clip",
    "rain_dir": "test_data/rain",
    "out_dir": "runtime_stream",
    "img_size": 256,
    "geo_bounds": [97.3, 108.4, 26.1, 34.25],
    "poll_seconds": 20,
}

if __name__ == "__main__":
    main()
