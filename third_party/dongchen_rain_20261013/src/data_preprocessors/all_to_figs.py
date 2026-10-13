import io
from bisect import bisect_right
from datetime import datetime, timedelta
from math import ceil
from pathlib import Path
from typing import Literal, cast

import dateutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import tifffile
import torch
import torch.nn.functional as F
import webdataset as wds
from litdata.streaming.writer import BinaryWriter
from litdata.streaming.cache import Cache
from loguru import logger
from skimage.filters import gaussian
from torch import Tensor

from src.dataset.geo_utils import (
    any_modaility_time_to_local,
    local_to_any_modality_time,
)
from src.dataset.read_nc_file_mapped import radar_read, satellite_read
from src.tools.rain_station_excel_to_shard_db import ShardedRainDataImporter
from src.utils.logging.print import catch_any, log_print, logger, set_logger_file
from src.utils.visualization.color import color_rain_map

# *==============================================================
# * Utilities
# *==============================================================


def find_closest_time(target_time: datetime, time_list: list[datetime], use_prev: bool = False):
    if not use_prev:
        closest_index = min(range(len(time_list)), key=lambda i: abs(time_list[i] - target_time))
    else:
        closest_index = max(
            (i for i, t in enumerate(time_list) if t <= target_time),
            key=lambda i: time_list[i],
            default=-1,
        )
        # 如果没有找到目标时间之前的时间点，则使用第一个时间点
        if closest_index == -1:
            closest_index = 0
    closest_time = time_list[closest_index]
    delta_time = abs(closest_time - target_time)

    return closest_time, closest_index, delta_time


def _find_prev_next_time(target_time: datetime, sorted_times: list[datetime]) -> tuple[int, int]:
    """Return indices (prev_idx, next_idx) in a sorted time list such that:
    sorted_times[prev_idx] <= target_time <= sorted_times[next_idx].

    Falls back to edges if target_time is out of bounds.
    """
    if len(sorted_times) == 0:
        raise ValueError("sorted_times is empty")
    insert_at = bisect_right(sorted_times, target_time)
    prev_idx = max(0, insert_at - 1)
    next_idx = min(len(sorted_times) - 1, insert_at)
    return prev_idx, next_idx


def data_interpolate(data: np.ndarray | Tensor, img_size: int):
    data = torch.as_tensor(data).float()
    data = data.unsqueeze(0).unsqueeze(0) if data.dim() == 2 else data.unsqueeze(0)
    data = F.interpolate(data, size=(img_size, img_size), mode="bilinear", align_corners=False).squeeze(0)
    return data.numpy()


def gaussian_data(data: np.ndarray, sigma: float = 1.5, unchanged=True, n_times: int = 1):
    for i in range(n_times):
        max_d = np.max(data)
        data = gaussian(data, sigma=sigma)
        if unchanged:
            data = data * max_d / np.max(data)

    return data


@catch_any()
def take_all_modality_data(
    start_time_str: str,
    end_time_str: str,
    interval_minutes: int = 5,
    data_dir: str = "data2/",
    month: int = 5,
    img_size=256,
    geo_bounds=(97.3, 108.4, 26.1, 34.25),  # (97.0, 109.0, 26.0, 35.0),
    skip_no_rain=True,
    cond_min_delta_time=10,
    use_prev_time=False,
    year: int = 2023,
    rain_shards_name: str = "rainfall_shards",
    interp_non_rain_modalities: bool = False,
    max_interp_gap_minutes: int = 20,
):
    # dirs
    radar_dir = Path(data_dir) / "radar" / f"{year}{month:02d}"
    satellite_dir = Path(data_dir) / "satellite" / f"{year}{month:02d}"
    # rain shards can live outside `data_dir` (e.g. a shared global shards folder)
    rain_shards_path = Path(rain_shards_name)
    if rain_shards_path.is_absolute() or rain_shards_path.exists():
        rain_dir = rain_shards_path
    else:
        rain_dir = Path(data_dir) / rain_shards_name  # internal rain shards, not used in real dataset loading

    assert radar_dir.exists(), f"Radar directory {radar_dir} does not exist."
    assert satellite_dir.exists(), f"Satellite directory {satellite_dir} does not exist."
    assert rain_dir.exists(), f"Rainfall directory {rain_dir} does not exist."

    rain_ds = ShardedRainDataImporter(str(rain_dir))
    # print("Importing station information...")
    # rain_ds.import_station_info()

    # time
    start_dt = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
    end_dt = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")

    # required time list
    base_time_list = []
    current_dt = start_dt
    while current_dt <= end_dt:
        base_time_list.append(current_dt)
        current_dt += timedelta(minutes=interval_minutes)
    log_print(f"Processing data from {start_dt} to {end_dt}, total {len(base_time_list)} timestamps.")

    # radar/satellite times
    radar_file = list(radar_dir.rglob("*.nc"))
    radar_times = []
    satellite_times = []
    for file in radar_file:
        t = ".".join(file.stem.rsplit(".", 2)[-2:])
        t_dt = any_modaility_time_to_local(t, "radar")
        radar_times.append(t_dt)

    sate_files = list(satellite_dir.rglob("*.nc"))
    for file in sate_files:
        t = file.stem
        if t.startswith("NC"):
            # NC_H08_20251215_0500_R21_FLDK.06001_06001 for example
            yd, time = t.split(".")[:2]
            # convert into 20251215_0500
            regex_pattern = r"NC_\D+\d+_(\d+_\d+)_R21"
            import re

            match = re.search(regex_pattern, t)
            if match:
                t = match.group(1)
            else:
                raise ValueError(f"satellite name {t} can not be convert into universal time.")

        t_dt = any_modaility_time_to_local(t, "satellite")
        satellite_times.append(t_dt)

    assert len(radar_times) == len(radar_file), "Radar files and times mismatch."
    assert len(satellite_times) == len(sate_files), "Satellite files and times mismatch."

    # For interpolation we need time-sorted lists
    radar_pairs = sorted(zip(radar_times, radar_file), key=lambda x: x[0])
    radar_times_sorted = [p[0] for p in radar_pairs]
    radar_files_sorted = [p[1] for p in radar_pairs]

    satellite_pairs = sorted(zip(satellite_times, sate_files), key=lambda x: x[0])
    satellite_times_sorted = [p[0] for p in satellite_pairs]
    satellite_files_sorted = [p[1] for p in satellite_pairs]

    interp_radar = None
    interp_sat = None
    radar_cache: dict[str, object] = {}
    sat_cache: dict[str, object] = {}

    def _ensure_interp_loaded():
        nonlocal interp_radar, interp_sat
        if not interp_non_rain_modalities:
            return
        if interp_radar is None or interp_sat is None:
            from src.tools.optical_flow_interpolator import AnyModalityAnyFramesInterpolation

            interp_radar = AnyModalityAnyFramesInterpolation
            interp_sat = AnyModalityAnyFramesInterpolation

    def _get_radar_at_time(target_time: datetime) -> tuple[np.ndarray | None, dict[str, str]]:
        meta: dict[str, str] = {}
        if not interp_non_rain_modalities:
            closest_radar_time, index, delta_time = find_closest_time(target_time, radar_times, use_prev=use_prev_time)
            if delta_time > timedelta(minutes=cond_min_delta_time):
                log_print(f"Radar time is too far from target time: {delta_time}")
                return None, meta
            radar_file_path = radar_file[index]
            if not radar_file_path.exists():
                log_print(f"Radar file {radar_file_path} does not exist, skipping.", "warning")
                return None, meta
            radar_res = radar_read(
                radar_file_path,
                grid_width=None,
                grid_height=None,
                target_proj="epsg:4326",
                interpolation_method="regular_grid",
                crop_bounds_latlon=geo_bounds,
            )
            radar_data = radar_res.get("mapped_data")
            if radar_data is None or radar_data.size == 0:
                return None, meta
            radar_data = data_interpolate(radar_data, img_size)
            meta.update(
                {
                    "raw_radar_time": closest_radar_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "raw_radar_file": (radar_file_path.relative_to(data_dir)).as_posix(),
                }
            )
            return cast(np.ndarray, radar_data)[0], meta

        _ensure_interp_loaded()

        prev_idx, next_idx = _find_prev_next_time(target_time, radar_times_sorted)
        prev_t = radar_times_sorted[prev_idx]
        next_t = radar_times_sorted[next_idx]

        gap_min = int(round((next_t - prev_t).total_seconds() / 60.0))
        offset_min = int(round((target_time - prev_t).total_seconds() / 60.0))
        offset_min = max(0, min(offset_min, max(gap_min, 0)))

        if prev_idx == next_idx or gap_min <= 0 or gap_min > max_interp_gap_minutes:
            closest_radar_time, index, delta_time = find_closest_time(target_time, radar_times, use_prev=use_prev_time)
            if delta_time > timedelta(minutes=cond_min_delta_time):
                return None, meta
            radar_file_path = radar_file[index]
            if not radar_file_path.exists():
                return None, meta
            radar_res = radar_read(
                radar_file_path,
                grid_width=None,
                grid_height=None,
                target_proj="epsg:4326",
                interpolation_method="regular_grid",
                crop_bounds_latlon=geo_bounds,
            )
            radar_data = radar_res.get("mapped_data")
            if radar_data is None or radar_data.size == 0:
                return None, meta
            radar_data = data_interpolate(radar_data, img_size)
            meta.update(
                {
                    "raw_radar_time": closest_radar_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "raw_radar_file": (radar_file_path.relative_to(data_dir)).as_posix(),
                    "raw_radar_time_prev": prev_t.strftime("%Y-%m-%d %H:%M:%S"),
                    "raw_radar_time_next": next_t.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            return cast(np.ndarray, radar_data)[0], meta

        cache_key = (prev_idx, next_idx)
        cached_key = radar_cache.get("key")
        if cached_key != cache_key:
            prev_fp = radar_files_sorted[prev_idx]
            next_fp = radar_files_sorted[next_idx]
            if not prev_fp.exists() or not next_fp.exists():
                return None, meta

            prev_res = radar_read(
                prev_fp,
                grid_width=None,
                grid_height=None,
                target_proj="epsg:4326",
                interpolation_method="regular_grid",
                crop_bounds_latlon=geo_bounds,
            )
            next_res = radar_read(
                next_fp,
                grid_width=None,
                grid_height=None,
                target_proj="epsg:4326",
                interpolation_method="regular_grid",
                crop_bounds_latlon=geo_bounds,
            )
            prev_data = prev_res.get("mapped_data")
            next_data = next_res.get("mapped_data")
            if prev_data is None or prev_data.size == 0 or next_data is None or next_data.size == 0:
                return None, meta
            prev_data = data_interpolate(prev_data, img_size)[0]
            next_data = data_interpolate(next_data, img_size)[0]

            n_interp = gap_min - 1
            interpolator = cast(object, interp_radar)("radar", interp_n_frames=n_interp)
            seq = interpolator(np.stack([prev_data, next_data], axis=0))
            radar_cache["key"] = cache_key
            radar_cache["seq"] = seq
            radar_cache["prev_fp"] = prev_fp
            radar_cache["next_fp"] = next_fp
            radar_cache["prev_t"] = prev_t
            radar_cache["next_t"] = next_t

        seq = cast(np.ndarray, radar_cache["seq"])
        prev_fp = cast(Path, radar_cache["prev_fp"])
        next_fp = cast(Path, radar_cache["next_fp"])
        prev_t = cast(datetime, radar_cache["prev_t"])
        next_t = cast(datetime, radar_cache["next_t"])

        meta.update(
            {
                "raw_radar_time": prev_t.strftime("%Y-%m-%d %H:%M:%S"),
                "raw_radar_file": (prev_fp.relative_to(data_dir)).as_posix(),
                "raw_radar_time_prev": prev_t.strftime("%Y-%m-%d %H:%M:%S"),
                "raw_radar_time_next": next_t.strftime("%Y-%m-%d %H:%M:%S"),
                "raw_radar_file_prev": (prev_fp.relative_to(data_dir)).as_posix(),
                "raw_radar_file_next": (next_fp.relative_to(data_dir)).as_posix(),
            }
        )
        return seq[offset_min], meta

    def _get_satellite_at_time(target_time: datetime) -> tuple[np.ndarray | None, dict[str, str]]:
        meta: dict[str, str] = {}
        if not interp_non_rain_modalities:
            closest_satellite_time, index, delta_time = find_closest_time(
                target_time, satellite_times, use_prev=use_prev_time
            )
            if delta_time > timedelta(minutes=10):
                log_print(f"Satellite time is too far from target time: {delta_time}")
                return None, meta
            satellite_file_path = sate_files[index]
            if not satellite_file_path.exists():
                log_print(f"Satellite file {satellite_file_path} does not exist, skipping.", "warning")
                return None, meta
            sat_res = satellite_read(
                satellite_file_path,
                grid_width=img_size,
                grid_height=img_size,
                bands_range=(7, 17),
                target_proj="epsg:4326",
                interpolation_method="regular_grid",
                crop_bounds_latlon=geo_bounds,
                stack=True,
            )
            satellite_data = sat_res.get("mapped_bands")
            if satellite_data is None or satellite_data.size == 0:
                return None, meta
            satellite_data = data_interpolate(satellite_data, img_size)
            meta.update(
                {
                    "raw_satellite_time": closest_satellite_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "raw_satellite_file": (satellite_file_path.relative_to(data_dir)).as_posix(),
                }
            )
            return cast(np.ndarray, satellite_data), meta

        _ensure_interp_loaded()

        prev_idx, next_idx = _find_prev_next_time(target_time, satellite_times_sorted)
        prev_t = satellite_times_sorted[prev_idx]
        next_t = satellite_times_sorted[next_idx]

        gap_min = int(round((next_t - prev_t).total_seconds() / 60.0))
        offset_min = int(round((target_time - prev_t).total_seconds() / 60.0))
        offset_min = max(0, min(offset_min, max(gap_min, 0)))

        if prev_idx == next_idx or gap_min <= 0 or gap_min > max_interp_gap_minutes:
            closest_satellite_time, index, delta_time = find_closest_time(
                target_time, satellite_times, use_prev=use_prev_time
            )
            if delta_time > timedelta(minutes=10):
                return None, meta
            satellite_file_path = sate_files[index]
            if not satellite_file_path.exists():
                return None, meta
            sat_res = satellite_read(
                satellite_file_path,
                grid_width=img_size,
                grid_height=img_size,
                bands_range=(7, 17),
                target_proj="epsg:4326",
                interpolation_method="regular_grid",
                crop_bounds_latlon=geo_bounds,
                stack=True,
            )
            satellite_data = sat_res.get("mapped_bands")
            if satellite_data is None or satellite_data.size == 0:
                return None, meta
            satellite_data = data_interpolate(satellite_data, img_size)
            meta.update(
                {
                    "raw_satellite_time": closest_satellite_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "raw_satellite_file": (satellite_file_path.relative_to(data_dir)).as_posix(),
                    "raw_satellite_time_prev": prev_t.strftime("%Y-%m-%d %H:%M:%S"),
                    "raw_satellite_time_next": next_t.strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
            return cast(np.ndarray, satellite_data), meta

        cache_key = (prev_idx, next_idx)
        cached_key = sat_cache.get("key")
        if cached_key != cache_key:
            prev_fp = satellite_files_sorted[prev_idx]
            next_fp = satellite_files_sorted[next_idx]
            if not prev_fp.exists() or not next_fp.exists():
                return None, meta

            prev_res = satellite_read(
                prev_fp,
                grid_width=img_size,
                grid_height=img_size,
                bands_range=(7, 17),
                target_proj="epsg:4326",
                interpolation_method="regular_grid",
                crop_bounds_latlon=geo_bounds,
                stack=True,
            )
            next_res = satellite_read(
                next_fp,
                grid_width=img_size,
                grid_height=img_size,
                bands_range=(7, 17),
                target_proj="epsg:4326",
                interpolation_method="regular_grid",
                crop_bounds_latlon=geo_bounds,
                stack=True,
            )
            prev_data = prev_res.get("mapped_bands")
            next_data = next_res.get("mapped_bands")
            if prev_data is None or prev_data.size == 0 or next_data is None or next_data.size == 0:
                return None, meta
            prev_data = data_interpolate(prev_data, img_size)
            next_data = data_interpolate(next_data, img_size)

            n_interp = gap_min - 1
            interpolator = cast(object, interp_sat)("satellite", interp_n_frames=n_interp)
            seq = interpolator(np.stack([prev_data, next_data], axis=1))
            sat_cache["key"] = cache_key
            sat_cache["seq"] = seq
            sat_cache["prev_fp"] = prev_fp
            sat_cache["next_fp"] = next_fp
            sat_cache["prev_t"] = prev_t
            sat_cache["next_t"] = next_t

        seq = cast(np.ndarray, sat_cache["seq"])
        prev_fp = cast(Path, sat_cache["prev_fp"])
        next_fp = cast(Path, sat_cache["next_fp"])
        prev_t = cast(datetime, sat_cache["prev_t"])
        next_t = cast(datetime, sat_cache["next_t"])

        meta.update(
            {
                "raw_satellite_time": prev_t.strftime("%Y-%m-%d %H:%M:%S"),
                "raw_satellite_file": (prev_fp.relative_to(data_dir)).as_posix(),
                "raw_satellite_time_prev": prev_t.strftime("%Y-%m-%d %H:%M:%S"),
                "raw_satellite_time_next": next_t.strftime("%Y-%m-%d %H:%M:%S"),
                "raw_satellite_file_prev": (prev_fp.relative_to(data_dir)).as_posix(),
                "raw_satellite_file_next": (next_fp.relative_to(data_dir)).as_posix(),
            }
        )
        return seq[:, offset_min], meta

    # data taking
    for ii, t in enumerate(base_time_list):
        # rain data
        rain_start_t = t.strftime("%Y-%m-%d %H:%M:%S")
        rain_end_t = (t + timedelta(minutes=interval_minutes)).strftime("%Y-%m-%d %H:%M:%S")
        rain_res = rain_ds.meshgrid_rain(
            rain_start_t,
            rain_end_t,
            grid_width=img_size,
            grid_height=img_size,
            bounds=geo_bounds,
        )
        mapped = rain_res.get("mapped_data")
        if mapped is None:
            log_print(
                f"Rain debug: time={t}, window=[{rain_start_t} ~ {rain_end_t}], mapped_data is None",
                "warning",
            )
            continue

        rain_data = mapped[None].astype("float")
        stats = rain_res.get("statistics", {}) or {}
        max_rainfall = stats.get("max_rainfall", float(rain_data.max()))

        if skip_no_rain:
            log_print(
                f"Rain debug: time={t}, window=[{rain_start_t} ~ {rain_end_t}], "
                f"max_rainfall={max_rainfall:.4f}, rain_data_max={float(rain_data.max()):.4f}",
                "debug",
            )
            if max_rainfall < 0.01:
                continue  # skip if no rain data

        radar_data, radar_meta = _get_radar_at_time(t)
        if radar_data is None:
            log_print("Radar data is empty, skipping.", "warning")
            continue
        satellite_data, sat_meta = _get_satellite_at_time(t)
        if satellite_data is None:
            log_print("Satellite data is empty, skipping.", "warning")
            continue

        log_print(
            f"\n --------------------------------------------------------------\n"
            f"Processing time {t}, \n"
            f"  Radar time(meta): {radar_meta.get('raw_radar_time','')} \n"
            f"  Satellite time(meta): {sat_meta.get('raw_satellite_time','')} \n"
            f"  Rain data time: {t} with accumulated time {interval_minutes} minutes \n"
            f"     Rain data range <green>({rain_data.min()}, {rain_data.max()})</> \n"
            f"-----------------------------------------------------------------",
        )

        combined_meta = {}
        combined_meta.update(radar_meta)
        combined_meta.update(sat_meta)

        yield (
            {
                # time
                "time": t,
                "raw_radar_time": combined_meta.get("raw_radar_time", ""),
                "raw_satellite_time": combined_meta.get("raw_satellite_time", ""),
                # file path (backward-compatible "closest/prev" file)
                "raw_radar_file": combined_meta.get("raw_radar_file", ""),
                "raw_satellite_file": combined_meta.get("raw_satellite_file", ""),
                # optional interpolation endpoints
                "raw_radar_time_prev": combined_meta.get("raw_radar_time_prev", ""),
                "raw_radar_time_next": combined_meta.get("raw_radar_time_next", ""),
                "raw_radar_file_prev": combined_meta.get("raw_radar_file_prev", ""),
                "raw_radar_file_next": combined_meta.get("raw_radar_file_next", ""),
                "raw_satellite_time_prev": combined_meta.get("raw_satellite_time_prev", ""),
                "raw_satellite_time_next": combined_meta.get("raw_satellite_time_next", ""),
                "raw_satellite_file_prev": combined_meta.get("raw_satellite_file_prev", ""),
                "raw_satellite_file_next": combined_meta.get("raw_satellite_file_next", ""),
                # data
                "radar": radar_data,
                "satellite": satellite_data.transpose(1, 2, 0),  # (C, H, W) -> (H, W, C)
                "rain": rain_data[0],  # rain data is already in (H, W) format
                "rain_range": (rain_data.min(), rain_data.max()),
                "rainfall_bins": stats.get("rainfall_bins", {}),
            },
            ii,
            len(base_time_list),
        )


def save_tiff(save_path: str, data: np.ndarray, compression: str = "zlib"):
    assert compression in ["zlib", "jpeg2000"]

    if compression == "jpeg2000":
        compression_args = {"level": 85, "reversible": False}
    else:
        compression_args = None

    tifffile.imwrite(
        save_path,
        data,
        shape=data.shape,
        compression=compression,
        compressionargs=compression_args,
    )


def save_tiff_io(data: np.ndarray, compression: str = "zlib"):
    assert compression in ["zlib", "jpeg2000"]

    if compression == "jpeg2000":
        compression_args = {"level": 85, "reversible": False}
    else:
        compression_args = None

    bytes_io = io.BytesIO()
    tifffile.imwrite(
        bytes_io,
        data,
        shape=data.shape,
        compression=compression,
        compressionargs=compression_args,
    )

    return bytes_io.getvalue()


# *==============================================================
# * Main entry
# *==============================================================
def get_total_iters(st, et, interval):
    total_seconds = (et - st).total_seconds()
    return ceil(total_seconds / (interval * 60)) + 1


def save_all_modality_data_dirs(
    year: int = 2025,
    month: int = 5,
    interval: int = 30,
    output_dir: str | Path = "data_original/zihan_processed",
    skip_no_rain=True,
    cond_min_delta_time: int = 10,
    use_prev_time=False,
):
    output_dir = Path(output_dir) / f"2023{month:02d}"

    metainfo_file = output_dir / f"metadata.parquet"
    radar_output_dir = output_dir / "radar"
    satellite_output_dir = output_dir / "satellite"
    rain_output_dir = output_dir / "rain"
    rain_thumbnail_dir = output_dir / "rain_thumbnail"

    radar_output_dir.mkdir(parents=True, exist_ok=True)
    satellite_output_dir.mkdir(parents=True, exist_ok=True)
    rain_output_dir.mkdir(parents=True, exist_ok=True)
    rain_thumbnail_dir.mkdir(parents=True, exist_ok=True)

    parquet_writer = None
    schema = None

    rain_plot_fn, *_ = color_rain_map()

    try:
        st = datetime(2023, month, 1, 0, 0, 0)
        et = st + dateutil.relativedelta.relativedelta(months=1) - timedelta(seconds=1)
        for data, processed, total in take_all_modality_data(
            start_time_str=st.strftime("%Y-%m-%d %H:%M:%S"),
            end_time_str=et.strftime("%Y-%m-%d %H:%M:%S"),
            interval_minutes=interval,
            data_dir="data_original/",
            month=month,
            img_size=512,
            skip_no_rain=skip_no_rain,
            cond_min_delta_time=cond_min_delta_time,
            use_prev_time=use_prev_time,
            year=year,
        ):
            timestamp = data["time"].strftime("%Y%m%d_%H%M%S")

            radar_file_path = radar_output_dir / f"radar_{timestamp}.tiff"
            save_tiff(str(radar_file_path), data["radar"], compression="zlib")

            satellite_file_path = satellite_output_dir / f"satellite_{timestamp}.tiff"
            save_tiff(
                str(satellite_file_path),
                data["satellite"].astype("uint16"),
                compression="jpeg2000",
            )

            rain_file_path = rain_output_dir / f"rain_{timestamp}.tiff"
            save_tiff(str(rain_file_path), data["rain"])

            # plot rain
            rain_preview_path = rain_thumbnail_dir / f"modalities_{timestamp}.jpg"
            rain_data = data["rain"]
            rain_data_G = gaussian_data(rain_data, n_times=2)
            *_, rain_plotted = rain_plot_fn(rain_data_G, return_ndarray=True)

            # plot three modalities
            fig, axs = plt.subplot_mosaic(
                [["radar", "satellite", "rain"]],
                layout="constrained",
                width_ratios=[1, 1, 1],
                figsize=(12, 5),
            )
            axs["radar"].imshow(data["radar"] / data["radar"].max(), cmap="turbo")
            sat_rgb = data["satellite"] / data["satellite"].max()
            sat_rgb = sat_rgb[..., [9, 8, 7]]
            axs["satellite"].imshow(sat_rgb)
            axs["rain"].imshow(rain_plotted[..., :3])
            plt.suptitle(f"Time: {data['time'].strftime('%Y-%m-%d %H:%M:%S')}")
            plt.savefig(rain_preview_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            plt.clf()
            plt.cla()

            metadata_record = {
                "time": data["time"].strftime("%Y-%m-%d %H:%M:%S"),
                "radar_file": str(radar_file_path.relative_to(output_dir)),
                "satellite_file": str(satellite_file_path.relative_to(output_dir)),
                "rain_file": str(rain_file_path.relative_to(output_dir)),
                # raw time
                "raw_radar_time": data["raw_radar_time"],
                "raw_satellite_time": data["raw_satellite_time"],
                # raw file
                "raw_radar_file": data["raw_radar_file"],
                "raw_satellite_file": data["raw_satellite_file"],
                "rain_range_min": data["rain_range"][0],
                "rain_range_max": data["rain_range"][1],
            }

            if schema is None:
                df = pd.DataFrame([metadata_record])
                table = pa.Table.from_pandas(df)
                schema = table.schema
                parquet_writer = pq.ParquetWriter(metainfo_file, schema)

            df = pd.DataFrame([metadata_record])
            table = pa.Table.from_pandas(df, schema=schema)
            parquet_writer.write_table(table)

            log_print(
                f"Processed and saved data for time: {data['time']}",
                index=processed,
                total=total,
            )

    except Exception as e:
        log_print(f"Error processing data index={processed}, total={total}: {e}", "error")
    finally:
        parquet_writer.close() if parquet_writer is not None else None
        log_print("Finished processing all data")


def save_all_modality_data_wids(
    month: int = 5,
    interval: int = 30,
    data_dir: str = "data",
    output_dir: str | Path = "data2/wids2023",
    skip_no_rain=True,
    cond_min_delta_time: int = 10,
    use_prev_time=False,
):
    output_dir = Path(output_dir) / f"2023{month:02d}"

    metainfo_file = output_dir / f"metadata.parquet"
    # radar_output_dir = output_dir / "radar"
    # satellite_output_dir = output_dir / "satellite"
    # rain_output_dir = output_dir / "rain"
    webdataset_dir = output_dir / "pairs"
    rain_thumbnail_dir = output_dir / "rain_thumbnail"

    webdataset_dir.mkdir(parents=True, exist_ok=True)
    # satellite_output_dir.mkdir(parents=True, exist_ok=True)
    # rain_output_dir.mkdir(parents=True, exist_ok=True)
    rain_thumbnail_dir.mkdir(parents=True, exist_ok=True)

    wds_writer = wds.ShardWriter((webdataset_dir / "pairs_%02d.tar").as_posix(), maxcount=500)

    parquet_writer = None
    schema = None

    rain_plot_fn, *_ = color_rain_map()

    try:
        st = datetime(2023, month, 1, 0, 0, 0)
        et = st + dateutil.relativedelta.relativedelta(months=1) - timedelta(seconds=1)
        for index, (data, processed, total) in enumerate(
            take_all_modality_data(
                start_time_str=st.strftime("%Y-%m-%d %H:%M:%S"),
                end_time_str=et.strftime("%Y-%m-%d %H:%M:%S"),
                interval_minutes=interval,
                data_dir=data_dir,
                month=month,
                img_size=512,
                skip_no_rain=skip_no_rain,
                cond_min_delta_time=cond_min_delta_time,
                use_prev_time=use_prev_time,
            )
        ):
            timestamp = data["time"].strftime("%Y%m%d_%H%M%S")

            radar_bytes = save_tiff_io(data["radar"], compression="zlib")
            sat_bytes = save_tiff_io(data["satellite"].astype("uint16"), compression="jpeg2000")
            rain_bytes = save_tiff_io(data["rain"], compression="zlib")

            # plot rain
            rain_preview_path = rain_thumbnail_dir / f"modalities_{timestamp}.jpg"
            rain_data = data["rain"]
            rain_data_G = gaussian_data(rain_data, n_times=2)
            *_, rain_plotted = rain_plot_fn(rain_data_G, return_ndarray=True)
            rain_data_G_bytes = save_tiff_io(rain_data_G, compression="zlib")

            # plot three modalities
            # fig, axs = plt.subplot_mosaic(
            #     [["radar", "satellite", "rain"]],
            #     layout="constrained",
            #     width_ratios=[1, 1, 1],
            #     figsize=(12, 5),
            # )
            # axs["radar"].imshow(data["radar"] / data["radar"].max(), cmap="turbo")
            # sat_rgb = data["satellite"] / data["satellite"].max()
            # sat_rgb = sat_rgb[..., [9, 8, 7]]
            # axs["satellite"].imshow(sat_rgb)
            # axs["rain"].imshow(rain_plotted[..., :3])
            # plt.suptitle(f"Time: {data['time'].strftime('%Y-%m-%d %H:%M:%S')}")
            # plt.savefig(rain_preview_path, dpi=200, bbox_inches="tight")
            # plt.close(fig)
            # plt.clf()
            # plt.cla()

            # wds dict
            wds_saved = {
                "__key__": f"modalities_{timestamp}",
                "radar.tiff": radar_bytes,
                "satellite.tiff": sat_bytes,
                "rain.tiff": rain_bytes,
                "rain_interpolated.tiff": rain_data_G_bytes,
            }

            metadata_record = {
                "shard_index": index,
                "time": data["time"].strftime("%Y-%m-%d %H:%M:%S"),
                # "radar_file": str(radar_file_path.relative_to(output_dir)),
                # "satellite_file": str(satellite_file_path.relative_to(output_dir)),
                # "rain_file": str(rain_file_path.relative_to(output_dir)),
                # raw time
                "raw_radar_time": data["raw_radar_time"],
                "raw_satellite_time": data["raw_satellite_time"],
                # raw file
                "raw_radar_file": data["raw_radar_file"],
                "raw_satellite_file": data["raw_satellite_file"],
                "rain_range_min": data["rain_range"][0],
                "rain_range_max": data["rain_range"][1],
            }

            if schema is None:
                df = pd.DataFrame([metadata_record])
                table = pa.Table.from_pandas(df)
                schema = table.schema
                parquet_writer = pq.ParquetWriter(metainfo_file, schema)

            df = pd.DataFrame([metadata_record])
            table = pa.Table.from_pandas(df, schema=schema)
            parquet_writer.write_table(table)
            wds_writer.write(wds_saved)

            log_print(
                f"Processed and saved data for time: {data['time']}",
                index=processed,
                total=total,
            )

    except Exception as e:
        log_print(f"Error processing data index={processed}, total={total}: {e}", "error")
    finally:
        parquet_writer.close() if parquet_writer is not None else None
        wds_writer.close()
        log_print("Finished processing all data")


def save_all_modality_data_litdata(
    year: int = 2025,
    month: int = 5,
    interval: int = 30,
    data_dir: str = "data2/",
    output_dir: str | Path = "data2/litdata2025",
    skip_no_rain=True,
    cond_min_delta_time: int = 10,
    use_prev_time=False,
):
    from tqdm import tqdm

    output_dir = Path(output_dir) / f"{year}{month:02d}"

    metainfo_file = output_dir / f"metadata.parquet"
    # radar_output_dir = output_dir / "radar"
    # satellite_output_dir = output_dir / "satellite"
    # rain_output_dir = output_dir / "rain"
    litdata_dir = output_dir / "pairs"
    rain_thumbnail_dir = output_dir / "rain_thumbnail"

    litdata_dir.mkdir(parents=True, exist_ok=True)
    # satellite_output_dir.mkdir(parents=True, exist_ok=True)
    # rain_output_dir.mkdir(parents=True, exist_ok=True)
    rain_thumbnail_dir.mkdir(parents=True, exist_ok=True)

    parquet_writer = None
    schema = None

    rain_plot_fn, *_ = color_rain_map()

    # writer = BinaryWriter(str(litdata_dir), chunk_bytes="128Mb")
    writer = Cache(str(litdata_dir), chunk_bytes="256Mb")

    st = datetime(year, month, 1, 0, 0, 0)
    et = st + dateutil.relativedelta.relativedelta(months=1) - timedelta(seconds=1)
    total_iters = get_total_iters(st, et, interval)
    iter_fn = enumerate(
        take_all_modality_data(
            start_time_str=st.strftime("%Y-%m-%d %H:%M:%S"),
            end_time_str=et.strftime("%Y-%m-%d %H:%M:%S"),
            interval_minutes=interval,
            data_dir=data_dir,
            month=month,
            img_size=512,
            skip_no_rain=skip_no_rain,
            cond_min_delta_time=cond_min_delta_time,
            use_prev_time=use_prev_time,
            year=year,
        )
    )

    try:
        for index, (data, processed, total) in (tbar := tqdm(iter_fn, total=total_iters)):
            timestamp = data["time"].strftime("%Y%m%d_%H%M%S")

            radar_bytes = save_tiff_io(data["radar"], compression="zlib")
            sat_bytes = save_tiff_io(data["satellite"].astype("uint16"), compression="jpeg2000")
            rain_bytes = save_tiff_io(data["rain"], compression="zlib")

            # plot rain
            rain_preview_path = rain_thumbnail_dir / f"modalities_{timestamp}.jpg"
            rain_data = data["rain"]
            rain_data_G = gaussian_data(rain_data, n_times=2)
            *_, rain_plotted = rain_plot_fn(rain_data_G, return_ndarray=True)
            rain_data_G_bytes = save_tiff_io(rain_data_G, compression="zlib")

            # plot three modalities
            fig, axs = plt.subplot_mosaic(
                [["radar", "satellite", "rain"]],
                layout="constrained",
                width_ratios=[1, 1, 1],
                figsize=(12, 5),
            )
            axs["radar"].imshow(data["radar"] / data["radar"].max(), cmap="turbo")
            sat_rgb = data["satellite"] / data["satellite"].max()
            sat_rgb = sat_rgb[..., [9, 8, 7]]
            axs["satellite"].imshow(sat_rgb)
            axs["rain"].imshow(rain_plotted[..., :3])
            plt.suptitle(f"Time: {data['time'].strftime('%Y-%m-%d %H:%M:%S')}")
            plt.savefig(rain_preview_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            plt.clf()
            plt.cla()

            # wds dict
            wds_saved = {
                "__key__": f"modalities_{timestamp}",
                "radar": radar_bytes,
                "satellite": sat_bytes,
                "rain": rain_bytes,
                "rain_interpolated": rain_data_G_bytes,
            }

            metadata_record = {
                "shard_index": index,
                "time": data["time"].strftime("%Y-%m-%d %H:%M:%S"),
                # "radar_file": str(radar_file_path.relative_to(output_dir)),
                # "satellite_file": str(satellite_file_path.relative_to(output_dir)),
                # "rain_file": str(rain_file_path.relative_to(output_dir)),
                # raw time
                "raw_radar_time": data["raw_radar_time"],
                "raw_satellite_time": data["raw_satellite_time"],
                # raw file
                "raw_radar_file": data["raw_radar_file"],
                "raw_satellite_file": data["raw_satellite_file"],
                "rain_range_min": data["rain_range"][0],
                "rain_range_max": data["rain_range"][1],
                "rainfall_bins": data["rainfall_bins"],
            }

            if schema is None:
                df = pd.DataFrame([metadata_record])
                table = pa.Table.from_pandas(df)
                schema = table.schema
                parquet_writer = pq.ParquetWriter(metainfo_file, schema)

            df = pd.DataFrame([metadata_record])
            table = pa.Table.from_pandas(df, schema=schema)
            parquet_writer.write_table(table)
            writer._add_item(index, wds_saved)
            tbar.set_description(f"Processing data index={index}/{total_iters}")

            log_print(
                f"Processed and saved data for time: {data['time']}",
                index=processed,
                total=total,
            )

    except Exception as e:
        logger.warning(f"Error processing data index={processed}, total={total}: {e}", tqdm=True)
    finally:
        parquet_writer.close() if parquet_writer is not None else None
        writer.done()
        log_print("Finished processing all data")


def test_all_structural_loading():
    for data in take_all_modality_data(
        start_time_str="2023-05-01 00:00:00",
        end_time_str="2023-05-31 00:00:00",
        interval_minutes=60,
        data_dir="data_original/",
        month=5,
        img_size=256,
    ):
        # print the time and data shapes
        log_print(f"Time: {data['time']}")
        log_print(f"Radar shape: {data['radar'].shape}")
        log_print(f"Satellite shape: {data['satellite'].shape}")
        log_print(f"Rain shape: {data['rain'].shape}")


############# configurations

# ----------------------------------

months = list(range(5,11))
interval = 10  # minutes
data_dir = "data/raw_dataset_2023"
output_dir = f"data2/litdata_train_2023_2025/litdata_interval_{interval}"
use_prev_time = True
year = 2023

# ----------------------------------


########## single processing

file_handler = None
for month in months:
    if file_handler is not None:
        logger.remove(file_handler)
    set_logger_file(
        file=Path(output_dir) / f"{year}{month:02d}.log",
        level="debug",
        add_time=False,
    )
    log_print(f"Processing month: {month}")
    # save_all_modality_data_dirs(
    #     month=month,
    #     interval=interval,
    #     output_dir=output_dir,
    #     use_prev_time=use_prev_time,
    # )

    # save_all_modality_data_wids(
    #     month=month,
    #     interval=interval,
    #     output_dir=output_dir,
    #     use_prev_time=use_prev_time,
    # )
    save_all_modality_data_litdata(
        year=year,
        data_dir=data_dir,
        month=month,
        interval=interval,
        output_dir=output_dir,
        use_prev_time=use_prev_time,
    )

########## multiprocessing

# import concurrent.futures as cf

# f_ids = []
# with cf.ProcessPoolExecutor(max_workers=4) as executor:
#     for i, month in enumerate(months):
#         futures = [
#             executor.submit(
#                 save_all_modality_data,
#                 month=month,
#                 interval=interval,
#                 output_dir=output_dir,
#             )
#         ]
#         f_ids.append(futures)

#     for future, process_id in zip(cf.as_completed(futures), f_ids):
#         try:
#             with logger.contextualize(pid=process_id):
#                 future.result()
#         except Exception as e:
#             log_print(f"Error processing month: {e}", "error")


########## other tests

# time_str_list = [
#     "20230501_020000",
#     "20230501_000000",
#     "20230501_003000",
#     "20230501_023000",
#     "20230501_010000",
# ]

# consecutive_times = find_consecutive_time(time_str_list)

# for times, indices in consecutive_times:
#     print("Times:", [t.strftime("%Y%m%d_%H%M%S") for t in times])
#     print("Original Indices:", indices)
