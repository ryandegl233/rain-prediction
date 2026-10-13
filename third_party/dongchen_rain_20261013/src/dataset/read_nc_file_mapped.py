import datetime
import os
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytz
import xarray as xr

from src.dataset.geo_utils import (
    create_display_coordinates,
    create_output_grid,
    detect_projection,
    interpolate_data,
    setup_coordinate_transform,
)
from src.utils.logging import log_print

warnings.filterwarnings("ignore", category=xr.SerializationWarning)
warnings.filterwarnings("ignore", message=".*multiple fill values.*")


# * --- time converter --- #


def utc_to_local(filename: str, local_tz: str = "Asia/Shanghai"):
    utc_time_str = filename.split(".")[-2] + filename.split(".")[-1].replace(".nc", "")
    utc_time = datetime.datetime.strptime(utc_time_str, "%Y%m%d%H%M%S")

    # 标记为UTC时间
    utc_time = utc_time.replace(tzinfo=pytz.UTC)

    # 转换为东八区（北京时间）
    beijing_time = utc_time.astimezone(pytz.timezone(local_tz))

    return beijing_time.strftime("%Y%m%d.%H%M%S")


# * --- Satellite reader --- #


def satellite_read(
    file: str,
    grid_width: int | None = None,
    grid_height: int | None = None,
    bands_range: tuple[int, int] = (9, 17),
    target_proj: str = "epsg:4326",
    force_source_proj: Optional[str] = None,
    interpolation_method: str = "regular_grid",
    stack: bool = True,
    crop_bounds_latlon: Optional[Tuple[float, float, float, float]] = None,  # (min_lon, max_lon, min_lat, max_lat)
):
    """
    读取卫星数据并转换到标准地图格式，可选择按经纬度裁切。

    interpolation_method选项:
    - "fast_bilinear": 使用skimage的快速双线性插值（最快）
    - "regular_grid": 使用RegularGridInterpolator（中等速度，高精度）
    - "nearest": 使用KDTree最近邻（快速）
    - "griddata": 使用scipy的griddata（最慢但最灵活）

    Args:
        crop_bounds_latlon: 可选参数，格式为 (min_lon, max_lon, min_lat, max_lat)，用于裁切输出结果。
                           裁切基于最终的经纬度坐标。
    """
    assert os.path.exists(file), f"File {file} does not exist."

    try:
        data = xr.open_dataset(file)
    except OSError as e:
        if "HDF error" in str(e):
            log_print(f"Corrupted NetCDF file: {file}, skipping.")
            return {"mapped_bands": None}
        else:
            raise
    latitudes = data["latitude"]
    longitudes = data["longitude"]

    # 检测源投影
    if force_source_proj:
        source_proj = force_source_proj
    else:
        source_proj = detect_projection(longitudes, latitudes)

    # 设置坐标转换和网格
    transformer, bounds, is_regular_grid = setup_coordinate_transform(
        source_proj, target_proj, longitudes.values, latitudes.values
    )

    # 设置输出网格
    # Store original requested grid_width and grid_height before potential modification by cropping
    requested_grid_width, requested_grid_height = grid_width or 1000, grid_height or 800
    x_mesh_regular, y_mesh_regular = create_output_grid(bounds, requested_grid_width, requested_grid_height)

    # 处理原始坐标网格
    if is_regular_grid:
        original_lon_mesh, original_lat_mesh = np.meshgrid(longitudes.values, latitudes.values)
    else:
        original_lon_mesh = longitudes.values
        original_lat_mesh = latitudes.values

    bands = [f"tbb_{i:02d}" for i in range(bands_range[0], bands_range[1])]

    # Handle case where no bands are selected
    if not bands:
        log_print("No bands selected based on bands_range.", "warning")
        empty_mapped_bands_dict = {}
        empty_mapped_bands_stacked = np.empty((0, requested_grid_height, requested_grid_width))

        final_mapped_bands = empty_mapped_bands_stacked if stack else empty_mapped_bands_dict
        final_x_mesh = x_mesh_regular
        final_y_mesh = y_mesh_regular
        final_lon_mesh, final_lat_mesh = create_display_coordinates(x_mesh_regular, y_mesh_regular, target_proj)
        final_bounds_target = bounds
        final_grid_width = requested_grid_width
        final_grid_height = requested_grid_height

        # If crop_bounds_latlon is also specified with no bands, result is still empty
        if crop_bounds_latlon:
            log_print("Cropping specified but no bands to crop.", "warning")
            final_mapped_bands = np.empty((0, 0, 0)) if stack else {}
            final_x_mesh, final_y_mesh = np.empty((0, 0)), np.empty((0, 0))
            final_lon_mesh, final_lat_mesh = np.empty((0, 0)), np.empty((0, 0))
            final_grid_width, final_grid_height = 0, 0
            final_bounds_target = (np.nan, np.nan, np.nan, np.nan)

    else:
        processed_mapped_bands_dict = {}
        for band in bands:
            # log_print(
            #     f"Processing {band} with {interpolation_method} method...", "debug"
            # )
            if band not in data:
                log_print(f"file: {file} - Band {band} not found in the dataset. Skipping.", "warning")
                continue
            original_band_data = data[band].values
            original_band_data = np.nan_to_num(original_band_data, nan=0.0)

            mapped_band_data = interpolate_data(
                original_band_data,
                (original_lon_mesh, original_lat_mesh),
                (x_mesh_regular, y_mesh_regular),
                transformer,
                interpolation_method,
                is_regular_grid,
            )
            processed_mapped_bands_dict[band] = mapped_band_data

        if not processed_mapped_bands_dict:  # All bands were skipped
            log_print("No bands were successfully processed.", "warning")
            # Similar empty handling as above for no bands selected
            empty_mapped_bands_stacked = np.empty((0, requested_grid_height, requested_grid_width))
            final_mapped_bands = empty_mapped_bands_stacked if stack else {}
            final_x_mesh = x_mesh_regular
            final_y_mesh = y_mesh_regular
            final_lon_mesh, final_lat_mesh = create_display_coordinates(x_mesh_regular, y_mesh_regular, target_proj)
            final_bounds_target = bounds
            final_grid_width = requested_grid_width
            final_grid_height = requested_grid_height
            if crop_bounds_latlon:  # Crop on already empty data
                final_mapped_bands = np.empty((0, 0, 0)) if stack else {}
                final_x_mesh, final_y_mesh = np.empty((0, 0)), np.empty((0, 0))
                final_lon_mesh, final_lat_mesh = np.empty((0, 0)), np.empty((0, 0))
                final_grid_width, final_grid_height = 0, 0
                final_bounds_target = (np.nan, np.nan, np.nan, np.nan)

        else:
            if stack:
                initial_mapped_bands = np.stack(list(processed_mapped_bands_dict.values()), axis=0)
            else:
                initial_mapped_bands = processed_mapped_bands_dict

            # Create display coordinates for the full interpolated grid
            initial_lon_mesh_display, initial_lat_mesh_display = create_display_coordinates(
                x_mesh_regular, y_mesh_regular, target_proj
            )

            # Initialize variables that might be changed by cropping
            final_mapped_bands = initial_mapped_bands
            final_x_mesh = x_mesh_regular
            final_y_mesh = y_mesh_regular
            final_lon_mesh = initial_lon_mesh_display
            final_lat_mesh = initial_lat_mesh_display
            final_bounds_target = bounds
            final_grid_width = requested_grid_width
            final_grid_height = requested_grid_height

            if crop_bounds_latlon:
                min_lon_crop, max_lon_crop, min_lat_crop, max_lat_crop = crop_bounds_latlon

                # Create mask using the display coordinates (which are in EPSG:4326)
                crop_mask = (
                    (initial_lon_mesh_display >= min_lon_crop)
                    & (initial_lon_mesh_display <= max_lon_crop)
                    & (initial_lat_mesh_display >= min_lat_crop)
                    & (initial_lat_mesh_display <= max_lat_crop)
                )

                if np.any(crop_mask):
                    rows_to_keep = np.any(crop_mask, axis=1)
                    cols_to_keep = np.any(crop_mask, axis=0)

                    if np.any(rows_to_keep) and np.any(cols_to_keep):
                        row_indices = np.where(rows_to_keep)[0]
                        col_indices = np.where(cols_to_keep)[0]

                        row_slice = slice(row_indices[0], row_indices[-1] + 1)
                        col_slice = slice(col_indices[0], col_indices[-1] + 1)

                        final_lon_mesh = initial_lon_mesh_display[row_slice, col_slice]
                        final_lat_mesh = initial_lat_mesh_display[row_slice, col_slice]
                        final_x_mesh = x_mesh_regular[row_slice, col_slice]
                        final_y_mesh = y_mesh_regular[row_slice, col_slice]

                        if stack:
                            final_mapped_bands = initial_mapped_bands[:, row_slice, col_slice]
                        else:
                            final_mapped_bands = {
                                band_name: data_array[row_slice, col_slice]
                                for band_name, data_array in initial_mapped_bands.items()
                            }

                        final_grid_height = final_lon_mesh.shape[0]
                        final_grid_width = final_lon_mesh.shape[1]

                        if final_x_mesh.size > 0:
                            final_bounds_target = (
                                float(final_x_mesh.min()),
                                float(final_x_mesh.max()),
                                float(final_y_mesh.min()),
                                float(final_y_mesh.max()),
                            )
                        else:
                            final_bounds_target = (np.nan, np.nan, np.nan, np.nan)
                    else:
                        log_print(
                            "Crop area resulted in no valid data rows or columns after masking.",
                            "warning",
                        )
                        empty_shape_2d = (0, 0)
                        num_bands_original = (
                            initial_mapped_bands.shape[0]
                            if stack and hasattr(initial_mapped_bands, "ndim") and initial_mapped_bands.ndim == 3
                            else 0
                        )
                        final_mapped_bands = (
                            np.empty((num_bands_original, 0, 0))
                            if stack
                            else {bn: np.empty(empty_shape_2d) for bn in initial_mapped_bands}
                        )
                        final_x_mesh, final_y_mesh = (
                            np.empty(empty_shape_2d),
                            np.empty(empty_shape_2d),
                        )
                        final_lon_mesh, final_lat_mesh = (
                            np.empty(empty_shape_2d),
                            np.empty(empty_shape_2d),
                        )
                        final_grid_width, final_grid_height = 0, 0
                        final_bounds_target = (np.nan, np.nan, np.nan, np.nan)
                else:
                    log_print(
                        "Crop area is outside the data extent or resulted in an all-False mask.",
                        "warning",
                    )
                    empty_shape_2d = (0, 0)
                    num_bands_original = (
                        initial_mapped_bands.shape[0]
                        if stack and hasattr(initial_mapped_bands, "ndim") and initial_mapped_bands.ndim == 3
                        else 0
                    )
                    final_mapped_bands = (
                        np.empty((num_bands_original, 0, 0))
                        if stack
                        else {bn: np.empty(empty_shape_2d) for bn in initial_mapped_bands}
                    )
                    final_x_mesh, final_y_mesh = (
                        np.empty(empty_shape_2d),
                        np.empty(empty_shape_2d),
                    )
                    final_lon_mesh, final_lat_mesh = (
                        np.empty(empty_shape_2d),
                        np.empty(empty_shape_2d),
                    )
                    final_grid_width, final_grid_height = 0, 0
                    final_bounds_target = (np.nan, np.nan, np.nan, np.nan)

    # 获取原始边界 (始终是源数据的完整边界)
    x_min_src = float(longitudes.min())
    x_max_src = float(longitudes.max())
    y_min_src = float(latitudes.min())
    y_max_src = float(latitudes.max())

    return dict(
        data=data,  # 原始 xarray Dataset
        mapped_bands=final_mapped_bands,
        x_mesh=final_x_mesh,  # 目标投影下的X坐标 (可能已裁切)
        y_mesh=final_y_mesh,  # 目标投影下的Y坐标 (可能已裁切)
        lon_mesh=final_lon_mesh,  # EPSG:4326经度 (可能已裁切)
        lat_mesh=final_lat_mesh,  # EPSG:4326纬度 (可能已裁切)
        bounds_target=final_bounds_target,  # 目标投影下 (可能已裁切) 数据的边界
        bounds_lonlat=(
            x_min_src,
            x_max_src,
            y_min_src,
            y_max_src,
        ),  # 源数据的原始经纬度边界
        grid_width=final_grid_width,  # (可能已裁切后) 的网格宽度
        grid_height=final_grid_height,  # (可能已裁切后) 的网格高度
        source_proj=source_proj,
        target_proj=target_proj,
    )


# * --- Radar reader --- #


def radar_read(
    file: str,
    grid_width: int | None = None,
    grid_height: int | None = None,
    variable_name: str = "var",
    target_proj: str = "epsg:4326",
    force_source_proj: Optional[str] = None,
    interpolation_method: str = "regular_grid",
    crop_bounds_latlon: Optional[Tuple[float, float, float, float]] = None,
):
    """读取雷达数据并转换到标准地图格式，可选择按经纬度裁切。"""
    assert os.path.exists(file), f"File {file} does not exist."
    data = xr.open_dataset(file)

    # 坐标识别
    if "coord" in data:
        if "lat" in data.coords and "lon" in data.coords:
            latitudes = data["lat"]
            longitudes = data["lon"]
        elif "lat" in data.coords and "lng" in data.coords:
            latitudes = data["lat"]
            longitudes = data["lng"]
        elif "latitude" in data.coords and "longitude" in data.coords:
            latitudes = data["latitude"]
            longitudes = data["longitude"]
        else:
            raise ValueError("Cannot find latitude/longitude coordinates in radar data")
    elif "lat" in data and "lon" in data:
        latitudes = data["lat"]
        longitudes = data["lon"]
    else:
        raise ValueError(f"Cannot find latitude/longitude coordinates in radar data: {data}")

    # log_print(f"Radar data shape: {data[variable_name].shape}", "debug")
    # log_print(f"Coordinates: lat {latitudes.shape}, lon {longitudes.shape}", "debug")

    if force_source_proj:
        source_proj = force_source_proj
    else:
        source_proj = detect_projection(longitudes, latitudes)

    # 先建立 transformer，这里 transformer 可能为 None（即源投影即目标投影）
    transformer, _, is_regular_grid = setup_coordinate_transform(
        source_proj, target_proj, longitudes.values, latitudes.values
    )

    requested_grid_width = grid_width or data[variable_name].shape[-1]
    requested_grid_height = grid_height or data[variable_name].shape[-2]

    # 构建原始网格
    if is_regular_grid:
        original_lon_mesh, original_lat_mesh = np.meshgrid(longitudes.values, latitudes.values)
    else:
        original_lon_mesh = longitudes.values
        original_lat_mesh = latitudes.values

    original_data_values = data[variable_name].values
    original_data_values = np.nan_to_num(original_data_values, nan=0.0)

    # ====== 提前裁剪原始数据 ======
    if crop_bounds_latlon:
        min_lon_crop, max_lon_crop, min_lat_crop, max_lat_crop = crop_bounds_latlon
        crop_mask = (
            (original_lon_mesh >= min_lon_crop)
            & (original_lon_mesh <= max_lon_crop)
            & (original_lat_mesh >= min_lat_crop)
            & (original_lat_mesh <= max_lat_crop)
        )
        if np.any(crop_mask):
            rows_to_keep = np.any(crop_mask, axis=1)
            cols_to_keep = np.any(crop_mask, axis=0)
            if np.any(rows_to_keep) and np.any(cols_to_keep):
                row_indices = np.where(rows_to_keep)[0]
                col_indices = np.where(cols_to_keep)[0]
                row_slice = slice(row_indices[0], row_indices[-1] + 1)
                col_slice = slice(col_indices[0], col_indices[-1] + 1)
                original_lon_mesh = original_lon_mesh[row_slice, col_slice]
                original_lat_mesh = original_lat_mesh[row_slice, col_slice]
                original_data_values = original_data_values[row_slice, col_slice]
            else:
                log_print(
                    "Radar crop area resulted in no valid data rows or columns after masking.",
                    "warning",
                )
                return _empty_radar_result(data, longitudes, latitudes, source_proj, target_proj, variable_name)
        else:
            log_print(
                "Radar crop area is outside the data extent or resulted in an all-False mask.",
                "warning",
            )
            return _empty_radar_result(data, longitudes, latitudes, source_proj, target_proj, variable_name)
    # ====== 裁剪结束 ======

    # ====== 更新插值边界 ======
    if original_lon_mesh.size > 0:
        if transformer is not None:
            x_flat, y_flat = transformer.transform(original_lon_mesh.flatten(), original_lat_mesh.flatten())
            bounds = (
                float(np.min(x_flat)),
                float(np.max(x_flat)),
                float(np.min(y_flat)),
                float(np.max(y_flat)),
            )
        else:
            # 当 transformer 为 None 时，说明源与目标投影一致
            bounds = (
                float(np.min(original_lon_mesh)),
                float(np.max(original_lon_mesh)),
                float(np.min(original_lat_mesh)),
                float(np.max(original_lat_mesh)),
            )
    else:
        return _empty_radar_result(data, longitudes, latitudes, source_proj, target_proj, variable_name)

    # ====== 确定插值网格大小 ======
    if grid_width is None and grid_height is None:
        out_grid_width = original_data_values.shape[-1]
        out_grid_height = original_data_values.shape[-2]
    else:
        out_grid_width = requested_grid_width
        out_grid_height = requested_grid_height

    # log_print(f"裁剪后数据形状: {original_data_values.shape}", "debug")
    # log_print(f"插值网格边界: {bounds}", "debug")

    # 创建输出网格
    x_mesh_regular_full, y_mesh_regular_full = create_output_grid(bounds, out_grid_width, out_grid_height)

    # ====== 插值 ======
    initial_mapped_data = interpolate_data(
        original_data_values,
        (original_lon_mesh, original_lat_mesh),
        (x_mesh_regular_full, y_mesh_regular_full),
        transformer,
        interpolation_method,
        is_regular_grid,
    )

    initial_lon_mesh_display, initial_lat_mesh_display = create_display_coordinates(
        x_mesh_regular_full, y_mesh_regular_full, target_proj
    )

    # 获取原始边界（未裁剪）
    x_min_src = float(longitudes.min())
    x_max_src = float(longitudes.max())
    y_min_src = float(latitudes.min())
    y_max_src = float(latitudes.max())

    return dict(
        data=data,
        mapped_data=initial_mapped_data,
        x_mesh=x_mesh_regular_full,
        y_mesh=y_mesh_regular_full,
        lon_mesh=initial_lon_mesh_display,
        lat_mesh=initial_lat_mesh_display,
        bounds_target=bounds,
        bounds_lonlat=(x_min_src, x_max_src, y_min_src, y_max_src),
        grid_width=out_grid_width,
        grid_height=out_grid_height,
        source_proj=source_proj,
        target_proj=target_proj,
        variable_name=variable_name,
    )


def _empty_radar_result(data, longitudes, latitudes, source_proj, target_proj, variable_name):
    empty_shape_2d = (0, 0)
    return dict(
        data=data,
        mapped_data=np.empty(empty_shape_2d),
        x_mesh=np.empty(empty_shape_2d),
        y_mesh=np.empty(empty_shape_2d),
        lon_mesh=np.empty(empty_shape_2d),
        lat_mesh=np.empty(empty_shape_2d),
        bounds_target=(np.nan, np.nan, np.nan, np.nan),
        bounds_lonlat=(
            float(longitudes.min()),
            float(longitudes.max()),
            float(latitudes.min()),
            float(latitudes.max()),
        ),
        grid_width=0,
        grid_height=0,
        source_proj=source_proj,
        target_proj=target_proj,
        variable_name=variable_name,
    )


def radar_read_origin(
    file: str,
    grid_width: int | None = None,
    grid_height: int | None = None,
    variable_name: str = "var",
    target_proj: str = "epsg:4326",
    force_source_proj: Optional[str] = None,
    interpolation_method: str = "regular_grid",
    crop_bounds_latlon: Optional[Tuple[float, float, float, float]] = None,  # (min_lon, max_lon, min_lat, max_lat)
):
    """
    读取雷达数据并转换到标准地图格式，可选择按经纬度裁切。

    Args:
        file: 雷达数据文件路径
        grid_width: 输出网格宽度
        grid_height: 输出网格高度
        variable_name: 数据变量名称 (默认 "var")
        target_proj: 目标投影
        force_source_proj: 强制指定源投影
        interpolation_method: 插值方法
        crop_bounds_latlon: 可选参数，格式为 (min_lon, max_lon, min_lat, max_lat)，用于裁切输出结果。
                           裁切基于最终的经纬度坐标。
    """
    assert os.path.exists(file), f"File {file} does not exist."

    data = xr.open_dataset(file)

    # 雷达数据可能有不同的坐标命名方式
    if "lat" in data.coords and "lon" in data.coords:
        latitudes = data["lat"]
        longitudes = data["lon"]
    elif "lat" in data.coords and "lng" in data.coords:
        latitudes = data["lat"]
        longitudes = data["lng"]
    elif "latitude" in data.coords and "longitude" in data.coords:
        latitudes = data["latitude"]
        longitudes = data["longitude"]
    else:
        raise ValueError("Cannot find latitude/longitude coordinates in radar data")

    log_print(f"Radar data shape: {data[variable_name].shape}", "debug")
    log_print(f"Coordinates: lat {latitudes.shape}, lon {longitudes.shape}", "debug")

    # 检测源投影
    if force_source_proj:
        source_proj = force_source_proj
    else:
        source_proj = detect_projection(longitudes, latitudes)

    # 设置坐标转换和网格
    transformer, bounds, is_regular_grid = setup_coordinate_transform(
        source_proj, target_proj, longitudes.values, latitudes.values
    )

    # Store original requested grid_width and grid_height
    requested_grid_width, requested_grid_height = (
        grid_width or data[variable_name].shape[-1],
        grid_height or data[variable_name].shape[-2],
    )

    # 处理原始坐标网格
    if is_regular_grid:
        # 检查雷达数据的维度结构
        original_lon_mesh, original_lat_mesh = np.meshgrid(longitudes.values, latitudes.values)
    else:
        original_lon_mesh = longitudes.values
        original_lat_mesh = latitudes.values

    log_print(f"Processing {variable_name} with {interpolation_method} method...", "debug")

    # 获取雷达数据
    original_data_values = data[variable_name].values

    # 设置输出网格
    # If grid_width or grid_height are not provided, use original dimensions if no transformation,
    # otherwise default to 1000x800 if transformation occurs.
    if grid_width is None and grid_height is None:
        if transformer is None and source_proj == target_proj:  # No transformation, use original shape
            out_grid_width, out_grid_height = (
                original_data_values.shape[-1],
                original_data_values.shape[-2],
            )
        else:  # Transformation occurs, or target_proj is different, use default or provided
            out_grid_width, out_grid_height = (
                requested_grid_width,
                requested_grid_height,
            )
    else:
        out_grid_width, out_grid_height = requested_grid_width, requested_grid_height

    x_mesh_regular_full, y_mesh_regular_full = create_output_grid(bounds, out_grid_width, out_grid_height)

    # 使用通用插值函数
    initial_mapped_data = interpolate_data(
        original_data_values,
        (original_lon_mesh, original_lat_mesh),
        (x_mesh_regular_full, y_mesh_regular_full),
        transformer,
        interpolation_method,
        is_regular_grid,
    )

    # 创建显示坐标
    initial_lon_mesh_display, initial_lat_mesh_display = create_display_coordinates(
        x_mesh_regular_full, y_mesh_regular_full, target_proj
    )

    # Initialize variables that might be changed by cropping
    final_mapped_data = initial_mapped_data
    final_x_mesh = x_mesh_regular_full
    final_y_mesh = y_mesh_regular_full
    final_lon_mesh = initial_lon_mesh_display
    final_lat_mesh = initial_lat_mesh_display
    final_bounds_target = bounds
    final_grid_width = out_grid_width
    final_grid_height = out_grid_height

    if crop_bounds_latlon:
        min_lon_crop, max_lon_crop, min_lat_crop, max_lat_crop = crop_bounds_latlon

        crop_mask = (
            (initial_lon_mesh_display >= min_lon_crop)
            & (initial_lon_mesh_display <= max_lon_crop)
            & (initial_lat_mesh_display >= min_lat_crop)
            & (initial_lat_mesh_display <= max_lat_crop)
        )

        if np.any(crop_mask):
            rows_to_keep = np.any(crop_mask, axis=0)  # For radar, typically (height, width)
            cols_to_keep = np.any(crop_mask, axis=1)  # For radar, typically (height, width)

            # Correction: For 2D arrays (height, width), mask.any(axis=1) gives rows, mask.any(axis=0) gives columns
            rows_to_keep = np.any(crop_mask, axis=1)
            cols_to_keep = np.any(crop_mask, axis=0)

            if np.any(rows_to_keep) and np.any(cols_to_keep):
                row_indices = np.where(rows_to_keep)[0]
                col_indices = np.where(cols_to_keep)[0]

                row_slice = slice(row_indices[0], row_indices[-1] + 1)
                col_slice = slice(col_indices[0], col_indices[-1] + 1)

                final_lon_mesh = initial_lon_mesh_display[row_slice, col_slice]
                final_lat_mesh = initial_lat_mesh_display[row_slice, col_slice]
                final_x_mesh = x_mesh_regular_full[row_slice, col_slice]
                final_y_mesh = y_mesh_regular_full[row_slice, col_slice]
                final_mapped_data = initial_mapped_data[row_slice, col_slice]

                final_grid_height = final_lon_mesh.shape[0]
                final_grid_width = final_lon_mesh.shape[1]

                if final_x_mesh.size > 0:
                    final_bounds_target = (
                        float(final_x_mesh.min()),
                        float(final_x_mesh.max()),
                        float(final_y_mesh.min()),
                        float(final_y_mesh.max()),
                    )
                else:
                    final_bounds_target = (np.nan, np.nan, np.nan, np.nan)
            else:
                log_print(
                    "Radar crop area resulted in no valid data rows or columns after masking.",
                    "warning",
                )
                empty_shape_2d = (0, 0)
                final_mapped_data = np.empty(empty_shape_2d)
                final_x_mesh, final_y_mesh = (
                    np.empty(empty_shape_2d),
                    np.empty(empty_shape_2d),
                )
                final_lon_mesh, final_lat_mesh = (
                    np.empty(empty_shape_2d),
                    np.empty(empty_shape_2d),
                )
                final_grid_width, final_grid_height = 0, 0
                final_bounds_target = (np.nan, np.nan, np.nan, np.nan)
        else:
            log_print(
                "Radar crop area is outside the data extent or resulted in an all-False mask.",
                "warning",
            )
            empty_shape_2d = (0, 0)
            final_mapped_data = np.empty(empty_shape_2d)
            final_x_mesh, final_y_mesh = (
                np.empty(empty_shape_2d),
                np.empty(empty_shape_2d),
            )
            final_lon_mesh, final_lat_mesh = (
                np.empty(empty_shape_2d),
                np.empty(empty_shape_2d),
            )
            final_grid_width, final_grid_height = 0, 0
            final_bounds_target = (np.nan, np.nan, np.nan, np.nan)

    # 获取原始边界
    x_min_src = float(longitudes.min())
    x_max_src = float(longitudes.max())
    y_min_src = float(latitudes.min())
    y_max_src = float(latitudes.max())

    return dict(
        data=data,
        mapped_data=final_mapped_data,
        x_mesh=final_x_mesh,
        y_mesh=final_y_mesh,
        lon_mesh=final_lon_mesh,
        lat_mesh=final_lat_mesh,
        bounds_target=final_bounds_target,
        bounds_lonlat=(x_min_src, x_max_src, y_min_src, y_max_src),
        grid_width=final_grid_width,
        grid_height=final_grid_height,
        source_proj=source_proj,
        target_proj=target_proj,
        variable_name=variable_name,
    )


# * --- testers --- #


def test_satellite_read():
    import time

    # 测试不同方法的性能
    methods = ["fast_bilinear", "regular_grid", "nearest"]

    for method in methods:
        print(f"\n=== Testing satellite {method} method ===")
        start_time = time.time()

        result = satellite_read(
            file="data/satellite/202305/20230501_0800.nc",
            target_proj="epsg:4326",  # 不做投影转换，更快
            grid_width=512,
            grid_height=512,
            interpolation_method=method,
            crop_bounds_latlon=(97.0, 109.0, 26.0, 35.0),
        )

        end_time = time.time()
        print(f"Processing time: {end_time - start_time:.2f} seconds")

        # 检查结果
        bands = [f"tbb_{i:02d}" for i in range(9, 17)]
        mapped_bands = result["mapped_bands"]
        print(f"Valid data points: {(~np.isnan(mapped_bands)).sum()}")


def test_radar_read():
    import time

    # 测试雷达数据读取
    methods = ["regular_grid", "nearest"]
    # Define a crop area for testing
    test_crop_bounds = (97.0, 109.0, 26.0, 35.0)  # Example: Sichuan Basin area

    for method in methods:
        print(f"\n=== Testing radar {method} method ===")
        start_time = time.time()

        result = radar_read(
            file="data/radar/202305/20230501/ACHN.QREF000.20230430.160000.nc",
            target_proj="epsg:4326",
            grid_width=None,
            grid_height=None,
            interpolation_method=method,
            crop_bounds_latlon=test_crop_bounds,
        )

        end_time = time.time()
        print(f"Processing time: {end_time - start_time:.2f} seconds")

        # 检查结果
        mapped_data = result["mapped_data"]
        print(f"Output shape: {mapped_data.shape}")
        print(f"Lon mesh shape: {result['lon_mesh'].shape}")
        print(f"Lat mesh shape: {result['lat_mesh'].shape}")
        print(f"Grid width/height: {result['grid_width']}/{result['grid_height']}")
        if mapped_data.size > 0:
            print(f"Valid data points: {(~np.isnan(mapped_data)).sum()}")
            print(f"Data range: [{np.nanmin(mapped_data):.2f}, {np.nanmax(mapped_data):.2f}]")
            print(f"Cropped Lon range: [{np.nanmin(result['lon_mesh']):.2f}, {np.nanmax(result['lon_mesh']):.2f}]")
            print(f"Cropped Lat range: [{np.nanmin(result['lat_mesh']):.2f}, {np.nanmax(result['lat_mesh']):.2f}]")
            print(f"Cropped bounds_target: {result['bounds_target']}")
        else:
            print("No data in cropped region.")


if __name__ == "__main__":
    # test_satellite_read()
    test_radar_read()
