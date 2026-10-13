import os
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
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
):
    """
    读取卫星数据并转换到标准地图格式

    interpolation_method选项:
    - "fast_bilinear": 使用skimage的快速双线性插值（最快）
    - "regular_grid": 使用RegularGridInterpolator（中等速度，高精度）
    - "nearest": 使用KDTree最近邻（快速）
    - "griddata": 使用scipy的griddata（最慢但最灵活）
    """
    assert os.path.exists(file), f"File {file} does not exist."

    data = xr.open_dataset(file)
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
    grid_width, grid_height = grid_width or 1000, grid_height or 800
    x_mesh_regular, y_mesh_regular = create_output_grid(bounds, grid_width, grid_height)

    # 处理原始坐标网格
    if is_regular_grid:
        original_lon_mesh, original_lat_mesh = np.meshgrid(longitudes.values, latitudes.values)
    else:
        original_lon_mesh = longitudes.values
        original_lat_mesh = latitudes.values

    bands = [f"tbb_{i:02d}" for i in range(bands_range[0], bands_range[1])]
    mapped_bands = {}

    for band in bands:
        log_print(f"Processing {band} with {interpolation_method} method...", "debug")

        original_data = data[band].values

        # 使用通用插值函数
        mapped_data = interpolate_data(
            original_data,
            (original_lon_mesh, original_lat_mesh),
            (x_mesh_regular, y_mesh_regular),
            transformer,
            interpolation_method,
            is_regular_grid,
        )

        mapped_bands[band] = mapped_data

    if stack:
        mapped_bands = np.stack(list(mapped_bands.values()), axis=0)

    # 创建显示坐标
    lon_mesh_display, lat_mesh_display = create_display_coordinates(
        x_mesh_regular, y_mesh_regular, target_proj
    )

    # 获取原始边界
    x_min_src = float(longitudes.min())
    x_max_src = float(longitudes.max())
    y_min_src = float(latitudes.min())
    y_max_src = float(latitudes.max())

    return dict(
        data=data,
        mapped_bands=mapped_bands,
        x_mesh=x_mesh_regular,
        y_mesh=y_mesh_regular,
        lon_mesh=lon_mesh_display,
        lat_mesh=lat_mesh_display,
        bounds_target=bounds,
        bounds_lonlat=(x_min_src, x_max_src, y_min_src, y_max_src),
        grid_width=grid_width,
        grid_height=grid_height,
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
):
    """
    读取雷达数据并转换到标准地图格式

    Args:
        file: 雷达数据文件路径
        grid_width: 输出网格宽度
        grid_height: 输出网格高度
        variable_name: 数据变量名称 (默认 "var")
        target_proj: 目标投影
        force_source_proj: 强制指定源投影
        interpolation_method: 插值方法
    """
    assert os.path.exists(file), f"File {file} does not exist."

    data = xr.open_dataset(file)

    # 雷达数据可能有不同的坐标命名方式
    if "lat" in data.coords and "lon" in data.coords:
        latitudes = data["lat"]
        longitudes = data["lon"]
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

    # 设置输出网格
    grid_width, grid_height = grid_width or 1000, grid_height or 800
    x_mesh_regular, y_mesh_regular = create_output_grid(bounds, grid_width, grid_height)

    # 处理原始坐标网格
    if is_regular_grid:
        # 注意雷达数据可能使用 'lng' 而不是 'lon'
        if data[variable_name].dims == ("lat", "lng"):
            # 创建正确的坐标网格
            original_lon_mesh, original_lat_mesh = np.meshgrid(longitudes.values, latitudes.values)
        else:
            original_lon_mesh, original_lat_mesh = np.meshgrid(longitudes.values, latitudes.values)
    else:
        original_lon_mesh = longitudes.values
        original_lat_mesh = latitudes.values

    log_print(f"Processing {variable_name} with {interpolation_method} method...", "debug")

    # 获取雷达数据
    original_data = data[variable_name].values

    # 使用通用插值函数
    mapped_data = interpolate_data(
        original_data,
        (original_lon_mesh, original_lat_mesh),
        (x_mesh_regular, y_mesh_regular),
        transformer,
        interpolation_method,
        is_regular_grid,
    )

    # 创建显示坐标
    lon_mesh_display, lat_mesh_display = create_display_coordinates(
        x_mesh_regular, y_mesh_regular, target_proj
    )

    # 获取原始边界
    x_min_src = float(longitudes.min())
    x_max_src = float(longitudes.max())
    y_min_src = float(latitudes.min())
    y_max_src = float(latitudes.max())

    return dict(
        data=data,
        mapped_data=mapped_data,
        x_mesh=x_mesh_regular,
        y_mesh=y_mesh_regular,
        lon_mesh=lon_mesh_display,
        lat_mesh=lat_mesh_display,
        bounds_target=bounds,
        bounds_lonlat=(x_min_src, x_max_src, y_min_src, y_max_src),
        grid_width=grid_width,
        grid_height=grid_height,
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
            grid_width=700,
            grid_height=500,
            interpolation_method=method,
        )

        end_time = time.time()
        print(f"Processing time: {end_time - start_time:.2f} seconds")

        # 检查结果
        mapped_bands = result["mapped_bands"]
        print(f"Output shape: {mapped_bands.shape}")
        print(f"Valid data points: {(~np.isnan(mapped_bands)).sum()}")


def test_radar_read():
    import time

    # 测试雷达数据读取
    methods = ["regular_grid", "nearest"]

    for method in methods:
        print(f"\n=== Testing radar {method} method ===")
        start_time = time.time()

        result = radar_read(
            file="data/radar/202305/20230501/ACHN.QREF000.20230430.160000.nc",
            target_proj="epsg:4326",
            grid_width=500,
            grid_height=400,
            interpolation_method=method,
        )

        end_time = time.time()
        print(f"Processing time: {end_time - start_time:.2f} seconds")

        # 检查结果
        mapped_data = result["mapped_data"]
        print(f"Output shape: {mapped_data.shape}")
        print(f"Valid data points: {(~np.isnan(mapped_data)).sum()}")
        print(f"Data range: [{np.nanmin(mapped_data):.2f}, {np.nanmax(mapped_data):.2f}]")


if __name__ == "__main__":
    # test_satellite_read()
    test_radar_read()
