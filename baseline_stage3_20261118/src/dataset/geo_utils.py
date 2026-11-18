"""
地理坐标转换和插值的通用工具函数
"""

import datetime
import warnings
from re import L
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytz
import xarray as xr
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from skimage.transform import resize

from src.utils.logging import log_print

warnings.filterwarnings("ignore", category=xr.SerializationWarning)
warnings.filterwarnings("ignore", message=".*multiple fill values.*")

# * --- time converter --- #


def file_name_to_strptime(filename: str):
    if "nc" in filename:
        strptime = "".join(filename.split(".")[-3:-1])
    else:
        if "." in filename:
            strptime = filename.replace(".", "")
        else:
            strptime = filename

    return strptime


def utc_to_local(filename_or_timestring: str, local_tz: str = "Asia/Shanghai"):
    # inputs:
    # data/radar/202306/20230601/ACHN.QREF000.20230531.160000.nc
    # 20230531.160000.nc
    # 20230531.160000

    utc_time_str = file_name_to_strptime(filename_or_timestring)

    # 解析UTC时间字符串
    utc_time = datetime.datetime.strptime(utc_time_str, "%Y%m%d%H%M%S")

    # 标记为UTC时间
    utc_time = utc_time.replace(tzinfo=pytz.UTC)

    # 转换为东八区（北京时间）
    beijing_time = utc_time.astimezone(pytz.timezone(local_tz))

    return beijing_time.strftime("%Y%m%d.%H%M%S")


def any_modaility_time_to_local(
    any_modality_time: str,
    modality: str = "radar",
    local_tz: str = "Asia/Shanghai",
) -> datetime.datetime:
    """
    将特定模态的时间字符串转换为本地时间
    """
    if modality == "radar":
        # 雷达数据格式：20230531.160000 utc时间，需要转换到东八区
        # 例如：20230531.160000
        t = datetime.datetime.strptime(any_modality_time, "%Y%m%d.%H%M%S")
        # 将UTC时间转换为本地时间
        utc_time = t.replace(tzinfo=pytz.UTC)
        # 转换为东八区（北京时间）
        local_time = utc_time.astimezone(pytz.timezone(local_tz))
        return local_time.replace(tzinfo=None)
    elif modality == "satellite":
        # 卫星数据格式：20230531_1600
        local_time = datetime.datetime.strptime(any_modality_time, "%Y%m%d_%H%M")
    elif modality == "rain":
        local_time = datetime.datetime.strptime(any_modality_time, "%Y-%m-%d %H:%M:%S")
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    return local_time


def local_to_any_modality_time(
    local_time: datetime.datetime,
    modality: str = "radar",
    local_tz: str = "Asia/Shanghai",
):
    """
    将本地时间转换为特定模态的时间字符串格式
    """
    if modality == "radar":
        # 雷达数据格式：20230531.160000
        # 雷达是utc时间
        # 将本地时间转换为UTC时间
        utc_time = local_time.astimezone(pytz.UTC)
        # 注意：这里的local_time已经是东八区时间了，所以需要转换为UTC时间
        return utc_time.strftime("%Y%m%d.%H%M%S")
    elif modality == "satellite":
        # 卫星数据格式：20230531_1600
        return local_time.strftime("%Y%m%d_%H%M")
    elif modality == "rain":
        return local_time.strftime("%Y-%m-%d %H:%M:%S")
    else:
        raise ValueError(f"Unsupported modality: {modality}")


def local_to_any_modality_datetime(
    local_time: datetime.datetime,
    modality: str = "radar",
    local_tz_name: str = "Asia/Shanghai",
) -> datetime.datetime:
    # 统一处理输入 local_time，确保它是带时区信息的
    target_local_tz = pytz.timezone(local_tz_name)
    if local_time.tzinfo is None or local_time.tzinfo.utcoffset(local_time) is None:
        local_time_aware = target_local_tz.localize(
            local_time
        )  # 假定 naive local_time 就是北京时间
    else:
        local_time_aware = local_time.astimezone(
            target_local_tz
        )  # 如果已经是 aware，转换到目标本地时区

    if modality == "radar":
        # 雷达索引是 UTC，所以将查找目标转换为 UTC
        return local_time_aware.astimezone(pytz.UTC)
    elif modality == "satellite":
        # 卫星索引现在存储的是北京时间，所以查找目标也应该是北京时间
        return local_time_aware  # <-- 返回已经处理成带时区信息的北京时间
    elif modality == "rain":
        # 降雨索引也存储北京时间
        return local_time_aware
    else:
        raise ValueError(f"Unsupported modality: {modality}")


def any_modality_time_to_any_modality_time(
    any_modality_time: str,
    from_modality: str = "radar",
    to_modality: str = "satellite",
    local_tz: str = "Asia/Shanghai",
) -> str:
    """
    将特定模态的时间字符串转换为另一模态的时间字符串格式
    """
    local_time = any_modaility_time_to_local(any_modality_time, from_modality, local_tz)
    return local_to_any_modality_time(local_time, to_modality, local_tz)


# * --- projections --- #


def detect_projection(longitudes, latitudes) -> str:
    """
    根据坐标数据特征检测可能的投影方式
    """
    lon_min, lon_max = float(longitudes.min()), float(longitudes.max())
    lat_min, lat_max = float(latitudes.min()), float(latitudes.max())

    # log_print(
    #     f"Coordinate ranges: Lon[{lon_min:.3f}, {lon_max:.3f}], Lat[{lat_min:.3f}, {lat_max:.3f}]",
    #     "debug",
    # )

    # 判断是否为地理坐标系（经纬度）
    if (
        -180 <= lon_min <= 180
        and -180 <= lon_max <= 180
        and -90 <= lat_min <= 90
        and -90 <= lat_max <= 90
    ):
        # log_print("Detected: Geographic coordinates (WGS84)", "debug")
        return "epsg:4326"

    # 判断是否为投影坐标系
    elif abs(lon_min) > 1000 or abs(lon_max) > 1000:
        log_print("Detected: Projected coordinates", "debug")
        if abs(lon_min) > 1e6:  # Web Mercator范围
            log_print("Possible projection: Web Mercator (EPSG:3857)", "debug")
            return "epsg:3857"
        else:
            log_print("Unknown projection, assuming UTM or similar", "debug")
            return "unknown"
    else:
        log_print("Assuming geographic coordinates", "debug")
        return "epsg:4326"


def setup_coordinate_transform(
    source_proj: str, target_proj: str, longitudes: np.ndarray, latitudes: np.ndarray
) -> Tuple[Optional[Transformer], Tuple[float, float, float, float], bool]:
    """
    设置坐标转换器并处理原始坐标

    Returns:
        transformer: 坐标转换器 (如果需要转换)
        bounds: 目标坐标系的边界 (x_min, x_max, y_min, y_max)
        is_regular_grid: 是否为规则网格
    """
    # 设置坐标转换器
    if source_proj != target_proj:
        transformer = Transformer.from_crs(source_proj, target_proj, always_xy=True)
    else:
        transformer = None

    # 处理原始坐标
    if len(longitudes.shape) == 1 and len(latitudes.shape) == 1:
        # 规则网格情况
        is_regular_grid = True
        original_lon_mesh, original_lat_mesh = np.meshgrid(longitudes, latitudes)
    else:
        # 非规则网格情况
        is_regular_grid = False
        original_lon_mesh = longitudes
        original_lat_mesh = latitudes

    # 获取边界
    x_min_src = float(longitudes.min())
    x_max_src = float(longitudes.max())
    y_min_src = float(latitudes.min())
    y_max_src = float(latitudes.max())

    # 坐标转换
    if transformer:
        log_print(f"Converting from {source_proj} to {target_proj}", "debug")

        # 转换边界
        corners_x = [x_min_src, x_max_src, x_min_src, x_max_src]
        corners_y = [y_min_src, y_min_src, y_max_src, y_max_src]
        corners_x_target, corners_y_target = transformer.transform(corners_x, corners_y)

        x_min_target = min(corners_x_target)
        x_max_target = max(corners_x_target)
        y_min_target = min(corners_y_target)
        y_max_target = max(corners_y_target)
    else:
        # log_print("No projection conversion needed", "debug")
        x_min_target, x_max_target = x_min_src, x_max_src
        y_min_target, y_max_target = y_min_src, y_max_src

    return (
        transformer,
        (x_min_target, x_max_target, y_min_target, y_max_target),
        is_regular_grid,
    )


def fast_bilinear_interpolation(
    source_data: np.ndarray, source_coords: Tuple, target_coords: Tuple
) -> np.ndarray:
    """
    使用skimage进行快速双线性插值
    适用于规则网格到规则网格的快速转换
    """
    # 计算缩放因子
    target_height, target_width = target_coords[0].shape
    source_height, source_width = source_data.shape

    # 使用skimage的resize函数（非常快）
    resized = resize(
        source_data,
        (target_height, target_width),
        anti_aliasing=True,
        preserve_range=True,
    )

    # 处理NaN值
    nan_mask = np.isnan(source_data)
    if nan_mask.any():
        nan_resized = (
            resize(nan_mask.astype(float), (target_height, target_width)) > 0.5
        )
        resized[nan_resized] = np.nan

    return resized


def fast_nearest_neighbor(
    valid_points: np.ndarray,
    valid_data: np.ndarray,
    target_grid: Tuple,
    max_distance: Optional[float] = None,
) -> np.ndarray:
    """
    使用KDTree进行快速最近邻插值
    """
    # 构建KDTree
    tree = cKDTree(valid_points)

    # 查找最近邻
    target_points = np.column_stack([target_grid[0].ravel(), target_grid[1].ravel()])
    distances, indices = tree.query(target_points)

    # 应用最大距离限制（可选）
    result = valid_data[indices]
    if max_distance is not None:
        result[distances > max_distance] = np.nan

    return result.reshape(target_grid[0].shape)


def interpolate_data(
    original_data: np.ndarray,
    source_coords: Tuple[np.ndarray, np.ndarray],
    target_mesh: Tuple[np.ndarray, np.ndarray],
    transformer: Optional[Transformer],
    interpolation_method: str = "regular_grid",
    is_regular_grid: bool = True,
) -> np.ndarray:
    """
    通用的数据插值函数

    Args:
        original_data: 原始数据数组
        source_coords: 源坐标 (lon_mesh, lat_mesh)
        target_mesh: 目标网格 (x_mesh, y_mesh)
        transformer: 坐标转换器
        interpolation_method: 插值方法
        is_regular_grid: 是否为规则网格

    Returns:
        插值后的数据
    """
    original_lon_mesh, original_lat_mesh = source_coords
    x_mesh_regular, y_mesh_regular = target_mesh

    # 坐标转换
    if transformer:
        target_x_mesh, target_y_mesh = transformer.transform(
            original_lon_mesh, original_lat_mesh
        )
    else:
        target_x_mesh, target_y_mesh = original_lon_mesh, original_lat_mesh

    # 根据方法选择插值算法
    if interpolation_method == "fast_bilinear" and is_regular_grid and not transformer:
        # 最快的方法：直接缩放规则网格
        mapped_data = fast_bilinear_interpolation(
            original_data,
            (original_lon_mesh, original_lat_mesh),
            (x_mesh_regular, y_mesh_regular),
        )

    elif interpolation_method == "regular_grid" and is_regular_grid:
        # 使用RegularGridInterpolator
        try:
            # 获取1D坐标数组
            if len(original_lon_mesh.shape) == 2:
                lon_1d = original_lon_mesh[0, :]
                lat_1d = original_lat_mesh[:, 0]
            else:
                lon_1d = original_lon_mesh
                lat_1d = original_lat_mesh

            interp = RegularGridInterpolator(
                (lat_1d, lon_1d),
                original_data,
                method="linear",
                bounds_error=False,
                fill_value=np.nan,
            )
            points = np.column_stack([y_mesh_regular.ravel(), x_mesh_regular.ravel()])
            mapped_data = interp(points).reshape(x_mesh_regular.shape)
        except Exception as e:
            log_print(
                f"RegularGridInterpolator failed: {e}, falling back to nearest neighbor",
                "debug",
            )
            interpolation_method = "nearest"

    if interpolation_method == "nearest":
        # 使用KDTree最近邻
        source_x_flat = target_x_mesh.flatten()
        source_y_flat = target_y_mesh.flatten()
        original_data_flat = original_data.flatten()

        # 移除无效值
        valid_mask = (
            ~np.isnan(original_data_flat)
            & ~np.isnan(source_x_flat)
            & ~np.isnan(source_y_flat)
        )

        if valid_mask.sum() > 0:
            valid_points = np.column_stack(
                [source_x_flat[valid_mask], source_y_flat[valid_mask]]
            )
            valid_data = original_data_flat[valid_mask]

            mapped_data = fast_nearest_neighbor(
                valid_points, valid_data, (x_mesh_regular, y_mesh_regular)
            )
        else:
            mapped_data = np.full_like(x_mesh_regular, np.nan)

    elif interpolation_method == "griddata":
        # 传统的griddata方法（最慢）
        from scipy.interpolate import griddata

        source_x_flat = target_x_mesh.flatten()
        source_y_flat = target_y_mesh.flatten()
        original_data_flat = original_data.flatten()

        valid_mask = (
            ~np.isnan(original_data_flat)
            & ~np.isnan(source_x_flat)
            & ~np.isnan(source_y_flat)
        )

        if valid_mask.sum() > 4:
            valid_x = source_x_flat[valid_mask]
            valid_y = source_y_flat[valid_mask]
            valid_data = original_data_flat[valid_mask]

            try:
                mapped_data = griddata(
                    points=(valid_x, valid_y),
                    values=valid_data,
                    xi=(x_mesh_regular, y_mesh_regular),
                    method="linear",
                    fill_value=np.nan,
                )
            except Exception as e:
                log_print(
                    f"Linear interpolation failed: {e}, trying nearest neighbor",
                    "debug",
                )
                mapped_data = griddata(
                    points=(valid_x, valid_y),
                    values=valid_data,
                    xi=(x_mesh_regular, y_mesh_regular),
                    method="nearest",
                    fill_value=np.nan,
                )
        else:
            mapped_data = np.full_like(x_mesh_regular, np.nan)

    return mapped_data


def create_output_grid(
    bounds: Tuple[float, float, float, float], grid_width: int, grid_height: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建输出网格

    Args:
        bounds: (x_min, x_max, y_min, y_max)
        grid_width: 网格宽度
        grid_height: 网格高度

    Returns:
        (x_mesh, y_mesh): 目标坐标网格
    """
    x_min, x_max, y_min, y_max = bounds
    x_grid = np.linspace(x_min, x_max, grid_width)
    y_grid = np.linspace(y_min, y_max, grid_height)
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    return x_mesh, y_mesh


def create_display_coordinates(
    x_mesh: np.ndarray, y_mesh: np.ndarray, target_proj: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建用于显示的经纬度坐标

    Args:
        x_mesh: 目标坐标系的X网格
        y_mesh: 目标坐标系的Y网格
        target_proj: 目标投影

    Returns:
        (lon_mesh, lat_mesh): 经纬度网格
    """
    if target_proj != "epsg:4326":
        back_transformer = Transformer.from_crs(
            target_proj, "epsg:4326", always_xy=True
        )
        lon_mesh, lat_mesh = back_transformer.transform(x_mesh, y_mesh)
    else:
        lon_mesh, lat_mesh = x_mesh, y_mesh

    return lon_mesh, lat_mesh


if __name__ == "__main__":
    local_time = datetime.datetime(
        2023, 5, 31, 16, 0, 0, tzinfo=pytz.timezone("Asia/Shanghai")
    )
    # print(local_to_any_modality_time(local_time, modality="radar"))

    radar_time_str = "20230503.000000"
    t = any_modality_time_to_any_modality_time(
        radar_time_str,
        from_modality="radar",
        to_modality="satellite",
        local_tz="Asia/Shanghai",
    )
    print(t)  # 输出: 20230503_0800
