import os
import re
import warnings
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
import pytz

from src.dataset.geo_utils import (
    create_display_coordinates,
    create_output_grid,
    detect_projection,
    interpolate_data,
    setup_coordinate_transform,
)
from src.utils.logging import log_print

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# 雨量站信息文件路径
RAIN_STATION_FILE = "data2/四川省雨量站信息.csv"


def parse_sheet_name(sheet_name: Union[str, int]) -> Optional[int]:
    """
    从Excel表名解析时间窗口小时数

    Args:
        sheet_name: 表名，如 "1小时", "12小时" 等，或直接的整数

    Returns:
        时间窗口小时数，如果无法解析则返回None
    """
    if isinstance(sheet_name, int):
        return sheet_name
    elif isinstance(sheet_name, str):
        # 匹配类似 "1小时", "12小时" 的格式
        match = re.search(r"(\d+)小时", sheet_name)
        if match:
            return int(match.group(1))
    return None


def load_rain_station_info(station_file: str = RAIN_STATION_FILE) -> pd.DataFrame:
    """
    加载雨量站位置信息

    Returns:
        DataFrame: 包含station_id, lng, lat等信息的DataFrame
    """
    if not os.path.exists(station_file):
        raise FileNotFoundError(f"Rain station file not found: {station_file}")

    station_info = pd.read_csv(station_file, encoding="utf-8")
    log_print(f"Loaded {len(station_info)} rain stations", "debug")

    return station_info


def process_rain_data(
    rain_data: pd.DataFrame,
    station_info: pd.DataFrame,
    target_hour: Optional[int] = None,
    target_minute: Optional[int] = None,
    time_window_hours: Optional[int] = None,
    time_window_minutes: Optional[int] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> pd.DataFrame:
    """
    处理雨量数据：时间转换、合并位置信息、数据清洗

    Args:
        rain_data: 原始雨量数据
        station_info: 雨量站位置信息
        target_hour: 目标小时 (0-23)，如果None则使用所有数据
        target_minute: 目标分钟 (0-59)，如果None则使用所有数据
        time_window_hours: 时间窗口小时数（注意：需要在调用前预处理多个sheet的数据）
        time_window_minutes: 时间窗口分钟数，计算该窗口内的累积降雨
        start_time: 自定义开始时间（格式："YYYY-MM-DD HH:MM:SS"或"HH:MM:SS"）
        end_time: 自定义结束时间（格式："YYYY-MM-DD HH:MM:SS"或"HH:MM:SS"）

    Returns:
        DataFrame: 处理后的数据，包含lng, lat, rainfall等字段
    """
    # 时间转换
    rain_data["time"] = pd.to_datetime(rain_data["数据时间戳"], unit="s")
    rain_data["time"] = rain_data["time"] + pd.Timedelta(hours=8)
    rain_data["hour"] = rain_data["time"].dt.hour  # 0-23小时格式（标准格式）
    rain_data["minute"] = rain_data["time"].dt.minute  # 0-59分钟格式
    rain_data["date"] = rain_data["time"].dt.date

    log_print(
        f"Time range: {np.array(rain_data['time']).min()} to {np.array(rain_data['time']).max()}",
        "debug",
    )

    # 早期过滤：移除abnormal状态的数据和无效降雨量
    original_count = len(rain_data)

    # 过滤条件：
    # 1. 数据状态为normal
    # 2. 降雨量非空且非负
    rain_data = rain_data[
        (rain_data["数据状态"].str.strip().str.lower() == "normal")
        & (rain_data["雨量(单位:mm)"].notna())
        & (rain_data["雨量(单位:mm)"] >= 0)
    ].copy()

    filtered_count = len(rain_data)
    log_print(
        f"Filtered data: {original_count} -> {filtered_count} records (removed {original_count - filtered_count} abnormal/invalid)",
        "debug",
    )

    if len(rain_data) == 0:
        log_print("No valid data after filtering abnormal records", "warning")
        return pd.DataFrame()

    # 根据参数类型选择处理方式
    if start_time is not None and end_time is not None:
        # 使用自定义时间范围
        log_print(f"Using custom time range: {start_time} to {end_time}", "debug")

        # 解析时间字符串
        def parse_time_string(time_str: str, reference_date: pd.Timestamp) -> pd.Timestamp:
            """解析时间字符串，支持完整时间戳或仅时分秒"""
            try:
                if ":" in time_str and "-" not in time_str:
                    # 仅时分秒格式，使用参考日期
                    time_part = pd.to_datetime(time_str, format="%H:%M:%S").time()
                    return pd.Timestamp.combine(reference_date.date(), time_part)
                else:
                    # 完整时间戳格式
                    return pd.to_datetime(time_str)
            except Exception as e:
                raise ValueError(f"Invalid time format '{time_str}': {e}")

        # 使用数据中的第一个日期作为参考
        reference_date = pd.to_datetime(rain_data["time"].iloc[0])
        start_timestamp = parse_time_string(start_time, reference_date)
        end_timestamp = parse_time_string(end_time, reference_date)

        # 筛选自定义时间范围内的数据
        window_data = rain_data[
            (rain_data["time"] >= start_timestamp) & (rain_data["time"] <= end_timestamp)
        ]

        if len(window_data) == 0:
            log_print(f"No data found in custom time range {start_time} to {end_time}", "warning")
            return pd.DataFrame()

        # 按站点分组计算累积降雨
        aggregated_data = (
            window_data.groupby("设备id")
            .agg(
                {
                    "雨量(单位:mm)": "sum",  # 累积降雨
                    "数据状态": "last",  # 最后状态
                    "time": "max",  # 最新时间
                }
            )
            .reset_index()
        )

        rain_data_to_process = aggregated_data

    elif time_window_hours is not None:
        # 计算时间窗口内的累积降雨
        log_print(f"Calculating {time_window_hours}-hour cumulative rainfall", "debug")

        # 找到数据的结束时间
        max_time = np.array(rain_data["time"]).max()
        window_start_time = max_time - pd.Timedelta(hours=time_window_hours)

        # 筛选时间窗口内的数据
        window_data = rain_data[rain_data["time"] >= window_start_time]

        if len(window_data) == 0:
            log_print(f"No data found in {time_window_hours}-hour window", "warning")
            return pd.DataFrame()

        # 按站点分组计算累积降雨
        aggregated_data = (
            window_data.groupby("设备id")
            .agg(
                {
                    "雨量(单位:mm)": "sum",  # 累积降雨
                    "数据状态": "last",  # 最后状态
                    "time": "max",  # 最新时间
                }
            )
            .reset_index()
        )

        rain_data_to_process = aggregated_data
    elif time_window_minutes is not None:
        # 计算分钟时间窗口内的累积降雨
        log_print(f"Calculating {time_window_minutes}-minute cumulative rainfall", "debug")

        # 找到数据的结束时间
        max_time = np.array(rain_data["time"]).max()
        start_time = max_time - pd.Timedelta(minutes=time_window_minutes)

        # 筛选时间窗口内的数据
        window_data = rain_data[rain_data["time"] >= start_time]

        if len(window_data) == 0:
            log_print(f"No data found in {time_window_minutes}-minute window", "warning")
            return pd.DataFrame()

        # 按站点分组计算累积降雨
        aggregated_data = (
            window_data.groupby("设备id")
            .agg(
                {
                    "雨量(单位:mm)": "sum",  # 累积降雨
                    "数据状态": "last",  # 最后状态
                    "time": "max",  # 最新时间
                }
            )
            .reset_index()
        )

        rain_data_to_process = aggregated_data
    else:
        # 筛选特定时间点的数据
        filtered_data = rain_data.copy()

        if target_hour is not None:
            if not (0 <= target_hour <= 23):
                raise ValueError(f"target_hour must be 0-23, got {target_hour}")
            filtered_data = filtered_data[filtered_data["hour"] == target_hour]
            log_print(f"Filtered to hour {target_hour}: {len(filtered_data)} records", "debug")

        if target_minute is not None:
            if not (0 <= target_minute <= 59):
                raise ValueError(f"target_minute must be 0-59, got {target_minute}")
            filtered_data = filtered_data[filtered_data["minute"] == target_minute]
            log_print(f"Filtered to minute {target_minute}: {len(filtered_data)} records", "debug")

        if len(filtered_data) == 0:
            log_print(f"No data found for hour {target_hour}, minute {target_minute}", "warning")
            return pd.DataFrame()

        rain_data_to_process = filtered_data

    # 重命名列
    rain_data_to_process["station_id"] = rain_data_to_process["设备id"]
    rain_data_to_process["rainfall"] = rain_data_to_process["雨量(单位:mm)"]
    rain_data_to_process["status"] = rain_data_to_process["数据状态"].str.strip().str.lower()

    # 合并位置信息
    merged_data = pd.merge(rain_data_to_process, station_info, on="station_id", how="left")

    # 筛选有效数据
    valid_data = merged_data[
        (merged_data["status"] == "normal")
        & (merged_data["lng"].notna())
        & (merged_data["lat"].notna())
    ]

    if len(valid_data) == 0:
        log_print("No valid rain data found", "warning")
        return pd.DataFrame()

    # 按站点分组，取平均值（处理同一站点多个记录的情况）
    agg_dict = {
        "lng": "first",
        "lat": "first",
        "rainfall": "mean",
        "status": "first",
        "time": "first",
    }

    # 只有在列存在时才添加聚合规则
    if "hour" in valid_data.columns:
        agg_dict["hour"] = "first"
    if "minute" in valid_data.columns:
        agg_dict["minute"] = "first"

    result = valid_data.groupby("station_id").agg(agg_dict).reset_index()

    log_print(f"Processed {len(result)} valid rain stations", "debug")

    return result


def rain_read_stations(
    file: str,
    sheet_name: Union[str, int] = "1小时",
    target_proj: str = "epsg:4326",
    force_source_proj: Optional[str] = None,
    min_rainfall: float = 0.0,
    station_file: str = RAIN_STATION_FILE,
    target_hour: Optional[int] = None,
    target_minute: Optional[int] = None,
    time_window_hours: Optional[int] = None,
    time_window_minutes: Optional[int] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> Dict[str, Any]:
    """
    读取雨量站原始数据，返回站点信息和降雨量

    Args:
        file: Excel文件路径
        sheet_name: 工作表名称或小时数 (1-24)
        target_proj: 目标投影
        force_source_proj: 强制指定源投影
        min_rainfall: 最小降雨量阈值
        station_file: 雨量站信息文件路径
        target_hour: 目标小时 (0-23)，如果None则使用所有数据
        target_minute: 目标分钟 (0-59)，如果None则使用所有数据
        time_window_hours: 时间窗口小时数，会自动读取多个sheet来获取完整数据
        time_window_minutes: 时间窗口分钟数，计算该窗口内的累积降雨
        start_time: 自定义开始时间（格式："YYYY-MM-DD HH:MM:SS"或"HH:MM:SS"）
        end_time: 自定义结束时间（格式："YYYY-MM-DD HH:MM:SS"或"HH:MM:SS"）

    Returns:
        Dict: 包含站点数据的字典
    """
    assert os.path.exists(file), f"File {file} does not exist."

    # 加载雨量站位置信息
    station_info = load_rain_station_info(station_file)

    # 检查是否需要读取多个sheet
    need_multi_sheet = False
    required_hours = 1  # 默认只需要1小时数据

    if time_window_hours is not None and time_window_hours > 1:
        need_multi_sheet = True
        required_hours = time_window_hours
        log_print(f"Reading {time_window_hours} hours of data across multiple sheets", "debug")

    elif start_time is not None and end_time is not None:
        # 计算自定义时间范围跨越的小时数
        def calculate_time_span_hours(start_str: str, end_str: str) -> int:
            """计算时间范围跨越的小时数"""
            try:
                # 尝试解析为时分秒格式
                if ":" in start_str and len(start_str.split(":")) >= 2:
                    start_parts = start_str.split(":")
                    end_parts = end_str.split(":")
                    start_hour = int(start_parts[0])
                    end_hour = int(end_parts[0])

                    if end_hour > start_hour:
                        return end_hour - start_hour + 1
                    elif end_hour == start_hour:
                        return 1
                    else:
                        # 跨天的情况
                        return (24 - start_hour) + end_hour + 1
                else:
                    # 完整时间戳格式，解析时间差
                    start_dt = pd.to_datetime(start_str)
                    end_dt = pd.to_datetime(end_str)
                    hours_diff = (end_dt - start_dt).total_seconds() / 3600
                    return max(1, int(hours_diff) + 1)
            except Exception:
                return 1  # 解析失败时默认为1小时

        required_hours = calculate_time_span_hours(start_time, end_time)
        if required_hours > 1:
            need_multi_sheet = True
            log_print(f"Custom time range spans {required_hours} hours, reading multiple sheets", "debug")

    # 读取Excel数据
    if need_multi_sheet:
        # 获取Excel文件中所有可用的sheet
        try:
            with pd.ExcelFile(file) as xls:
                available_sheets = xls.sheet_names
        except Exception as e:
            raise ValueError(f"Failed to read Excel file metadata: {e}")

        # 解析可用的时间窗口sheets
        hour_sheets = []
        for sheet in available_sheets:
            hours = parse_sheet_name(sheet)
            if hours is not None and hours <= required_hours:
                hour_sheets.append((hours, sheet))

        # 按小时数排序，从最大到最小
        hour_sheets.sort(key=lambda x: x[0], reverse=True)

        if not hour_sheets:
            raise ValueError(f"No suitable sheets found for {required_hours}-hour window")

        # 读取所有需要的数据
        all_data = []
        for hours, sheet in hour_sheets:
            try:
                sheet_data = read_excel_fast(file, sheet)
                log_print(f"Loaded {len(sheet_data)} records from sheet '{sheet}' ({hours}h)", "debug")
                all_data.append(sheet_data)
            except Exception as e:
                log_print(f"Failed to read sheet '{sheet}': {e}", "warning")
                continue

        if not all_data:
            raise ValueError(f"Failed to read any data for {required_hours}-hour window")

        # 合并所有数据
        rain_data = pd.concat(all_data, ignore_index=True)
        log_print(f"Combined {len(rain_data)} total records from {len(all_data)} sheets", "debug")

    else:
        # 读取单个sheet的数据
        if isinstance(sheet_name, int):
            sheet_name = f"{sheet_name}小时"

        try:
            # Use fast Excel reading with automatic engine selection
            rain_data = read_excel_fast(file, sheet_name)
            log_print(f"Loaded {len(rain_data)} records from sheet '{sheet_name}'", "debug")
        except Exception as e:
            raise ValueError(f"Failed to read sheet '{sheet_name}' from {file}: {e}")

    # 处理数据
    processed_data = process_rain_data(
        rain_data,
        station_info,
        target_hour=target_hour,
        target_minute=target_minute,
        time_window_hours=time_window_hours,
        time_window_minutes=time_window_minutes,
        start_time=start_time,
        end_time=end_time,
    )

    if len(processed_data) == 0:
        raise ValueError("No valid rain data found after processing")

    # 筛选有降雨的站点
    rainfall_data = processed_data[processed_data["rainfall"] >= min_rainfall]

    # 检测源投影
    if force_source_proj:
        source_proj = force_source_proj
    else:
        source_proj = detect_projection(processed_data["lng"].values, processed_data["lat"].values)

    # 坐标转换（如果需要）
    if source_proj != target_proj:
        from pyproj import Transformer

        transformer = Transformer.from_crs(source_proj, target_proj, always_xy=True)

        # 转换所有站点坐标
        x_coords, y_coords = transformer.transform(
            processed_data["lng"].values, processed_data["lat"].values
        )
        processed_data = processed_data.copy()
        processed_data["x_proj"] = x_coords
        processed_data["y_proj"] = y_coords

        # 转换有降雨站点坐标
        if len(rainfall_data) > 0:
            x_rain_coords, y_rain_coords = transformer.transform(
                rainfall_data["lng"].values, rainfall_data["lat"].values
            )
            rainfall_data = rainfall_data.copy()
            rainfall_data["x_proj"] = x_rain_coords
            rainfall_data["y_proj"] = y_rain_coords
    else:
        processed_data = processed_data.copy()
        processed_data["x_proj"] = processed_data["lng"]
        processed_data["y_proj"] = processed_data["lat"]

        if len(rainfall_data) > 0:
            rainfall_data = rainfall_data.copy()
            rainfall_data["x_proj"] = rainfall_data["lng"]
            rainfall_data["y_proj"] = rainfall_data["lat"]

    # 获取边界
    if len(processed_data) > 0:
        x_min_src = float(processed_data["lng"].min())
        x_max_src = float(processed_data["lng"].max())
        y_min_src = float(processed_data["lat"].min())
        y_max_src = float(processed_data["lat"].max())
    else:
        x_min_src, x_max_src = 96.0, 109.0  # 默认四川省范围
        y_min_src, y_max_src = 25.0, 35.0

    return dict(
        data=processed_data,  # 所有有效站点数据（包含投影坐标）
        rainfall_data=rainfall_data,  # 有降雨的站点数据（包含投影坐标）
        bounds_lonlat=(x_min_src, x_max_src, y_min_src, y_max_src),
        source_proj=source_proj,
        target_proj=target_proj,
        sheet_name=sheet_name,
        station_count=len(processed_data),
        rainfall_station_count=len(rainfall_data),
        min_rainfall=min_rainfall,
    )


def rain_read_gridded(
    file: str,
    sheet_name: Union[str, int] = "1小时",
    grid_width: int = 1000,
    grid_height: int = 800,
    target_proj: str = "epsg:4326",
    force_source_proj: Optional[str] = None,
    min_rainfall: float = 0.0,
    station_file: str = RAIN_STATION_FILE,
    bounds: Optional[tuple] = None,
    target_hour: Optional[int] = None,
    target_minute: Optional[int] = None,
    time_window_hours: Optional[int] = None,
    time_window_minutes: Optional[int] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> Dict[str, Any]:
    """
    读取雨量站数据并映射到网格上，生成与雷达/卫星相同大小的稀疏矩阵

    Args:
        file: Excel文件路径
        sheet_name: 工作表名称或小时数 (1-24)
        grid_width: 输出网格宽度
        grid_height: 输出网格高度
        target_proj: 目标投影
        force_source_proj: 强制指定源投影
        min_rainfall: 最小降雨量阈值
        station_file: 雨量站信息文件路径
        bounds: 指定边界 (x_min, x_max, y_min, y_max)，如果None则自动计算

    Returns:
        Dict: 包含网格化数据的字典
    """
    assert os.path.exists(file), f"File {file} does not exist."

    # 加载雨量站位置信息
    station_info = load_rain_station_info(station_file)

    # 检查是否需要读取多个sheet
    need_multi_sheet = False
    required_hours = 1  # 默认只需要1小时数据

    if time_window_hours is not None and time_window_hours > 1:
        need_multi_sheet = True
        required_hours = time_window_hours
        log_print(f"Reading {time_window_hours} hours of data across multiple sheets", "debug")

    elif start_time is not None and end_time is not None:
        # 计算自定义时间范围跨越的小时数
        def calculate_time_span_hours(start_str: str, end_str: str) -> int:
            """计算时间范围跨越的小时数"""
            try:
                # 尝试解析为时分秒格式
                if ":" in start_str and len(start_str.split(":")) >= 2:
                    start_parts = start_str.split(":")
                    end_parts = end_str.split(":")
                    start_hour = int(start_parts[0])
                    end_hour = int(end_parts[0])

                    if end_hour > start_hour:
                        return end_hour - start_hour + 1
                    elif end_hour == start_hour:
                        return 1
                    else:
                        # 跨天的情况
                        return (24 - start_hour) + end_hour + 1
                else:
                    # 完整时间戳格式，解析时间差
                    start_dt = pd.to_datetime(start_str)
                    end_dt = pd.to_datetime(end_str)
                    hours_diff = (end_dt - start_dt).total_seconds() / 3600
                    return max(1, int(hours_diff) + 1)
            except Exception:
                return 1  # 解析失败时默认为1小时

        required_hours = calculate_time_span_hours(start_time, end_time)
        if required_hours > 1:
            need_multi_sheet = True
            log_print(f"Custom time range spans {required_hours} hours, reading multiple sheets", "debug")

    # 读取Excel数据
    if need_multi_sheet:
        # 获取Excel文件中所有可用的sheet
        try:
            with pd.ExcelFile(file) as xls:
                available_sheets = xls.sheet_names
        except Exception as e:
            raise ValueError(f"Failed to read Excel file metadata: {e}")

        # 解析可用的时间窗口sheets
        hour_sheets = []
        for sheet in available_sheets:
            hours = parse_sheet_name(sheet)
            if hours is not None and hours <= required_hours:
                hour_sheets.append((hours, sheet))

        # 按小时数排序，从最大到最小
        hour_sheets.sort(key=lambda x: x[0], reverse=True)

        if not hour_sheets:
            raise ValueError(f"No suitable sheets found for {required_hours}-hour window")

        # 读取所有需要的数据
        all_data = []
        for hours, sheet in hour_sheets:
            try:
                sheet_data = read_excel_fast(file, sheet)
                log_print(f"Loaded {len(sheet_data)} records from sheet '{sheet}' ({hours}h)", "debug")
                all_data.append(sheet_data)
            except Exception as e:
                log_print(f"Failed to read sheet '{sheet}': {e}", "warning")
                continue

        if not all_data:
            raise ValueError(f"Failed to read any data for {required_hours}-hour window")

        # 合并所有数据
        rain_data = pd.concat(all_data, ignore_index=True)
        log_print(f"Combined {len(rain_data)} total records from {len(all_data)} sheets", "debug")

    else:
        # 读取单个sheet的数据
        if isinstance(sheet_name, int):
            sheet_name = f"{sheet_name}小时"

        try:
            # Use fast Excel reading with automatic engine selection
            rain_data = read_excel_fast(file, sheet_name)
            log_print(f"Loaded {len(rain_data)} records from sheet '{sheet_name}'", "debug")
        except Exception as e:
            raise ValueError(f"Failed to read sheet '{sheet_name}' from {file}: {e}")

    # 处理数据
    processed_data = process_rain_data(
        rain_data,
        station_info,
        target_hour=target_hour,
        target_minute=target_minute,
        time_window_hours=time_window_hours,
        time_window_minutes=time_window_minutes,
        start_time=start_time,
        end_time=end_time,
    )

    if len(processed_data) == 0:
        raise ValueError("No valid rain data found after processing")

    # 筛选有降雨的站点（可选）
    rainfall_data = processed_data[processed_data["rainfall"] >= min_rainfall]

    if len(rainfall_data) == 0:
        log_print(f"No rainfall data above {min_rainfall}mm threshold", "warning")
        # 创建空的输出网格
        if bounds is None:
            bounds = (96.0, 109.0, 25.0, 35.0)  # 四川省大致范围
        x_mesh, y_mesh = create_output_grid(bounds, grid_width, grid_height)
        mapped_data = np.zeros_like(x_mesh)
        lon_mesh, lat_mesh = x_mesh, y_mesh
        source_proj = "epsg:4326"
    else:
        # 提取坐标和降雨量
        longitudes = rainfall_data["lng"].values
        latitudes = rainfall_data["lat"].values
        rainfall_values = rainfall_data["rainfall"].values

        log_print(f"Found {len(rainfall_data)} stations with rainfall >= {min_rainfall}mm", "debug")
        log_print(
            f"Rainfall range: [{np.array(rainfall_values).min():.2f}, {np.array(rainfall_values).max():.2f}] mm",
            "debug",
        )

        # 检测源投影
        if force_source_proj:
            source_proj = force_source_proj
        else:
            source_proj = detect_projection(longitudes, latitudes)

        # 设置坐标转换和网格
        transformer, calculated_bounds, is_regular_grid = setup_coordinate_transform(
            source_proj, target_proj, np.array(longitudes), np.array(latitudes)
        )

        # 使用指定边界或计算出的边界
        final_bounds = bounds if bounds is not None else calculated_bounds

        # 创建输出网格
        x_mesh, y_mesh = create_output_grid(final_bounds, grid_width, grid_height)

        # 创建稀疏矩阵（将雨量站数据映射到网格点）
        mapped_data = np.zeros_like(x_mesh)

        # 如果需要坐标转换
        if transformer:
            station_x, station_y = transformer.transform(longitudes, latitudes)
        else:
            station_x, station_y = longitudes, latitudes

        # 将雨量站数据映射到最近的网格点
        for i, (x_pos, y_pos, rainfall) in enumerate(zip(station_x, station_y, rainfall_values)):
            # 找到最近的网格点
            x_idx = np.argmin(np.abs(x_mesh[0, :] - x_pos))
            y_idx = np.argmin(np.abs(y_mesh[:, 0] - y_pos))

            # 在该网格点设置降雨值（如果有多个站点映射到同一网格点，取最大值）
            if y_idx < mapped_data.shape[0] and x_idx < mapped_data.shape[1]:
                mapped_data[y_idx, x_idx] = max(mapped_data[y_idx, x_idx], rainfall)

        # 创建显示坐标
        lon_mesh, lat_mesh = create_display_coordinates(x_mesh, y_mesh, target_proj)

    # 获取原始边界
    if len(processed_data) > 0:
        x_min_src = float(processed_data["lng"].min())
        x_max_src = float(processed_data["lng"].max())
        y_min_src = float(processed_data["lat"].min())
        y_max_src = float(processed_data["lat"].max())
    else:
        x_min_src, x_max_src = 96.0, 109.0  # 默认四川省范围
        y_min_src, y_max_src = 25.0, 35.0

    return dict(
        data=processed_data,  # 原始处理后的站点数据
        rainfall_data=rainfall_data
        if len(rainfall_data) > 0
        else processed_data,  # 有降雨的站点数据
        mapped_data=mapped_data,
        x_mesh=x_mesh,
        y_mesh=y_mesh,
        lon_mesh=lon_mesh,
        lat_mesh=lat_mesh,
        bounds_target=final_bounds if "final_bounds" in locals() else bounds,
        bounds_lonlat=(x_min_src, x_max_src, y_min_src, y_max_src),
        grid_width=grid_width,
        grid_height=grid_height,
        source_proj=source_proj,
        target_proj=target_proj,
        sheet_name=sheet_name,
        station_count=len(processed_data),
        rainfall_station_count=len(rainfall_data) if len(rainfall_data) > 0 else 0,
    )


def read_excel_fast(file: str, sheet_name: str) -> pd.DataFrame:
    """
    使用最快的可用引擎读取Excel文件

    引擎性能排序（从快到慢）：
    1. calamine - Rust引擎，速度最快（5-10x faster）
    2. openpyxl - Python引擎，功能完整

    Args:
        file: Excel文件路径
        sheet_name: 工作表名称

    Returns:
        DataFrame: 读取的数据

    Raises:
        RuntimeError: 如果所有引擎都无法读取文件
    """
    # 按速度优先级尝试不同引擎
    engines_to_try = [
        ("calamine", "python-calamine"),
        ("openpyxl", "openpyxl"),  # 备选方案
    ]

    last_error = None

    for engine, package in engines_to_try:
        try:
            log_print(f"Trying to read with {engine} engine...", "debug")

            # 为不同引擎设置优化参数
            if engine == "calamine":
                # calamine引擎优化参数
                return pd.read_excel(
                    file,
                    sheet_name=sheet_name,
                    engine="calamine",  # 直接使用字符串字面量
                    dtype={
                        "设备id": "string",
                        "雨量(单位:mm)": "float32",  # 使用float32节省内存
                        "数据状态": "category",  # 状态数据用category类型
                        "数据时间戳": "int64",
                    },
                )
            else:
                # openpyxl引擎
                return pd.read_excel(file, sheet_name=sheet_name, engine="openpyxl")

        except ImportError as e:
            log_print(f"{package} not installed, trying next engine... ({e})", "debug")
            last_error = e
            continue
        except Exception as e:
            log_print(f"Failed to read with {engine}: {e}", "debug")
            last_error = e
            continue

    # 如果所有引擎都失败，抛出错误
    raise RuntimeError(f"Failed to read {file} with any available engine. Last error: {last_error}")


def test_rain_read():
    """测试雨量站数据读取功能"""
    import time

    # 测试文件
    file = "data/rain/202306/分钟级雨量站数据_20230603.xlsx"

    if not os.path.exists(file):
        print(f"Test file not found: {file}")
        return

    # 测试不同工作表
    sheets_to_test = ["1小时", "12小时", "24小时"]

    for sheet in sheets_to_test:
        print(f"\n=== Testing {sheet} ===")

        # 测试原始数据接口
        print(f"\n--- Testing rain_read_stations ---")
        start_time = time.time()

        try:
            result_stations = rain_read_stations(
                file=file,
                sheet_name=sheet,
                min_rainfall=0.1,  # 只考虑0.1mm以上的降雨
                time_window_minutes=10,
            )

            end_time = time.time()
            print(f"Processing time: {end_time - start_time:.2f} seconds")

            # 输出统计信息
            print(f"Total stations: {result_stations['station_count']}")
            print(f"Rainfall stations (>= 0.1mm): {result_stations['rainfall_station_count']}")

            if result_stations["rainfall_station_count"] > 0:
                rainfall_data = result_stations["rainfall_data"]
                print(
                    f"Rainfall range: [{rainfall_data['rainfall'].min():.2f}, {rainfall_data['rainfall'].max():.2f}] mm"
                )
                print(
                    f"Coordinate range: lng[{rainfall_data['lng'].min():.2f}, {rainfall_data['lng'].max():.2f}], lat[{rainfall_data['lat'].min():.2f}, {rainfall_data['lat'].max():.2f}]"
                )
            else:
                print("No rainfall data found")

        except Exception as e:
            print(f"Error processing {sheet} (stations): {e}")

        # 测试网格化接口
        print(f"\n--- Testing rain_read_gridded ---")
        start_time = time.time()

        try:
            result_gridded = rain_read_gridded(
                file=file,
                sheet_name=sheet,
                grid_width=500,
                grid_height=400,
                min_rainfall=0.1,  # 只考虑0.1mm以上的降雨
            )

            end_time = time.time()
            print(f"Processing time: {end_time - start_time:.2f} seconds")

            # 输出统计信息
            print(f"Total stations: {result_gridded['station_count']}")
            print(f"Rainfall stations (>= 0.1mm): {result_gridded['rainfall_station_count']}")

            mapped_data = result_gridded["mapped_data"]
            print(f"Output grid shape: {mapped_data.shape}")
            non_zero_count = (mapped_data > 0).sum()
            print(f"Non-zero grid points: {non_zero_count}")

            if non_zero_count > 0:
                print(
                    f"Grid rainfall range: [{mapped_data[mapped_data > 0].min():.2f}, {mapped_data.max():.2f}] mm"
                )
            else:
                print("No rainfall data in grid")

        except Exception as e:
            print(f"Error processing {sheet} (gridded): {e}")


if __name__ == "__main__":
    test_rain_read()
