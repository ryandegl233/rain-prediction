#!/usr/bin/env python3
# coding: utf-8
"""
测试脚本：将卫星亮温图、雷达映射图和雨量站网格图绘制到一个 1×3 的 matplotlib 网格中
"""

from datetime import datetime, timedelta

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time

# 设置中文字体和负号显示
matplotlib.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

from src.dataset.geo_utils import any_modality_time_to_any_modality_time, local_to_any_modality_time
from src.dataset.read_nc_file_mapped import radar_read, satellite_read
from src.tools.rain_station_excel_to_shard_db import ShardedRainDataImporter


def main():
    # 基于北京时间定义查询时刻
    t = "2023-05-02 00:00:00"
    t_d = datetime.strptime(t, "%Y-%m-%d %H:%M:%S")  # 北京时间

    # 设置统一的四川省边界范围 (经度, 纬度)
    unified_bounds = (97.0, 109.0, 26.0, 35.0)  # (lon_min, lon_max, lat_min, lat_max)
    grid_width, grid_height = 128, 128  # 统一网格分辨率

    print(f"使用统一边界: 经度 {unified_bounds[0]}°-{unified_bounds[1]}°, 纬度 {unified_bounds[2]}°-{unified_bounds[3]}°")
    print(f"统一网格分辨率: {grid_width}x{grid_height}")

    # 1. 读取并处理卫星数据（强制使用统一边界）
    sat_t = local_to_any_modality_time(t_d, modality='satellite')  # 格式 YYYYMMDD_HHMM
    sat_file = f"/HardDisk/JieYiZhu/MMRainPrediction/data/satellite/{t_d.strftime('%Y%m')}/{sat_t}.nc"

    # 创建自定义的网格读取函数来强制使用统一边界
    from src.dataset.geo_utils import create_output_grid

    sat_res = satellite_read(
        file=sat_file,
        grid_width=grid_width,
        grid_height=grid_height,
        target_proj="epsg:4326",
        interpolation_method="regular_grid",
        crop_bounds_latlon=unified_bounds,  # 使用统一边界
    )
    # 将多波段亮温数据取平均进行显示
    sat_img = np.asarray(sat_res["mapped_bands"])[6]

    # 2. 读取并处理雷达数据（强制使用统一边界）
    radar_dir = t_d.strftime("%Y%m%d")
    radar_t = local_to_any_modality_time(t_d, modality='radar')    # UTC 格式 YYYYMMDD.HHMMSS
    radar_file = f"/HardDisk/JieYiZhu/MMRainPrediction/data/radar/{t_d.strftime('%Y%m')}/{radar_dir}/ACHN.QREF000.{radar_t}.nc"
    rad_res = radar_read(
        file=radar_file,
        grid_width=None,
        grid_height=None,
        target_proj="epsg:4326",
        interpolation_method="regular_grid",
        crop_bounds_latlon=unified_bounds  # 使用统一边界
    )
    rad_img = rad_res["mapped_data"]

    # 3. 生成雨量站网格数据（使用统一边界和分辨率）
    importer = ShardedRainDataImporter("/HardDisk/JieYiZhu/MMRainPrediction/data/rainfall_shards")
    importer.import_station_info()
    start_time_d = (t_d - timedelta(hours=0)) # 直接使用北京时间
    start_time = start_time_d.strftime("%Y-%m-%d %H:%M:%S")
    end_time = (start_time_d + timedelta(minutes=30)).strftime("%Y-%m-%d %H:%M:%S")
    st_res = importer.meshgrid_rain(
        start_time=start_time,
        end_time=end_time,
        grid_width=128,
        grid_height=128,
        bounds=unified_bounds  # 使用统一边界
    )
    st_img = st_res["mapped_data"]

    # 4. 绘制 1×3 子图（使用统一边界）
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    print(f"卫星数据实际边界: {sat_res.get('bounds_lonlat', 'N/A')}")
    print(f"雷达数据实际边界: {rad_res.get('bounds_lonlat', 'N/A')}")
    print(f"雨量站数据实际边界: {st_res.get('bounds_lonlat', 'N/A')}")

    # 卫星数据 - 使用统一的经纬度坐标
    im0 = axes[0].imshow(
        sat_img,
        cmap="viridis",
        origin="upper",
        extent=[unified_bounds[0], unified_bounds[1], unified_bounds[2], unified_bounds[3]]
    )
    axes[0].set_title("卫星亮温平均图")
    axes[0].set_xlabel("经度 (°E)")
    axes[0].set_ylabel("纬度 (°N)")
    axes[0].grid(True, alpha=0.3)
    fig.colorbar(im0, ax=axes[0], orientation="vertical", label="亮温 (K)")

    # 雷达数据 - 使用统一的经纬度坐标
    im1 = axes[1].imshow(
        rad_img,
        cmap="rainbow",
        origin="upper",
        extent=[unified_bounds[0], unified_bounds[1], unified_bounds[2], unified_bounds[3]]
    )
    axes[1].set_title("雷达数据映射")
    axes[1].set_xlabel("经度 (°E)")
    axes[1].set_ylabel("纬度 (°N)")
    axes[1].grid(True, alpha=0.3)
    fig.colorbar(im1, ax=axes[1], orientation="vertical", label="反射率 (dBZ)")

    # 雨量站数据 - 使用统一的经纬度坐标
    im2 = axes[2].imshow(
        st_img,
        cmap="Blues",
        origin="upper",
        extent=[unified_bounds[0], unified_bounds[1], unified_bounds[2], unified_bounds[3]]
    )
    axes[2].set_title("雨量站网格化降雨")
    axes[2].set_xlabel("经度 (°E)")
    axes[2].set_ylabel("纬度 (°N)")
    axes[2].grid(True, alpha=0.3)
    fig.colorbar(im2, ax=axes[2], orientation="vertical", label="降雨量 (mm)")

    plt.suptitle(f"时间：{t_d.strftime('%Y-%m-%d %H:%M:%S')}", fontsize=18)

    plt.tight_layout()
    output_path = "data/all_data_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"图像已保存到: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
