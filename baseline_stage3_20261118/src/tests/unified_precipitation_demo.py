#!/usr/bin/env python3
"""
统一降水数据处理示例
展示如何将卫星、雷达、雨量站三种数据源统一到相同的坐标系统
"""

import os
import time
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np

from src.dataset.rain_station_mapped import rain_read_gridded, rain_read_stations

# 导入各种数据读取函数
from src.dataset.read_nc_file_mapped import satellite_read
from src.dataset.read_nc_file_mapped_v2 import radar_read


def unified_precipitation_demo():
    """统一降水数据处理演示"""

    print("="*80)
    print("统一降水数据处理系统演示")
    print("="*80)

    # 定义统一的参数
    target_proj = "epsg:4326"  # 统一使用WGS84地理坐标
    grid_width, grid_height = 500, 400  # 统一网格大小
    target_bounds = (103.0, 107.0, 30.0, 34.0)  # 四川中部区域

    print(f"\n统一参数:")
    print(f"  目标投影: {target_proj}")
    print(f"  网格大小: {grid_width} x {grid_height}")
    print(f"  目标区域: {target_bounds}")

    results = {}

    # 1. 处理卫星数据（如果存在）
    print(f"\n{'='*60}")
    print("【1】卫星降水数据处理")
    print(f"{'='*60}")

    satellite_files = [
        "data/satellite/GFS_20230603_00Z.nc",
        "data/satellite/satellite_precipitation.nc"
    ]

    satellite_data = None
    for sat_file in satellite_files:
        if os.path.exists(sat_file):
            print(f"找到卫星文件: {sat_file}")
            try:
                start_time = time.time()
                satellite_data = satellite_read(
                    sat_file,
                    grid_width=grid_width,
                    grid_height=grid_height,
                    target_proj=target_proj,
                    bounds=target_bounds,
                    interpolation_method="regular_grid"
                )
                processing_time = time.time() - start_time

                print(f"✓ 卫星数据处理成功，耗时: {processing_time:.3f}秒")
                print(f"  数据形状: {satellite_data['mapped_data'].shape}")
                print(f"  数据范围: [{np.nanmin(satellite_data['mapped_data']):.3f}, {np.nanmax(satellite_data['mapped_data']):.3f}]")
                print(f"  有效点数: {(~np.isnan(satellite_data['mapped_data'])).sum()}")

                results['satellite'] = satellite_data
                break

            except Exception as e:
                print(f"✗ 卫星数据处理失败: {e}")

    if satellite_data is None:
        print("⚠️ 未找到可用的卫星数据文件")

    # 2. 处理雷达数据（如果存在）
    print(f"\n{'='*60}")
    print("【2】雷达降水数据处理")
    print(f"{'='*60}")

    radar_files = [
        "data/radar/radar_precipitation.nc",
        "data/radar/RADAR_20230603.nc"
    ]

    radar_data = None
    for radar_file in radar_files:
        if os.path.exists(radar_file):
            print(f"找到雷达文件: {radar_file}")
            try:
                start_time = time.time()
                radar_data = radar_read(
                    radar_file,
                    grid_width=grid_width,
                    grid_height=grid_height,
                    target_proj=target_proj,
                    bounds=target_bounds,
                    interpolation_method="regular_grid"
                )
                processing_time = time.time() - start_time

                print(f"✓ 雷达数据处理成功，耗时: {processing_time:.3f}秒")
                print(f"  数据形状: {radar_data['mapped_data'].shape}")
                print(f"  数据范围: [{np.nanmin(radar_data['mapped_data']):.3f}, {np.nanmax(radar_data['mapped_data']):.3f}]")
                print(f"  有效点数: {(~np.isnan(radar_data['mapped_data'])).sum()}")

                results['radar'] = radar_data
                break

            except Exception as e:
                print(f"✗ 雷达数据处理失败: {e}")

    if radar_data is None:
        print("⚠️ 未找到可用的雷达数据文件")

    # 3. 处理雨量站数据
    print(f"\n{'='*60}")
    print("【3】雨量站降水数据处理")
    print(f"{'='*60}")

    rain_file = "data/rain/202306/分钟级雨量站数据_20230603.xlsx"

    if os.path.exists(rain_file):
        print(f"找到雨量站文件: {rain_file}")

        try:
            # 获取原始站点数据
            start_time = time.time()
            rain_stations = rain_read_stations(
                rain_file,
                sheet_name="1小时",
                target_proj=target_proj,
                min_rainfall=0.1
            )

            # 获取网格化数据
            rain_gridded = rain_read_gridded(
                rain_file,
                sheet_name="1小时",
                grid_width=grid_width,
                grid_height=grid_height,
                target_proj=target_proj,
                bounds=target_bounds,
                min_rainfall=0.1
            )
            processing_time = time.time() - start_time

            print(f"✓ 雨量站数据处理成功，耗时: {processing_time:.3f}秒")
            print(f"  总站点数: {rain_stations['station_count']}")
            print(f"  有降雨站点数: {rain_stations['rainfall_station_count']}")
            print(f"  网格化数据形状: {rain_gridded['mapped_data'].shape}")
            print(f"  网格非零点数: {(rain_gridded['mapped_data'] > 0).sum()}")

            if rain_stations['rainfall_station_count'] > 0:
                rainfall_data = rain_stations['rainfall_data']
                print(f"  降雨量范围: [{rainfall_data['rainfall'].min():.3f}, {rainfall_data['rainfall'].max():.3f}]mm")

            results['rain_stations'] = rain_stations
            results['rain_gridded'] = rain_gridded

        except Exception as e:
            print(f"✗ 雨量站数据处理失败: {e}")
    else:
        print(f"⚠️ 未找到雨量站数据文件: {rain_file}")

    # 4. 数据统一性验证
    print(f"\n{'='*60}")
    print("【4】数据统一性验证")
    print(f"{'='*60}")

    # 检查所有数据是否具有相同的网格大小和坐标系
    grid_shapes = []
    coordinate_systems = []

    for name, data in results.items():
        if name == 'rain_stations':
            continue  # 跳过原始站点数据

        if 'mapped_data' in data:
            grid_shapes.append((name, data['mapped_data'].shape))
            coordinate_systems.append((name, data.get('target_proj', 'unknown')))

    print("网格形状对比:")
    for name, shape in grid_shapes:
        print(f"  {name}: {shape}")

    print("\n坐标系对比:")
    for name, proj in coordinate_systems:
        print(f"  {name}: {proj}")

    # 检查一致性
    all_shapes = [shape for _, shape in grid_shapes]
    all_projs = [proj for _, proj in coordinate_systems]

    shapes_consistent = len(set(all_shapes)) <= 1
    projs_consistent = len(set(all_projs)) <= 1

    print(f"\n✓ 网格形状一致: {shapes_consistent}")
    print(f"✓ 坐标系一致: {projs_consistent}")

    if shapes_consistent and projs_consistent:
        print("🎉 所有数据源已成功统一到相同的坐标系统！")
    else:
        print("⚠️ 数据源之间存在不一致，需要进一步调整")

    # 5. 数据融合示例
    print(f"\n{'='*60}")
    print("【5】多源数据融合示例")
    print(f"{'='*60}")

    if len(results) > 1:
        print("可用数据源:")
        for name in results.keys():
            if name != 'rain_stations':
                print(f"  - {name}")

        # 示例：创建一个简单的数据融合
        if 'rain_gridded' in results:
            rain_grid = results['rain_gridded']['mapped_data']
            print(f"\n雨量站网格数据统计:")
            print(f"  有降雨网格点: {(rain_grid > 0).sum()}")
            print(f"  最大降雨量: {rain_grid.max():.3f}mm")
            print(f"  平均降雨量: {rain_grid[rain_grid > 0].mean():.3f}mm")

        print("\n数据融合策略建议:")
        print("  1. 使用雨量站数据作为地面真值进行验证")
        print("  2. 雷达数据提供高时空分辨率的降水结构")
        print("  3. 卫星数据填补雷达覆盖范围外的区域")
        print("  4. 可以使用加权平均或机器学习方法进行融合")

    # 6. 性能统计
    print(f"\n{'='*60}")
    print("【6】处理性能统计")
    print(f"{'='*60}")

    print("各数据源处理结果:")
    for name, data in results.items():
        if name == 'rain_stations':
            print(f"  {name}: {data['station_count']} 站点, {data['rainfall_station_count']} 有降雨")
        elif 'mapped_data' in data:
            grid_data = data['mapped_data']
            valid_points = (~np.isnan(grid_data)).sum() if 'satellite' in name or 'radar' in name else (grid_data > 0).sum()
            print(f"  {name}: 形状 {grid_data.shape}, 有效点 {valid_points}")

    print(f"\n系统具备以下能力:")
    print(f"  ✓ 多源数据统一坐标系转换")
    print(f"  ✓ 灵活的网格大小配置")
    print(f"  ✓ 高效的数据插值算法")
    print(f"  ✓ 稀疏数据的网格化处理")
    print(f"  ✓ 统一的数据接口和格式")

    return results


if __name__ == "__main__":
    results = unified_precipitation_demo()

    print(f"\n{'='*80}")
    print("演示完成！")
    print("系统已准备好进行多源降水数据的统一处理和分析。")
    print(f"{'='*80}")
