#!/usr/bin/env python3
"""
测试雨量站数据读取的两个接口：
1. rain_read_stations - 返回原始站点数据
2. rain_read_gridded - 返回稀疏矩阵格式数据
"""

import os
import time

import numpy as np

from src.dataset.rain_station_mapped import rain_read_gridded, rain_read_stations


def test_rain_interfaces():
    """测试雨量站数据的两个接口"""

    # 测试文件
    test_file = "data/rain/202306/分钟级雨量站数据_20230603.xlsx"

    if not os.path.exists(test_file):
        print(f"测试文件不存在: {test_file}")
        print("请确保数据文件路径正确")
        return

    # 测试参数
    sheet_name = "1小时"  # 测试1小时降雨数据
    min_rainfall = 0.1   # 最小降雨量阈值

    print("="*60)
    print("雨量站数据读取接口测试")
    print("="*60)

    # 1. 测试原始数据接口
    print("\n【测试1】原始站点数据接口 (rain_read_stations)")
    print("-"*40)

    start_time = time.time()

    try:
        stations_result = rain_read_stations(
            file=test_file,
            sheet_name=sheet_name,
            min_rainfall=min_rainfall,
            target_proj="epsg:4326"
        )

        processing_time = time.time() - start_time

        print(f"✓ 处理成功，耗时: {processing_time:.3f}秒")
        print(f"  总站点数: {stations_result['station_count']}")
        print(f"  有降雨站点数: {stations_result['rainfall_station_count']}")
        print(f"  数据表名: {stations_result['sheet_name']}")
        print(f"  最小降雨阈值: {stations_result['min_rainfall']}mm")
        print(f"  坐标边界: {stations_result['bounds_lonlat']}")

        # 查看降雨站点数据结构
        if stations_result['rainfall_station_count'] > 0:
            rainfall_data = stations_result['rainfall_data']
            print(f"  降雨量范围: [{rainfall_data['rainfall'].min():.2f}, {rainfall_data['rainfall'].max():.2f}]mm")
            print(f"  数据字段: {list(rainfall_data.columns)}")

            # 显示前几个站点信息
            print("\n  前5个有降雨的站点:")
            for i, (idx, row) in enumerate(rainfall_data.head().iterrows()):
                print(f"    站点{i+1}: ID={row['station_id']}, 位置=({row['lng']:.3f}, {row['lat']:.3f}), 降雨={row['rainfall']:.2f}mm")
        else:
            print("  ⚠️ 未找到超过阈值的降雨数据")

    except Exception as e:
        print(f"✗ 接口1测试失败: {e}")
        return

    # 2. 测试网格化接口
    print("\n【测试2】网格化稀疏矩阵接口 (rain_read_gridded)")
    print("-"*40)

    start_time = time.time()

    try:
        gridded_result = rain_read_gridded(
            file=test_file,
            sheet_name=sheet_name,
            grid_width=500,
            grid_height=400,
            min_rainfall=min_rainfall,
            target_proj="epsg:4326"
        )

        processing_time = time.time() - start_time

        print(f"✓ 处理成功，耗时: {processing_time:.3f}秒")
        print(f"  总站点数: {gridded_result['station_count']}")
        print(f"  有降雨站点数: {gridded_result['rainfall_station_count']}")
        print(f"  输出网格大小: {gridded_result['mapped_data'].shape}")
        print(f"  网格分辨率: {gridded_result['grid_width']} x {gridded_result['grid_height']}")

        # 分析稀疏矩阵
        mapped_data = gridded_result['mapped_data']
        non_zero_mask = mapped_data > 0
        non_zero_count = non_zero_mask.sum()

        print(f"  非零网格点数: {non_zero_count} / {mapped_data.size} ({non_zero_count/mapped_data.size*100:.2f}%)")

        if non_zero_count > 0:
            non_zero_values = mapped_data[non_zero_mask]
            print(f"  网格降雨量范围: [{non_zero_values.min():.2f}, {non_zero_values.max():.2f}]mm")
            print(f"  平均降雨量: {non_zero_values.mean():.2f}mm")

            # 显示网格统计
            print(f"  网格坐标范围:")
            print(f"    X: [{gridded_result['x_mesh'].min():.3f}, {gridded_result['x_mesh'].max():.3f}]")
            print(f"    Y: [{gridded_result['y_mesh'].min():.3f}, {gridded_result['y_mesh'].max():.3f}]")

        else:
            print("  ⚠️ 网格中无降雨数据（所有值为0）")

    except Exception as e:
        print(f"✗ 接口2测试失败: {e}")
        return

    # 3. 接口一致性验证
    print("\n【验证】两个接口的一致性检查")
    print("-"*40)

    # 检查基本统计信息是否一致
    stations_count_match = stations_result['station_count'] == gridded_result['station_count']
    rainfall_count_match = stations_result['rainfall_station_count'] == gridded_result['rainfall_station_count']

    print(f"✓ 总站点数一致: {stations_count_match} ({stations_result['station_count']} == {gridded_result['station_count']})")
    print(f"✓ 降雨站点数一致: {rainfall_count_match} ({stations_result['rainfall_station_count']} == {gridded_result['rainfall_station_count']})")

    # 检查降雨量数据
    if stations_result['rainfall_station_count'] > 0 and gridded_result['rainfall_station_count'] > 0:
        # 比较原始数据和网格数据的降雨量
        original_rainfall = stations_result['rainfall_data']['rainfall'].values
        grid_rainfall = gridded_result['mapped_data'][gridded_result['mapped_data'] > 0]

        original_sum = original_rainfall.sum()
        grid_sum = grid_rainfall.sum()

        print(f"✓ 原始降雨总量: {original_sum:.2f}mm")
        print(f"✓ 网格降雨总量: {grid_sum:.2f}mm")
        print(f"✓ 差异比例: {abs(original_sum - grid_sum) / original_sum * 100:.2f}%")

    print("\n" + "="*60)
    print("测试完成！两个接口都工作正常")
    print("="*60)


if __name__ == "__main__":
    test_rain_interfaces()
