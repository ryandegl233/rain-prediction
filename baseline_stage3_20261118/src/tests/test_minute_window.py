#!/usr/bin/env python3
"""
测试分钟级时间窗口功能
Test minute-level time window functionality for rain station data processing
"""

import os
import sys

sys.path.append("/Volumes/macminiData/Data/项目/高新减灾所")

from src.dataset.rain_station_mapped import rain_read_gridded, rain_read_stations


def test_minute_window():
    """测试分钟级时间窗口功能"""

    # 设置数据路径
    base_data_path = "/Volumes/macminiData/Data/项目/高新减灾所/data"
    rain_file = os.path.join(base_data_path, "rain", "202307", "分钟级雨量站数据_20230701.xlsx")
    station_file = os.path.join(base_data_path, "四川省雨量站信息.csv")

    # 检查文件是否存在
    if not os.path.exists(rain_file):
        print(f"雨量数据文件不存在: {rain_file}")
        print("可用的雨量文件:")
        rain_dir = os.path.join(base_data_path, "rain", "202307")
        if os.path.exists(rain_dir):
            for file in sorted(os.listdir(rain_dir))[:5]:  # 显示前5个文件
                print(f"  {file}")
        return

    if not os.path.exists(station_file):
        print(f"雨量站信息文件不存在: {station_file}")
        return

    print("=" * 60)
    print("分钟级时间窗口功能测试")
    print("=" * 60)

    # 测试不同的分钟时间窗口
    minute_windows = [30, 60, 120, 180]  # 30分钟、1小时、2小时、3小时

    for minutes in minute_windows:
        print(f"\n📊 测试 {minutes} 分钟时间窗口:")
        print("-" * 40)

        try:
            # 测试站点数据读取
            result = rain_read_stations(
                file=rain_file,
                station_file=station_file,
                time_window_minutes=minutes
            )

            if result and 'data' in result:
                station_data = result['data']

                if len(station_data) > 0:
                    total_rainfall = station_data['rainfall'].sum()
                    valid_stations = len(station_data[station_data['rainfall'] > 0.1])
                    max_rainfall = station_data['rainfall'].max()

                    print(f"  • 有效雨量站: {len(station_data)} 个")
                    print(f"  • 降雨站点 (>0.1mm): {valid_stations} 个")
                    print(f"  • 总降雨量: {total_rainfall:.2f} mm")
                    print(f"  • 最大降雨量: {max_rainfall:.2f} mm")

                    # 测试网格化数据
                    grid_result = rain_read_gridded(
                        file=rain_file,
                        station_file=station_file,
                        time_window_minutes=minutes,
                        grid_width=400,
                        grid_height=500,
                        min_rainfall=0.1
                    )

                    if grid_result is not None:
                        mapped_data = grid_result['mapped_data']
                        non_zero_count = (mapped_data > 0).sum()
                        grid_total = mapped_data.sum()

                        print(f"  • 网格大小: {mapped_data.shape}")
                        print(f"  • 有降雨网格点: {non_zero_count} 个")
                        print(f"  • 网格总降雨量: {grid_total:.2f} mm")
                        print(f"  • 数据一致性: {abs(total_rainfall - grid_total) / total_rainfall * 100:.2f}% 差异")
                    else:
                        print("  • 网格化处理失败")

                else:
                    print(f"  • 在 {minutes} 分钟窗口内未找到有效数据")
            else:
                print(f"  • 在 {minutes} 分钟窗口内未找到有效数据")

        except Exception as e:
            print(f"  ❌ 处理 {minutes} 分钟窗口时出错: {str(e)}")

    print("\n" + "=" * 60)
    print("测试完成")

    # 比较小时和分钟时间窗口
    print("\n🔄 比较小时和分钟时间窗口:")
    print("-" * 40)

    try:        # 1小时窗口 (小时模式)
        hour_data = rain_read_stations(
            file=rain_file,
            station_file=station_file,
            time_window_hours=1
        )

        # 60分钟窗口 (分钟模式)
        minute_data = rain_read_stations(
            file=rain_file,
            station_file=station_file,
            time_window_minutes=60
        )

        if len(hour_data) > 0 and len(minute_data) > 0:
            hour_total = hour_data['rainfall'].sum()
            minute_total = minute_data['rainfall'].sum()

            print(f"  • 1小时窗口 (小时模式): {len(hour_data)} 站点, 总降雨 {hour_total:.2f} mm")
            print(f"  • 60分钟窗口 (分钟模式): {len(minute_data)} 站点, 总降雨 {minute_total:.2f} mm")
            print(f"  • 两种模式差异: {abs(hour_total - minute_total) / hour_total * 100:.2f}%")
        else:
            print("  • 无法比较: 某个模式下没有有效数据")

    except Exception as e:
        print(f"  ❌ 比较过程中出错: {str(e)}")


if __name__ == "__main__":
    test_minute_window()
