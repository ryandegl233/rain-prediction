#!/usr/bin/env python3
"""
简化的分钟级时间窗口功能测试
"""

import os
import sys

sys.path.append("/Volumes/macminiData/Data/项目/高新减灾所")

from src.dataset.rain_station_mapped import rain_read_gridded, rain_read_stations


def test_minute_window_simple():
    """测试分钟级时间窗口功能"""

    # 设置数据路径
    base_data_path = "/Volumes/macminiData/Data/项目/高新减灾所/data"
    rain_file = os.path.join(base_data_path, "rain", "202307", "分钟级雨量站数据_20230701.xlsx")
    station_file = os.path.join(base_data_path, "四川省雨量站信息.csv")

    print("测试 30 分钟时间窗口:")
    print("-" * 40)

    try:
        # 测试30分钟窗口
        result = rain_read_stations(
            file=rain_file,
            station_file=station_file,
            time_window_minutes=30
        )

        if result and 'data' in result:
            station_data = result['data']
            print(f"✅ 成功读取 {len(station_data)} 个站点")

            if len(station_data) > 0:
                total_rainfall = station_data['rainfall'].sum()
                valid_stations = len(station_data[station_data['rainfall'] > 0.1])
                max_rainfall = station_data['rainfall'].max()

                print(f"  • 有效雨量站: {len(station_data)} 个")
                print(f"  • 降雨站点 (>0.1mm): {valid_stations} 个")
                print(f"  • 总降雨量: {total_rainfall:.2f} mm")
                print(f"  • 最大降雨量: {max_rainfall:.2f} mm")
            else:
                print("  • 在 30 分钟窗口内未找到有效数据")
        else:
            print("❌ 未返回有效结果")

    except Exception as e:
        print(f"❌ 错误: {str(e)}")

    print("\n测试 60 分钟时间窗口:")
    print("-" * 40)

    try:
        # 测试60分钟窗口
        result = rain_read_stations(
            file=rain_file,
            station_file=station_file,
            time_window_minutes=60
        )

        if result and 'data' in result:
            station_data = result['data']
            print(f"✅ 成功读取 {len(station_data)} 个站点")

            if len(station_data) > 0:
                total_rainfall = station_data['rainfall'].sum()
                valid_stations = len(station_data[station_data['rainfall'] > 0.1])
                max_rainfall = station_data['rainfall'].max()

                print(f"  • 有效雨量站: {len(station_data)} 个")
                print(f"  • 降雨站点 (>0.1mm): {valid_stations} 个")
                print(f"  • 总降雨量: {total_rainfall:.2f} mm")
                print(f"  • 最大降雨量: {max_rainfall:.2f} mm")
            else:
                print("  • 在 60 分钟窗口内未找到有效数据")
        else:
            print("❌ 未返回有效结果")

    except Exception as e:
        print(f"❌ 错误: {str(e)}")


if __name__ == "__main__":
    test_minute_window_simple()
