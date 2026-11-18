#!/usr/bin/env python3
"""
分钟级时间窗口功能完整测试
"""

import os
import sys

sys.path.append("/Volumes/macminiData/Data/项目/高新减灾所")

from src.dataset.rain_station_mapped import rain_read_stations


def test_time_windows():
    """测试不同时间窗口的降雨累积效果"""

    # 设置数据路径
    base_data_path = "/Volumes/macminiData/Data/项目/高新减灾所/data"
    rain_file = os.path.join(base_data_path, "rain", "202307", "分钟级雨量站数据_20230701.xlsx")
    station_file = os.path.join(base_data_path, "四川省雨量站信息.csv")

    print("=" * 80)
    print("分钟级时间窗口功能测试 - 累积降雨比较")
    print("=" * 80)

    # 测试不同的时间窗口
    test_cases = [
        ("15分钟窗口", {"time_window_minutes": 15}),
        ("30分钟窗口", {"time_window_minutes": 30}),
        ("45分钟窗口", {"time_window_minutes": 45}),
        ("60分钟窗口", {"time_window_minutes": 60}),
        ("90分钟窗口", {"time_window_minutes": 90}),
        ("120分钟窗口", {"time_window_minutes": 120}),
        ("1小时窗口 (小时模式)", {"time_window_hours": 1}),
        ("2小时窗口 (小时模式)", {"time_window_hours": 2}),
    ]

    results = []

    for window_name, params in test_cases:
        print(f"\n📊 测试 {window_name}:")
        print("-" * 50)

        try:
            result = rain_read_stations(file=rain_file, station_file=station_file, **params)

            if result and "data" in result:
                station_data = result["data"]

                if len(station_data) > 0:
                    total_rainfall = station_data["rainfall"].sum()
                    valid_stations = len(station_data)
                    rainfall_stations = len(station_data[station_data["rainfall"] > 0.1])
                    max_rainfall = station_data["rainfall"].max()
                    avg_rainfall = (
                        station_data[station_data["rainfall"] > 0.1]["rainfall"].mean()
                        if rainfall_stations > 0
                        else 0
                    )

                    print(f"  ✅ 成功读取")
                    print(f"  • 有效站点: {valid_stations:,} 个")
                    print(
                        f"  • 降雨站点 (>0.1mm): {rainfall_stations:,} 个 ({rainfall_stations / valid_stations * 100:.1f}%)"
                    )
                    print(f"  • 总降雨量: {total_rainfall:.2f} mm")
                    print(f"  • 平均降雨量: {avg_rainfall:.2f} mm (有降雨站点)")
                    print(f"  • 最大降雨量: {max_rainfall:.2f} mm")

                    results.append(
                        {
                            "window": window_name,
                            "stations": valid_stations,
                            "rainfall_stations": rainfall_stations,
                            "total_rainfall": total_rainfall,
                            "avg_rainfall": avg_rainfall,
                            "max_rainfall": max_rainfall,
                        }
                    )
                else:
                    print(f"  ❌ 在 {window_name} 内未找到有效数据")
            else:
                print(f"  ❌ 未返回有效结果")

        except Exception as e:
            print(f"  ❌ 错误: {str(e)}")

    # 显示结果汇总
    print("\n" + "=" * 80)
    print("结果汇总:")
    print("=" * 80)
    print(
        f"{'时间窗口':<20} {'站点数':<8} {'降雨站点':<10} {'总降雨量(mm)':<12} {'平均降雨(mm)':<12} {'最大降雨(mm)':<12}"
    )
    print("-" * 80)

    for r in results:
        print(
            f"{r['window']:<20} {r['stations']:<8,} {r['rainfall_stations']:<10,} {r['total_rainfall']:<12.2f} {r['avg_rainfall']:<12.2f} {r['max_rainfall']:<12.2f}"
        )

    # 分析时间窗口效应
    print("\n📈 时间窗口效应分析:")
    print("-" * 50)

    minute_results = [r for r in results if "分钟窗口" in r["window"]]
    if len(minute_results) >= 2:
        print(
            f"• 随着时间窗口增加，累积降雨量从 {minute_results[0]['total_rainfall']:.2f}mm 增加到 {minute_results[-1]['total_rainfall']:.2f}mm"
        )
        print(
            f"• 降雨站点数从 {minute_results[0]['rainfall_stations']} 个增加到 {minute_results[-1]['rainfall_stations']} 个"
        )

    # 比较小时和分钟模式
    hour_1_result = next((r for r in results if r["window"] == "1小时窗口 (小时模式)"), None)
    minute_60_result = next((r for r in results if r["window"] == "60分钟窗口"), None)

    if hour_1_result and minute_60_result:
        print(f"• 1小时窗口对比:")
        print(
            f"  - 小时模式: {hour_1_result['total_rainfall']:.2f}mm ({hour_1_result['rainfall_stations']} 站点)"
        )
        print(
            f"  - 分钟模式: {minute_60_result['total_rainfall']:.2f}mm ({minute_60_result['rainfall_stations']} 站点)"
        )
        diff_pct = (
            abs(hour_1_result["total_rainfall"] - minute_60_result["total_rainfall"])
            / hour_1_result["total_rainfall"]
            * 100
        )
        print(f"  - 差异: {diff_pct:.2f}%")


if __name__ == "__main__":
    test_time_windows()
