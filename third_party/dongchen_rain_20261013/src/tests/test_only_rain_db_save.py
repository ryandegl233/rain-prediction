"""
分片数据库验证测试
对比Excel原始数据与数据库存储/查询结果，验证完整数据存储方案的准确性
现在存储所有有效数据（包括0值），不再仅存储降雨数据
"""

import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.dataset.rain_station_mapped import read_excel_fast
from src.tools.rain_station_excel_to_shard_db import ShardedRainDataImporter
from src.utils.logging import log_print


class ShardedDBValidator:
    """分片数据库验证器"""

    def __init__(
        self, test_data_dir: str = "data/rain", shard_db_dir: str = "data/test_rainfall_shards"
    ):
        self.test_data_dir = Path(test_data_dir)
        self.shard_db_dir = Path(shard_db_dir)
        self.importer = ShardedRainDataImporter(str(self.shard_db_dir))

        # 清理测试环境
        self._cleanup_test_environment()

    def _cleanup_test_environment(self):
        """清理测试环境"""
        import shutil

        if self.shard_db_dir.exists():
            shutil.rmtree(self.shard_db_dir)
        self.shard_db_dir.mkdir(parents=True, exist_ok=True)

        # 重新初始化导入器
        self.importer = ShardedRainDataImporter(str(self.shard_db_dir))

    def find_test_files(self) -> List[Path]:
        """查找测试用的Excel文件"""
        test_files = []

        # 查找Excel文件
        for pattern in ["*.xlsx", "*.xls"]:
            test_files.extend(self.test_data_dir.glob(pattern))

        # 也查找子目录中的文件
        for subdir in self.test_data_dir.iterdir():
            if subdir.is_dir():
                for pattern in ["*.xlsx", "*.xls"]:
                    test_files.extend(subdir.glob(pattern))

        return sorted(test_files)[:3]  # 限制测试文件数量

    def read_excel_reference_data(self, excel_file: Path, sheets: List[str] = None) -> pd.DataFrame:
        """读取Excel文件作为参考数据"""
        if sheets is None:
            sheets = ["1小时", "12小时"]  # 默认测试工作表

        all_data = []

        for sheet_name in sheets:
            try:
                # 读取Excel数据
                data = read_excel_fast(str(excel_file), sheet_name)

                if len(data) == 0:
                    continue

                # 处理时间数据（与导入器相同的逻辑）
                data["time"] = pd.to_datetime(data["数据时间戳"], unit="s")
                data["time"] = data["time"] + pd.Timedelta(hours=8)  # UTC to Beijing time
                data["timestamp"] = data["数据时间戳"]
                data["hour"] = data["time"].dt.hour
                data["minute"] = data["time"].dt.minute
                data["date"] = data["time"].dt.date.astype(str)
                data["datetime_str"] = data["time"].dt.strftime("%Y-%m-%d %H:%M:%S")

                # 过滤有效数据
                valid_data = data[
                    (data["雨量(单位:mm)"].notna())
                    & (data["雨量(单位:mm)"] >= 0)
                    & (data["数据状态"].str.strip().str.lower() == "normal")
                ].copy()

                # 标准化列名
                valid_data["station_id"] = valid_data["设备id"].astype(str)
                valid_data["rainfall"] = valid_data["雨量(单位:mm)"].astype(float)
                valid_data["status"] = valid_data["数据状态"].str.strip().str.lower()
                valid_data["source_file"] = excel_file.name
                valid_data["source_sheet"] = sheet_name

                all_data.append(valid_data)

            except Exception as e:
                log_print(f"Failed to read sheet {sheet_name} from {excel_file}: {e}", "warning")
                continue

        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            return combined_data
        else:
            return pd.DataFrame()

    def test_single_file_import_and_query(self, excel_file: Path) -> Dict:
        """测试单个文件的导入和查询准确性"""
        print(f"\n=== 测试文件: {excel_file.name} ===")

        # 1. 读取Excel原始数据
        print("1. 读取Excel原始数据...")
        start_time = time.time()
        reference_data = self.read_excel_reference_data(excel_file)
        read_time = time.time() - start_time

        if len(reference_data) == 0:
            return {"status": "skipped", "reason": "No valid data in Excel file"}

        print(f"   Excel数据: {len(reference_data):,} 条记录 ({read_time:.3f}秒)")
        print(
            f"   时间范围: {reference_data['datetime_str'].min()} 到 {reference_data['datetime_str'].max()}"
        )
        print(f"   站点数量: {reference_data['station_id'].nunique()}")
        print(f"   降雨记录: {len(reference_data[reference_data['rainfall'] > 0]):,} 条")
        print(f"   零值记录: {len(reference_data[reference_data['rainfall'] == 0]):,} 条")

        # 2. 导入到分片数据库
        print("2. 导入到分片数据库...")
        start_time = time.time()
        self.importer.import_excel_file(str(excel_file), ["1小时", "12小时"])
        import_time = time.time() - start_time
        print(f"   导入耗时: {import_time:.3f}秒")

        # 3. 查询存储的数据（仅降雨数据）
        print("3. 查询存储的降雨数据...")
        start_time = time.time()

        # 确定查询时间范围
        min_time = reference_data["datetime_str"].min()
        max_time = reference_data["datetime_str"].max()

        stored_data = self.importer.query_by_time_range(
            start_time=min_time,
            end_time=max_time,
            min_rainfall=0.0,  # 获取所有存储的数据
        )
        query_time = time.time() - start_time

        print(f"   存储数据: {len(stored_data):,} 条记录 ({query_time:.3f}秒)")

        # 4. 验证降雨数据准确性
        print("4. 验证降雨数据准确性...")

        # 从原始数据中提取有降雨的记录（用于对比）
        reference_rainfall = reference_data[reference_data["rainfall"] > 0.1].copy()
        print(f"   原始降雨数据: {len(reference_rainfall):,} 条 (>0.1mm)")

        # 比较数据
        validation_results = self._compare_rainfall_data(reference_rainfall, stored_data)

        # 5. 测试完整时间序列查询
        print("5. 测试完整时间序列查询...")

        # 选择一个小时间范围进行完整序列测试
        test_start = reference_data["datetime_str"].min()
        test_end_dt = pd.to_datetime(test_start) + pd.Timedelta(hours=2)
        test_end = test_end_dt.strftime("%Y-%m-%d %H:%M:%S")

        start_time = time.time()
        complete_data = self.importer.query_by_time_range_with_fill(
            start_time=test_start, end_time=test_end, fill_zeros=True, time_interval_minutes=60
        )
        fill_query_time = time.time() - start_time

        print(f"   完整序列: {len(complete_data):,} 条记录 ({fill_query_time:.3f}秒)")

        if len(complete_data) > 0:
            filled_count = len(complete_data[complete_data["status"] == "filled"])
            actual_count = len(complete_data[complete_data["status"] != "filled"])
            print(f"   实际数据: {actual_count} 条, 填充数据: {filled_count} 条")

        # 6. 验证完整序列的准确性
        complete_validation = self._validate_complete_time_series(
            reference_data, complete_data, test_start, test_end
        )

        return {
            "status": "completed",
            "file_name": excel_file.name,
            "reference_total": len(reference_data),
            "reference_rainfall": len(reference_rainfall),
            "stored_rainfall": len(stored_data),
            "read_time": read_time,
            "import_time": import_time,
            "query_time": query_time,
            "fill_query_time": fill_query_time,
            "validation_results": validation_results,
            "complete_validation": complete_validation,
            "storage_efficiency": len(stored_data) / len(reference_data)
            if len(reference_data) > 0
            else 0,
        }

    def _compare_rainfall_data(self, reference: pd.DataFrame, stored: pd.DataFrame) -> Dict:
        """比较原始降雨数据与存储的数据 - 优化版本"""
        results = {
            "total_match": True,
            "count_match": len(reference) == len(stored),
            "missing_records": [],
            "extra_records": [],
            "value_mismatches": [],
        }

        if len(stored) == 0 and len(reference) > 0:
            results["total_match"] = False
            results["missing_records"] = [f"All {len(reference)} reference records missing"]
            return results

        # 使用pandas merge来比较数据
        # 标记数据来源
        reference_marked = reference[["station_id", "datetime_str", "rainfall"]].copy()
        reference_marked["source"] = "reference"

        stored_marked = stored[["station_id", "datetime_str", "rainfall"]].copy()
        stored_marked["source"] = "stored"

        # 外连接找出所有记录
        all_records = pd.concat([reference_marked, stored_marked], ignore_index=True)

        # 按站点和时间分组
        grouped = all_records.groupby(["station_id", "datetime_str"])

        for (station_id, datetime_str), group in grouped:
            key = f"{station_id}|||{datetime_str}"

            ref_data = group[group["source"] == "reference"]
            stored_data = group[group["source"] == "stored"]

            if len(ref_data) > 0 and len(stored_data) == 0:
                # 参考数据存在，但存储数据不存在
                results["missing_records"].append(key)

            elif len(ref_data) == 0 and len(stored_data) > 0:
                # 存储数据存在，但参考数据不存在
                results["extra_records"].append(key)

            elif len(ref_data) > 0 and len(stored_data) > 0:
                # 两边都有数据，比较数值
                ref_rainfall = ref_data.iloc[0]["rainfall"]
                stored_rainfall = stored_data.iloc[0]["rainfall"]

                if abs(ref_rainfall - stored_rainfall) > 0.001:  # 允许小数精度误差
                    results["value_mismatches"].append({
                        "key": key,
                        "reference": ref_rainfall,
                        "stored": stored_rainfall,
                        "difference": abs(ref_rainfall - stored_rainfall)
                    })

        # 更新总体匹配状态
        results["total_match"] = (
            len(results["missing_records"]) == 0
            and len(results["extra_records"]) == 0
            and len(results["value_mismatches"]) == 0
        )

        return results

    def _validate_complete_time_series(
        self, reference: pd.DataFrame, complete: pd.DataFrame, start_time: str, end_time: str
    ) -> Dict:
        """验证完整时间序列的准确性"""
        results = {
            "time_coverage_correct": True,
            "rainfall_values_correct": True,
            "zero_filling_correct": True,
            "details": {},
        }

        if len(complete) == 0:
            results["time_coverage_correct"] = False
            return results

        # 验证时间覆盖
        complete_times = set(complete["datetime_str"])
        expected_times = (
            pd.date_range(start=start_time, end=end_time, freq="60min")
            .strftime("%Y-%m-%d %H:%M:%S")
            .tolist()
        )

        expected_times_set = set(expected_times)

        # 按站点验证
        for station_id in complete["station_id"].unique():
            station_complete = complete[complete["station_id"] == station_id]
            station_times = set(station_complete["datetime_str"])

            if station_times != expected_times_set:
                results["time_coverage_correct"] = False
                results["details"][f"station_{station_id}_time_coverage"] = False

        # 验证降雨值的准确性
        # 获取参考时间段内的数据
        reference_period = reference[
            (reference["datetime_str"] >= start_time) & (reference["datetime_str"] <= end_time)
        ]

        # 检查有降雨的时间点
        for _, ref_row in reference_period.iterrows():
            if ref_row["rainfall"] > 0.1:
                matching_complete = complete[
                    (complete["station_id"] == ref_row["station_id"])
                    & (complete["datetime_str"] == ref_row["datetime_str"])
                ]

                if len(matching_complete) == 0:
                    results["rainfall_values_correct"] = False
                elif abs(matching_complete.iloc[0]["rainfall"] - ref_row["rainfall"]) > 0.001:
                    results["rainfall_values_correct"] = False

        # 验证零值填充
        filled_records = complete[complete["status"] == "filled"]
        for _, filled_row in filled_records.iterrows():
            # 检查这个时间点在原始数据中是否确实没有记录或为零
            ref_match = reference_period[
                (reference_period["station_id"] == filled_row["station_id"])
                & (reference_period["datetime_str"] == filled_row["datetime_str"])
            ]

            if len(ref_match) > 0 and ref_match.iloc[0]["rainfall"] > 0.1:
                # 原始数据有降雨，但被标记为填充，这是错误
                results["zero_filling_correct"] = False
                break

        return results

    def run_comprehensive_test(self) -> Dict:
        """运行综合测试"""
        print("🧪 开始分片数据库综合验证测试")
        print("=" * 60)

        # 导入站点信息
        print("📍 导入站点信息...")
        self.importer.import_station_info()

        # 查找测试文件
        test_files = self.find_test_files()

        if not test_files:
            print("❌ 未找到测试用的Excel文件")
            return {"status": "failed", "reason": "No test files found"}

        print(f"📁 找到 {len(test_files)} 个测试文件")

        overall_results = {
            "status": "success",
            "files_tested": len(test_files),
            "files_passed": 0,
            "files_failed": 0,
            "total_import_time": 0,
            "total_query_time": 0,
            "storage_efficiency_avg": 0,
            "file_results": [],
        }

        # 测试每个文件
        for i, test_file in enumerate(test_files):
            print(f"\n📊 测试进度: {i + 1}/{len(test_files)}")

            try:
                file_result = self.test_single_file_import_and_query(test_file)

                if file_result["status"] == "completed":
                    # 验证结果
                    validation = file_result["validation_results"]
                    complete_validation = file_result["complete_validation"]

                    if (
                        validation["total_match"]
                        and complete_validation["time_coverage_correct"]
                        and complete_validation["rainfall_values_correct"]
                    ):
                        overall_results["files_passed"] += 1
                        print("✅ 文件测试通过")
                    else:
                        overall_results["files_failed"] += 1
                        print("❌ 文件测试失败")
                        print(f"   降雨数据匹配: {validation['total_match']}")
                        print(f"   时间覆盖正确: {complete_validation['time_coverage_correct']}")
                        print(f"   降雨值正确: {complete_validation['rainfall_values_correct']}")

                    overall_results["total_import_time"] += file_result["import_time"]
                    overall_results["total_query_time"] += file_result["query_time"]
                    overall_results["storage_efficiency_avg"] += file_result["storage_efficiency"]

                else:
                    overall_results["files_failed"] += 1
                    print(f"⚠️  文件跳过: {file_result.get('reason', 'Unknown')}")

                overall_results["file_results"].append(file_result)

            except Exception as e:
                overall_results["files_failed"] += 1
                print(f"❌ 文件测试异常: {e}")
                overall_results["file_results"].append(
                    {"status": "error", "file_name": test_file.name, "error": str(e)}
                )

        # 计算平均值
        if overall_results["files_passed"] > 0:
            overall_results["storage_efficiency_avg"] /= overall_results["files_passed"]

        # 打印最终结果
        self._print_final_results(overall_results)

        return overall_results

    def _print_final_results(self, results: Dict):
        """打印最终测试结果"""
        print("\n" + "=" * 60)
        print("🎯 综合测试结果")
        print("=" * 60)

        print(f"📊 测试文件: {results['files_tested']}")
        print(f"✅ 通过: {results['files_passed']}")
        print(f"❌ 失败: {results['files_failed']}")
        print(f"📈 成功率: {results['files_passed'] / results['files_tested'] * 100:.1f}%")

        print(f"\n⏱️  性能指标:")
        print(f"   总导入时间: {results['total_import_time']:.2f} 秒")
        print(f"   总查询时间: {results['total_query_time']:.2f} 秒")
        print(f"   平均存储效率: {results['storage_efficiency_avg'] * 100:.1f}% (仅存储有意义降雨)")

        # 数据库统计
        stats = self.importer.get_statistics()
        print(f"\n💾 数据库统计:")
        print(f"   分片数量: {stats['shard_count']}")
        print(f"   总记录数: {stats['total_records']:,}")
        print(f"   存储大小: {stats['total_size_mb']:.2f} MB")
        print(f"   时间范围: {stats['date_range'][0]} 到 {stats['date_range'][1]}")

        if results["files_passed"] == results["files_tested"]:
            print("\n🎉 所有测试通过！优化方案验证成功！")
        else:
            print(f"\n⚠️  {results['files_failed']} 个文件测试失败，需要检查具体原因")


def main():
    """主测试函数"""
    validator = ShardedDBValidator()
    results = validator.run_comprehensive_test()

    # 保存测试结果
    import json

    results_file = Path("data/test_results_sharded_db.json")

    # 转换不可序列化的对象
    serializable_results = results.copy()
    for file_result in serializable_results.get("file_results", []):
        if "validation_results" in file_result:
            # 限制数组大小以便序列化
            validation = file_result["validation_results"]
            for key in ["missing_records", "extra_records", "value_mismatches"]:
                if key in validation and len(validation[key]) > 10:
                    validation[key] = validation[key][:10] + [
                        f"... and {len(validation[key]) - 10} more"
                    ]

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    print(f"\n📄 详细测试结果已保存到: {results_file}")

    return results


if __name__ == "__main__":
    main()
