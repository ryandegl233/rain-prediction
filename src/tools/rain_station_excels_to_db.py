import glob
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import duckdb
import pandas as pd

from src.dataset.rain_station_mapped import load_rain_station_info, read_excel_fast
from src.utils.logging import log_print


class RainDataImporterDuckDB:
    """雨量站数据导入器 - DuckDB版本"""

    def __init__(self, db_path: str = "data/rainfall_database.duckdb"):
        """
        初始化导入器

        Args:
            db_path: DuckDB数据库文件路径
        """
        self.db_path = db_path
        self.ensure_database_exists()

    def ensure_database_exists(self):
        """确保数据库和表结构存在"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        conn = duckdb.connect(self.db_path)

        # 创建雨量站基础信息表
        conn.execute("""
            CREATE TABLE IF NOT EXISTS stations (
                station_id VARCHAR PRIMARY KEY,
                name VARCHAR,
                lng DOUBLE NOT NULL,
                lat DOUBLE NOT NULL,
                region VARCHAR,
                altitude DOUBLE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # 创建降雨数据表 - 使用列式存储优化
        conn.execute("""
            CREATE TABLE IF NOT EXISTS rainfall_data (
                station_id VARCHAR NOT NULL,
                timestamp BIGINT NOT NULL,
                datetime_str VARCHAR NOT NULL,
                rainfall DOUBLE NOT NULL,
                status VARCHAR NOT NULL,
                hour INTEGER,
                minute INTEGER,
                date VARCHAR,
                source_file VARCHAR,
                source_sheet VARCHAR,
                import_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (station_id, timestamp)
            )
        """)

        # 创建索引以提高查询性能 - DuckDB会自动优化列式存储
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rainfall_datetime ON rainfall_data(datetime_str)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rainfall_station ON rainfall_data(station_id)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rainfall_time_range ON rainfall_data(datetime_str, station_id)"
        )

        conn.close()
        log_print("DuckDB database schema initialized", "info")

    def import_station_info(self, station_file: str = "data/四川省雨量站信息.csv"):
        """
        导入雨量站基础信息

        Args:
            station_file: 雨量站信息CSV文件路径
        """
        if not os.path.exists(station_file):
            log_print(f"Station file not found: {station_file}", "warning")
            return

        try:
            station_info = load_rain_station_info(station_file)

            conn = duckdb.connect(self.db_path)

            imported_count = 0
            updated_count = 0

            for _, row in station_info.iterrows():
                station_data = {
                    "station_id": row["station_id"],
                    "lng": float(row["lng"]),
                    "lat": float(row["lat"]),
                    "name": row.get("name", ""),
                    "region": row.get("region", ""),
                    "altitude": row.get("altitude", None),
                }

                # 检查站点是否已存在
                result = conn.execute(
                    "SELECT station_id FROM stations WHERE station_id = ?",
                    [station_data["station_id"]],
                ).fetchone()

                if result:
                    # 更新现有站点
                    conn.execute(
                        """
                        UPDATE stations
                        SET lng = ?, lat = ?, name = ?, region = ?, altitude = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE station_id = ?
                    """,
                        [
                            station_data["lng"],
                            station_data["lat"],
                            station_data["name"],
                            station_data["region"],
                            station_data["altitude"],
                            station_data["station_id"],
                        ],
                    )
                    updated_count += 1
                else:
                    # 插入新站点
                    conn.execute(
                        """
                        INSERT INTO stations (station_id, lng, lat, name, region, altitude)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """,
                        [
                            station_data["station_id"],
                            station_data["lng"],
                            station_data["lat"],
                            station_data["name"],
                            station_data["region"],
                            station_data["altitude"],
                        ],
                    )
                    imported_count += 1

            conn.close()
            log_print(
                f"Station info imported: {imported_count} new, {updated_count} updated", "info"
            )

        except Exception as e:
            log_print(f"Error importing station info: {e}", "error")
            raise

    def import_excel_file(self, file_path: str, sheets_to_import: Optional[List[str]] = None):
        """
        导入单个Excel文件的数据

        Args:
            file_path: Excel文件路径
            sheets_to_import: 要导入的工作表列表，None表示导入所有工作表
        """
        if not os.path.exists(file_path):
            log_print(f"File not found: {file_path}", "error")
            return

        try:
            # 获取文件中的所有工作表
            with pd.ExcelFile(file_path) as xls:
                available_sheets = xls.sheet_names

            # 确定要导入的工作表
            if sheets_to_import is None:
                sheets_to_import = available_sheets
            else:
                # 过滤出实际存在的工作表
                sheets_to_import = [
                    sheet for sheet in sheets_to_import if sheet in available_sheets
                ]

            log_print(f"Importing from {file_path}, sheets: {sheets_to_import}", "info")

            total_imported = 0
            file_name = os.path.basename(file_path)

            for sheet_name in sheets_to_import:
                try:
                    # 读取Excel数据
                    rain_data = read_excel_fast(file_path, sheet_name)
                    log_print(f"Read {len(rain_data)} records from sheet '{sheet_name}'", "debug")

                    if len(rain_data) == 0:
                        continue

                    # 处理时间数据
                    rain_data["time"] = pd.to_datetime(rain_data["数据时间戳"], unit="s")
                    rain_data["time"] = rain_data["time"] + pd.Timedelta(hours=8)
                    rain_data["timestamp"] = rain_data["数据时间戳"]  # 保留原始时间戳
                    rain_data["hour"] = rain_data["time"].dt.hour
                    rain_data["minute"] = rain_data["time"].dt.minute
                    rain_data["date"] = rain_data["time"].dt.date.astype(str)
                    rain_data["datetime_str"] = rain_data["time"].dt.strftime("%Y-%m-%d %H:%M:%S")

                    # 导入到数据库
                    imported_count = self._import_dataframe_to_db(rain_data, file_name, sheet_name)
                    total_imported += imported_count

                    log_print(
                        f"Imported {imported_count} records from sheet '{sheet_name}'", "info"
                    )

                except Exception as e:
                    log_print(f"Error importing sheet '{sheet_name}': {e}", "error")
                    continue

            log_print(f"Total imported from {file_name}: {total_imported} records", "info")

        except Exception as e:
            log_print(f"Error importing file {file_path}: {e}", "error")
            raise

    def _import_dataframe_to_db(self, df: pd.DataFrame, source_file: str, source_sheet: str) -> int:
        """
        将DataFrame导入到数据库

        Args:
            df: 要导入的数据
            source_file: 源文件名
            source_sheet: 源工作表名

        Returns:
            导入的记录数
        """
        # 过滤无效数据
        valid_df = df[
            (df["雨量(单位:mm)"].notna())
            & (df["雨量(单位:mm)"] >= 0)
            & (df["数据状态"].str.strip().str.lower() == "normal")
        ].copy()

        if len(valid_df) == 0:
            return 0

        # 准备数据
        valid_df["station_id"] = valid_df["设备id"].astype(str)
        valid_df["rainfall"] = valid_df["雨量(单位:mm)"].astype(float)
        valid_df["status"] = valid_df["数据状态"].str.strip().str.lower()
        valid_df["source_file"] = source_file
        valid_df["source_sheet"] = source_sheet

        # 选择需要的列
        columns_to_insert = [
            "station_id",
            "timestamp",
            "datetime_str",
            "rainfall",
            "status",
            "hour",
            "minute",
            "date",
            "source_file",
            "source_sheet",
        ]

        insert_df = valid_df[columns_to_insert]

        # 使用DuckDB的批量插入 - 比逐行插入快很多
        conn = duckdb.connect(self.db_path)

        try:
            # 使用 INSERT OR REPLACE 来处理重复数据
            conn.execute("""
                INSERT OR REPLACE INTO rainfall_data
                (station_id, timestamp, datetime_str, rainfall, status, hour, minute, date, source_file, source_sheet)
                SELECT * FROM insert_df
            """)

            imported_count = len(insert_df)

        except Exception as e:
            log_print(f"Error in batch insert: {e}", "error")
            # 回退到逐行插入
            imported_count = 0
            for _, row in insert_df.iterrows():
                try:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO rainfall_data
                        (station_id, timestamp, datetime_str, rainfall, status, hour, minute, date, source_file, source_sheet)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        [
                            row["station_id"],
                            int(row["timestamp"]),
                            row["datetime_str"],
                            float(row["rainfall"]),
                            row["status"],
                            int(row["hour"]),
                            int(row["minute"]),
                            row["date"],
                            row["source_file"],
                            row["source_sheet"],
                        ],
                    )
                    imported_count += 1
                except Exception as row_e:
                    log_print(f"Error importing record: {row_e}", "debug")
                    continue

        finally:
            conn.close()

        return imported_count

    def import_directory(
        self, directory: str, pattern: str = "*.xlsx", sheets_to_import: Optional[List[str]] = None
    ):
        """
        批量导入目录中的Excel文件

        Args:
            directory: 目录路径
            pattern: 文件匹配模式
            sheets_to_import: 要导入的工作表列表
        """
        if not os.path.exists(directory):
            log_print(f"Directory not found: {directory}", "error")
            return

        # 查找匹配的文件
        file_pattern = os.path.join(directory, pattern)
        excel_files = glob.glob(file_pattern)

        if not excel_files:
            log_print(f"No Excel files found in {directory} with pattern {pattern}", "warning")
            return

        log_print(f"Found {len(excel_files)} Excel files to import", "info")

        # 导入每个文件
        for file_path in excel_files:
            try:
                log_print(f"Processing file: {file_path}", "info")
                self.import_excel_file(file_path, sheets_to_import)
            except Exception as e:
                log_print(f"Error processing file {file_path}: {e}", "error")
                continue

    def get_statistics(self) -> dict:
        """获取数据库统计信息"""
        conn = duckdb.connect(self.db_path)

        # 站点统计
        station_count = conn.execute("SELECT COUNT(*) FROM stations").fetchone()[0]

        # 降雨数据统计
        rainfall_record_count = conn.execute("SELECT COUNT(*) FROM rainfall_data").fetchone()[0]

        # 时间范围
        time_range = conn.execute(
            "SELECT MIN(datetime_str), MAX(datetime_str) FROM rainfall_data"
        ).fetchone()

        # 数据源统计
        file_stats = conn.execute(
            "SELECT source_file, COUNT(*) FROM rainfall_data GROUP BY source_file"
        ).fetchall()

        conn.close()

        return {
            "station_count": station_count,
            "rainfall_record_count": rainfall_record_count,
            "time_range": time_range,
            "file_statistics": file_stats,
        }

    def query_by_time_range(
        self, start_time: str, end_time: str, station_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        按时间范围查询数据

        Args:
            start_time: 开始时间 ("YYYY-MM-DD HH:MM:SS")
            end_time: 结束时间 ("YYYY-MM-DD HH:MM:SS")
            station_ids: 站点ID列表，None表示查询所有站点

        Returns:
            查询结果DataFrame
        """
        conn = duckdb.connect(self.db_path)

        sql = """
            SELECT r.*, s.lng, s.lat, s.name, s.region
            FROM rainfall_data r
            LEFT JOIN stations s ON r.station_id = s.station_id
            WHERE r.datetime_str >= ? AND r.datetime_str <= ?
        """
        params = [start_time, end_time]

        if station_ids:
            placeholders = ",".join(["?" for _ in station_ids])
            sql += f" AND r.station_id IN ({placeholders})"
            params.extend(station_ids)

        sql += " ORDER BY r.datetime_str, r.station_id"

        result = conn.execute(sql, params).df()
        conn.close()

        return result

    def query_by_time_range_fast(
        self,
        start_time: str,
        end_time: str,
        station_ids: Optional[List[str]] = None,
        min_rainfall: float = 0.0,
    ) -> pd.DataFrame:
        """
        快速查询版本 - 返回指定降雨量阈值以上的数据

        Args:
            start_time: 开始时间
            end_time: 结束时间
            station_ids: 站点ID列表
            min_rainfall: 最小降雨量阈值（默认0.0，即包含所有数据）

        Returns:
            满足条件的降雨数据
        """
        conn = duckdb.connect(self.db_path)

        sql = """
            SELECT r.station_id, r.datetime_str, r.rainfall, r.hour, r.minute,
                s.lng, s.lat, s.name, s.region
            FROM rainfall_data r
            LEFT JOIN stations s ON r.station_id = s.station_id
            WHERE r.datetime_str >= ? AND r.datetime_str <= ?
            AND r.rainfall >= ?
        """
        params = [start_time, end_time, min_rainfall]

        if station_ids:
            placeholders = ",".join(["?" for _ in station_ids])
            sql += f" AND r.station_id IN ({placeholders})"
            params.extend(station_ids)

        sql += " ORDER BY r.datetime_str, r.station_id"

        result = conn.execute(sql, params).df()
        conn.close()

        return result

    def get_rainfall_summary(self, start_time: str, end_time: str) -> pd.DataFrame:
        """
        快速获取降雨汇总统计 - 适合机器学习训练
        """
        conn = duckdb.connect(self.db_path)

        sql = """
            SELECT station_id, 
                   COUNT(*) as record_count,
                   SUM(rainfall) as total_rainfall,
                   AVG(rainfall) as avg_rainfall,
                   MAX(rainfall) as max_rainfall,
                   MIN(datetime_str) as first_record,
                   MAX(datetime_str) as last_record
            FROM rainfall_data
            WHERE datetime_str >= ? AND datetime_str <= ?
              AND rainfall > 0
            GROUP BY station_id
            ORDER BY total_rainfall DESC
        """

        result = conn.execute(sql, [start_time, end_time]).df()
        conn.close()

        return result

    def get_hourly_aggregated_data(
        self, start_time: str, end_time: str, min_rainfall: float = 0.1
    ) -> pd.DataFrame:
        """
        获取按小时聚合的降雨数据 - 适合时序分析
        """
        conn = duckdb.connect(self.db_path)

        sql = """
            SELECT 
                station_id,
                DATE_TRUNC('hour', CAST(datetime_str AS TIMESTAMP)) as hour_timestamp,
                AVG(rainfall) as avg_rainfall,
                MAX(rainfall) as max_rainfall,
                SUM(rainfall) as total_rainfall,
                COUNT(*) as record_count
            FROM rainfall_data
            WHERE datetime_str >= ? AND datetime_str <= ?
              AND rainfall >= ?
            GROUP BY station_id, DATE_TRUNC('hour', CAST(datetime_str AS TIMESTAMP))
            ORDER BY hour_timestamp, station_id
        """

        result = conn.execute(sql, [start_time, end_time, min_rainfall]).df()
        conn.close()

        return result

    def get_spatial_rainfall_matrix(
        self, start_time: str, end_time: str, grid_size: float = 0.1
    ) -> pd.DataFrame:
        """
        获取空间网格化的降雨数据 - 适合空间分析

        Args:
            grid_size: 网格大小（度）
        """
        conn = duckdb.connect(self.db_path)

        sql = """
            SELECT 
                ROUND(s.lng / ?, 0) * ? as grid_lng,
                ROUND(s.lat / ?, 0) * ? as grid_lat,
                COUNT(DISTINCT r.station_id) as station_count,
                AVG(r.rainfall) as avg_rainfall,
                MAX(r.rainfall) as max_rainfall,
                SUM(r.rainfall) as total_rainfall
            FROM rainfall_data r
            JOIN stations s ON r.station_id = s.station_id
            WHERE r.datetime_str >= ? AND r.datetime_str <= ?
              AND r.rainfall > 0
            GROUP BY grid_lng, grid_lat
            HAVING station_count >= 2
            ORDER BY grid_lng, grid_lat
        """

        result = conn.execute(
            sql, [grid_size, grid_size, grid_size, grid_size, start_time, end_time]
        ).df()
        conn.close()

        return result

    def get_training_dataset(
        self,
        start_time: str,
        end_time: str,
        time_step_hours: int = 1,
        min_stations_per_time: int = 10,
    ) -> pd.DataFrame:
        """
        生成机器学习训练数据集

        Args:
            time_step_hours: 时间步长（小时）
            min_stations_per_time: 每个时间点最少站点数
        """
        conn = duckdb.connect(self.db_path)

        sql = """
            WITH time_grid AS (
                SELECT 
                    DATE_TRUNC('hour', CAST(datetime_str AS TIMESTAMP)) + 
                    INTERVAL (EXTRACT(hour FROM CAST(datetime_str AS TIMESTAMP)) / ?) * ? hour as time_bucket,
                    station_id,
                    AVG(rainfall) as rainfall,
                    AVG(s.lng) as lng,
                    AVG(s.lat) as lat
                FROM rainfall_data r
                JOIN stations s ON r.station_id = s.station_id
                WHERE datetime_str >= ? AND datetime_str <= ?
                GROUP BY time_bucket, station_id
            ),
            time_counts AS (
                SELECT 
                    time_bucket,
                    COUNT(*) as station_count
                FROM time_grid
                GROUP BY time_bucket
                HAVING COUNT(*) >= ?
            )
            SELECT 
                tg.time_bucket,
                tg.station_id,
                tg.rainfall,
                tg.lng,
                tg.lat
            FROM time_grid tg
            JOIN time_counts tc ON tg.time_bucket = tc.time_bucket
            ORDER BY tg.time_bucket, tg.station_id
        """

        result = conn.execute(
            sql, [time_step_hours, time_step_hours, start_time, end_time, min_stations_per_time]
        ).df()
        conn.close()

        return result

    def get_station_features(self) -> pd.DataFrame:
        """
        获取站点特征数据 - 包括地理位置和历史统计
        """
        conn = duckdb.connect(self.db_path)

        sql = """
            SELECT 
                s.station_id,
                s.lng,
                s.lat,
                s.region,
                s.altitude,
                COUNT(r.station_id) as total_records,
                AVG(r.rainfall) as avg_rainfall,
                STDDEV(r.rainfall) as std_rainfall,
                MAX(r.rainfall) as max_rainfall,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY r.rainfall) as median_rainfall,
                PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY r.rainfall) as p95_rainfall,
                COUNT(CASE WHEN r.rainfall > 0.1 THEN 1 END) as rainy_records,
                MIN(r.datetime_str) as first_record,
                MAX(r.datetime_str) as last_record
            FROM stations s
            LEFT JOIN rainfall_data r ON s.station_id = r.station_id
            GROUP BY s.station_id, s.lng, s.lat, s.region, s.altitude
            ORDER BY s.station_id
        """

        result = conn.execute(sql).df()
        conn.close()

        return result

    def optimize_database(self):
        """
        优化数据库性能 - 针对查询优化
        """
        conn = duckdb.connect(self.db_path)

        # 分析表统计信息
        conn.execute("ANALYZE stations")
        conn.execute("ANALYZE rainfall_data")

        # 创建额外的优化索引
        try:
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_rainfall_spatial 
                ON rainfall_data(station_id, datetime_str, rainfall)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_stations_spatial 
                ON stations(lng, lat)
            """)

            log_print("Database optimization completed", "info")
        except Exception as e:
            log_print(f"Optimization warning: {e}", "warning")

        conn.close()


def main():
    """主函数 - 示例用法"""
    # 创建导入器实例
    importer = RainDataImporterDuckDB("data/rainfall_database.duckdb")

    # 导入雨量站基础信息
    print("Importing station information...")
    importer.import_station_info()

    # 批量导入data目录下的Excel文件
    data_directory = "data/rain/202301"
    if os.path.exists(data_directory):
        print(f"Importing Excel files from {data_directory}...")
        # 只导入1小时和24小时的数据
        sheets_to_import = [f"{i}小时" for i in range(1, 24)]
        importer.import_directory(data_directory, "*.xlsx", sheets_to_import)
    else:
        print(f"Data directory not found: {data_directory}")

        # 导入单个文件示例
        sample_file = "data/rain/分钟级雨量站数据_20230603.xlsx"
        if os.path.exists(sample_file):
            print(f"Importing sample file: {sample_file}")
            importer.import_excel_file(sample_file, ["1小时", "12小时"])

    # # 显示统计信息
    print("\n=== Database Statistics ===")
    stats = importer.get_statistics()
    print(f"Total stations: {stats['station_count']}")
    print(f"Total rainfall records: {stats['rainfall_record_count']}")
    print(f"Time range: {stats['time_range'][0]} to {stats['time_range'][1]}")

    print("\nFile statistics:")
    for file_name, count in stats["file_statistics"]:
        print(f"  {file_name}: {count} records")

    # 查询示例
    print("\n=== Query Example ===")
    try:
        # 查询某个时间段的数据
        # test speed
        import time
        t1 = time.time()
        sample_data = importer.query_by_time_range_fast(
            "2023-06-03 00:00:00", "2023-06-03 20:59:59", min_rainfall=0.05
        )
        t2 = time.time()
        print(f"Sample query took {t2 - t1:.2f} seconds")

        print(f"Sample query returned {len(sample_data)} records")

        if len(sample_data) > 0:
            print("Sample data preview:")
            print(sample_data[["station_id", "datetime_str", "rainfall", "lng", "lat"]].head())

    except Exception as e:
        print(f"Query example failed: {e}")


if __name__ == "__main__":
    main()
