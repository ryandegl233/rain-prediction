"""
雨量站数据导入工具 - 分片DuckDB版本
按日期分片存储，提高文件管理效率和并发性能
"""

import glob
import json
import multiprocessing
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
from loguru import logger

from src.dataset.rain_station_mapped import load_rain_station_info, read_excel_fast
from src.utils.logging import log_print


class ShardedRainDataImporter:
    """分片式雨量站数据导入器 - DuckDB版本"""

    def __init__(self, base_dir: str = "data2/rainfall_shards"):
        """
        初始化分片导入器

        Args:
            base_dir: 分片数据库基础目录
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True, parents=True)

        # 元数据数据库路径（包含站点信息和分片索引）
        self.meta_db_path = self.base_dir / "metadata.duckdb"
        self.shard_index_file = self.base_dir / "shard_index.json"

        # 分片索引缓存
        self._shard_index = {}

        self.ensure_metadata_database()
        self.load_shard_index()

    def _resolve_shard_db_path(self, shard_id: str, file_path: str | None) -> str:
        """
        Resolve shard DB path robustly.

        `shard_index.json` might contain stale/absolute paths (e.g. moved folders). Prefer existing paths under base_dir.
        """
        candidates: list[Path] = []
        if file_path:
            p = Path(file_path)
            candidates.append(p)
            if not p.is_absolute():
                candidates.append(self.base_dir / p)
            candidates.append(self.base_dir / p.name)
        candidates.append(self.get_shard_path(shard_id))

        for cand in candidates:
            if cand.exists():
                return cand.as_posix()
        # fallback to the most reasonable location (even if missing, keeps error messages stable)
        return (self.base_dir / f"{shard_id}.duckdb").as_posix()

    def ensure_metadata_database(self):
        """创建元数据数据库"""
        conn = duckdb.connect(str(self.meta_db_path))

        # 站点信息表
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

        # 分片索引表
        conn.execute("""
            CREATE TABLE IF NOT EXISTS shard_index (
                shard_id VARCHAR PRIMARY KEY,
                start_date DATE NOT NULL,
                end_date DATE NOT NULL,
                file_path VARCHAR NOT NULL,
                record_count BIGINT DEFAULT 0,
                file_size_mb DOUBLE DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.close()
        log_print("Metadata database initialized", "info")

    def load_shard_index(self):
        """从文件加载分片索引缓存"""
        # import pdb; pdb.set_trace()
        if self.shard_index_file.exists():
            try:
                with open(self.shard_index_file, "r") as f:
                    self._shard_index = json.load(f)
                log_print(f"Loaded shard index with {len(self._shard_index)} shards", "debug")

                # Normalize file paths (handles moved base_dir / stale paths in index).
                changed = False
                for shard_id, shard_info in self._shard_index.items():
                    if not isinstance(shard_info, dict):
                        continue
                    resolved = self._resolve_shard_db_path(shard_id, shard_info.get("file_path"))
                    if shard_info.get("file_path") != resolved:
                        shard_info["file_path"] = resolved
                        changed = True
                if changed:
                    log_print("Normalized shard_index file paths based on current base_dir", "debug")

                # import pdb; pdb.set_trace()
            except Exception as e:
                log_print(f"Failed to load shard index: {e}", "warning")
                self._shard_index = {}
        else:
            self._shard_index = {}

    def save_shard_index(self):
        """保存分片索引缓存到文件"""
        try:
            with open(self.shard_index_file, "w") as f:
                json.dump(self._shard_index, f, indent=2)
        except Exception as e:
            log_print(f"Failed to save shard index: {e}", "warning")

    def get_shard_id(self, date_str: str) -> str:
        """根据日期获取分片ID（按月分片）"""
        if isinstance(date_str, str):
            if len(date_str) >= 10:
                date_obj = datetime.strptime(date_str[:10], "%Y-%m-%d")
            else:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
        else:
            date_obj = date_str
        return f"shard_{date_obj.strftime('%Y%m')}"

    def get_shard_path(self, shard_id: str) -> Path:
        """获取分片文件路径"""
        return self.base_dir / f"{shard_id}.duckdb"

    def create_shard(self, shard_id: str, date: str) -> bool:
        """创建新的分片文件（按月分片）"""
        shard_path = self.get_shard_path(shard_id)

        if shard_path.exists():
            return True

        try:
            # 计算月份的开始和结束日期
            date_obj = datetime.strptime(date, "%Y-%m-%d")
            start_date = date_obj.replace(day=1)
            # 下个月的第一天
            if date_obj.month == 12:
                next_month = date_obj.replace(year=date_obj.year + 1, month=1, day=1)
            else:
                next_month = date_obj.replace(month=date_obj.month + 1, day=1)
            # 本月最后一天
            end_date = next_month - timedelta(days=1)

            start_date_str = start_date.strftime("%Y-%m-%d")
            end_date_str = end_date.strftime("%Y-%m-%d")

            conn = duckdb.connect(str(shard_path))

            # 在分片中创建降雨数据表
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

            # 创建索引
            conn.execute("CREATE INDEX IF NOT EXISTS idx_datetime ON rainfall_data(datetime_str)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_station ON rainfall_data(station_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_rainfall ON rainfall_data(rainfall)")

            conn.close()

            # 更新分片索引
            self._shard_index[shard_id] = {
                "start_date": start_date_str,
                "end_date": end_date_str,
                "file_path": str(shard_path),
                "record_count": 0,
                "file_size_mb": 0.0,
            }

            # 同时更新数据库中的分片索引
            meta_conn = duckdb.connect(str(self.meta_db_path))
            meta_conn.execute(
                """
                INSERT OR REPLACE INTO shard_index
                (shard_id, start_date, end_date, file_path, record_count, file_size_mb)
                VALUES (?, ?, ?, ?, 0, 0)
            """,
                [shard_id, start_date_str, end_date_str, str(shard_path)],
            )
            meta_conn.close()

            self.save_shard_index()
            log_print(
                f"Created new monthly shard: {shard_id} for period {start_date_str} to {end_date_str}",
                "info",
            )
            return True

        except Exception as e:
            log_print(f"Failed to create shard {shard_id}: {e}", "error")
            return False

    def import_station_info(self, station_file: str = "data2/四川省雨量站信息.csv"):
        """导入雨量站基础信息到元数据库（批量操作优化版）"""
        if not os.path.exists(station_file):
            log_print(f"Station file not found: {station_file}", "warning")
            return

        try:
            station_info = load_rain_station_info(station_file)
            
            # 准备数据 - 确保列类型正确
            station_info = station_info.copy()
            station_info["lng"] = station_info["lng"].astype(float)
            station_info["lat"] = station_info["lat"].astype(float)
            
            # 确保必要的列存在
            if "name" not in station_info.columns:
                station_info["name"] = ""
            if "region" not in station_info.columns:
                station_info["region"] = ""
            if "altitude" not in station_info.columns:
                station_info["altitude"] = None
                
            # 选择需要的列
            cols_to_insert = ["station_id", "lng", "lat", "name", "region", "altitude"]
            df_to_insert = station_info[cols_to_insert].copy()

            conn = duckdb.connect(str(self.meta_db_path))

            # 统计导入前的记录数
            before_count = conn.execute("SELECT COUNT(*) FROM stations").fetchone()[0]

            # 使用事务和批量 UPSERT
            conn.execute("BEGIN TRANSACTION")
            try:
                # 注册 DataFrame 到 DuckDB（这样才能在 SQL 中引用）
                conn.register("df_to_insert", df_to_insert)
                
                # 批量插入或替换（DuckDB 支持 INSERT OR REPLACE）
                conn.execute("""
                    INSERT OR REPLACE INTO stations (station_id, lng, lat, name, region, altitude)
                    SELECT station_id, lng, lat, name, region, altitude FROM df_to_insert
                """)
                conn.execute("COMMIT")

                # 统计导入后的记录数
                after_count = conn.execute("SELECT COUNT(*) FROM stations").fetchone()[0]

                imported_count = len(df_to_insert)
                new_count = after_count - before_count
                updated_count = imported_count - new_count

                log_print(
                    f"Station info imported: {new_count} new, {updated_count} updated (total {after_count} stations)",
                    "info",
                )
            except Exception as e:
                conn.execute("ROLLBACK")
                raise e
            finally:
                conn.close()

        except Exception as e:
            log_print(f"Error importing station info: {e}", "error")
            raise

    def import_excel_file(
        self,
        file_path: str,
        sheets_to_import: Optional[List[str]] = None,
        include_zeros: bool = False,
    ):
        """导入单个Excel文件的数据到相应分片"""
        if not os.path.exists(file_path):
            log_print(f"File not found: {file_path}", "error")
            return

        try:
            # 获取文件中的所有工作表
            with pd.ExcelFile(file_path) as xls:
                available_sheets = xls.sheet_names

            # 确定要导入的工作表
            if sheets_to_import is None:
                sheets_to_import = [str(sheet) for sheet in available_sheets]
            else:
                # 过滤并转换为字符串类型
                sheets_to_import = [str(sheet) for sheet in sheets_to_import if str(sheet) in available_sheets]

            if not sheets_to_import:
                log_print(f"No valid sheets found in {file_path}", "warning")
                return

            log_print(f"Importing from {file_path}, sheets: {sheets_to_import}", "info")

            total_imported = 0
            file_name = os.path.basename(file_path)

            for sheet_name in sheets_to_import:
                try:
                    # 读取Excel数据
                    rain_data = read_excel_fast(file_path, sheet_name)
                    log_print(
                        f"Read {len(rain_data)} records from sheet '{sheet_name}'",
                        "debug",
                    )

                    if len(rain_data) == 0:
                        continue

                    # 处理时间数据
                    rain_data["time"] = pd.to_datetime(rain_data["数据时间戳"], unit="s")
                    rain_data["time"] = rain_data["time"] + pd.Timedelta(hours=8)
                    rain_data["timestamp"] = rain_data["数据时间戳"]
                    rain_data["hour"] = rain_data["time"].dt.hour
                    rain_data["minute"] = rain_data["time"].dt.minute
                    rain_data["date"] = rain_data["time"].dt.date.astype(str)
                    rain_data["datetime_str"] = rain_data["time"].dt.strftime("%Y-%m-%d %H:%M:%S")

                    # 按日期分组数据
                    date_groups = rain_data.groupby("date")

                    for date, group_data in date_groups:
                        imported_count = self._import_dataframe_to_shard(
                            group_data,
                            str(date),
                            file_name,
                            sheet_name,
                            include_zeros=include_zeros,
                        )
                        total_imported += imported_count

                        log_print(
                            f"Imported {imported_count} records for date {date}",
                            "debug",
                        )

                except Exception as e:
                    log_print(f"Error importing sheet '{sheet_name}': {e}", "error")
                    continue

            log_print(f"Total imported from {file_name}: {total_imported} records", "info")

        except Exception as e:
            log_print(f"Error importing file {file_path}: {e}", "error")
            raise

    def _import_dataframe_to_shard(
        self,
        df: pd.DataFrame,
        date: str,
        source_file: str,
        source_sheet: str,
        include_zeros: bool = True,
    ) -> int:
        """将DataFrame导入到对应日期的分片（存储所有有效数据，包括0值）

        Args:
            df: 要导入的数据框
            date: 数据日期
            source_file: 源文件名
            source_sheet: 源工作表名
            include_zeros: 是否包含0值数据（默认True，存储所有数据）

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

        # 决定要存储的数据
        if include_zeros:
            # 存储所有有效数据（包括0值）
            data_to_store = valid_df.copy()
            log_print(
                f"Storing all {len(data_to_store)} valid records (including zeros) for date {date}",
                "debug",
            )
        else:
            # 仅存储有降雨的数据（向后兼容）
            rainfall_threshold = 0.05  # minimal rainfall
            data_to_store = valid_df[valid_df["rainfall"] > rainfall_threshold].copy()
            skipped_records = len(valid_df) - len(data_to_store)
            log_print(
                f"Storing {len(data_to_store)} rainfall records (>{rainfall_threshold}mm), skipping {skipped_records} zero/low values for date {date}",
                "debug",
            )

        if len(data_to_store) == 0:
            return 0

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
        insert_df = data_to_store[columns_to_insert]

        # 获取分片信息
        shard_id = self.get_shard_id(date)

        # 确保分片存在
        if not self.create_shard(shard_id, date):
            log_print(f"Failed to create shard for date {date}", "error")
            return 0

        # 导入数据到分片
        shard_path = self.get_shard_path(shard_id)

        try:
            conn = duckdb.connect(str(shard_path))

            # 高速批量插入 - 直接使用DuckDB的DataFrame插入
            conn.execute("""
                INSERT OR REPLACE INTO rainfall_data
                (station_id, timestamp, datetime_str, rainfall, status, hour, minute, date, source_file, source_sheet)
                SELECT * FROM insert_df
            """)

            imported_count = len(insert_df)
            conn.close()

            # 更新分片统计信息
            self._update_shard_stats(shard_id)

            return imported_count

        except Exception as e:
            log_print(f"Error importing to shard {shard_id}: {e}", "error")
            return 0

    def _update_shard_stats(self, shard_id: str):
        """更新分片统计信息"""
        shard_path = self.get_shard_path(shard_id)

        if not shard_path.exists():
            return

        try:
            # 获取记录数
            conn = duckdb.connect(str(shard_path))
            result = conn.execute("SELECT COUNT(*) FROM rainfall_data").fetchone()
            record_count = result[0] if result else 0
            conn.close()

            # 获取文件大小
            file_size_mb = shard_path.stat().st_size / (1024 * 1024)

            # 更新缓存
            if shard_id in self._shard_index:
                self._shard_index[shard_id]["record_count"] = record_count
                self._shard_index[shard_id]["file_size_mb"] = file_size_mb

            # 更新数据库
            meta_conn = duckdb.connect(str(self.meta_db_path))
            meta_conn.execute(
                """
                UPDATE shard_index
                SET record_count = ?, file_size_mb = ?, updated_at = CURRENT_TIMESTAMP
                WHERE shard_id = ?
            """,
                [record_count, file_size_mb, shard_id],
            )
            meta_conn.close()

            self.save_shard_index()

        except Exception as e:
            log_print(f"Failed to update shard stats for {shard_id}: {e}", "warning")

    def import_directory(
        self,
        directory: str,
        pattern: str = "*.xlsx",
        sheets_to_import: Optional[List[str]] = None,
        include_zeros: bool = False,
    ):
        """批量导入目录中的Excel文件"""
        if not os.path.exists(directory):
            log_print(f"Directory not found: {directory}", "error")
            return

        file_pattern = os.path.join(directory, pattern)
        excel_files = glob.glob(file_pattern)

        if not excel_files:
            log_print(f"No Excel files found in {directory} with pattern {pattern}", "warning")
            return

        log_print(f"Found {len(excel_files)} Excel files to import", "info")

        for file_path in excel_files:
            try:
                log_print(f"Processing file: {file_path}", "info")
                self.import_excel_file(file_path, sheets_to_import, include_zeros)
            except Exception as e:
                log_print(f"Error processing file {file_path}: {e}", "error")
                continue

    def get_relevant_shards(self, start_date: str, end_date: str) -> List[Dict]:
        """获取指定日期范围内的相关分片"""
        relevant_shards = []

        for shard_id, shard_info in self._shard_index.items():
            shard_start = shard_info["start_date"]
            shard_end = shard_info["end_date"]
            shard_path = self._resolve_shard_db_path(shard_id, shard_info.get("file_path"))

            # 检查日期范围是否重叠
            if shard_start <= end_date and shard_end >= start_date:
                if os.path.exists(shard_path):
                    relevant_shards.append(
                        {
                            "shard_id": shard_id,
                            "file_path": shard_path,
                            "start_date": shard_start,
                            "end_date": shard_end,
                        }
                    )

        return relevant_shards

    def query_by_time_range(
        self,
        start_time: str,
        end_time: str,
        station_ids: Optional[List[str]] = None,
        min_rainfall: float = 0.0,
    ) -> pd.DataFrame:
        """跨分片查询数据"""
        # 提取日期范围
        start_date = start_time[:10]
        end_date = end_time[:10]

        # 获取相关分片
        relevant_shards = self.get_relevant_shards(start_date, end_date)

        if not relevant_shards:
            log_print(f"No shards found for date range {start_date} to {end_date}", "warning")
            return pd.DataFrame()

        # log_print(
        #     f"Querying {len(relevant_shards)} shards for date range {start_date} to {end_date}",
        #     "debug",
        # )

        all_results = []

        # 查询每个相关分片
        for shard_info in relevant_shards:
            try:
                conn = duckdb.connect(shard_info["file_path"])

                # 构建查询SQL
                sql = """
                    SELECT r.station_id, r.datetime_str, r.rainfall, r.hour, r.minute,
                           r.status, r.source_file, r.source_sheet
                    FROM rainfall_data r
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

                if len(result) > 0:
                    all_results.append(result)

            except Exception as e:
                log_print(f"Error querying shard {shard_info['shard_id']}: {e}", "error")
                continue

        if not all_results:
            return pd.DataFrame()

        # 合并所有结果
        combined_result = pd.concat(all_results, ignore_index=True)

        # 添加站点信息
        if len(combined_result) > 0:
            # 从元数据库获取站点信息
            meta_conn = duckdb.connect(str(self.meta_db_path))

            unique_stations = combined_result["station_id"].unique()
            if len(unique_stations) > 0:
                placeholders = ",".join(["?" for _ in unique_stations])
                stations_sql = f"""
                    SELECT station_id, lng, lat, name, region
                    FROM stations
                    WHERE station_id IN ({placeholders})
                """
                stations_df = meta_conn.execute(stations_sql, list(unique_stations)).df()

                # 合并站点信息
                combined_result = combined_result.merge(stations_df, on="station_id", how="left")

            meta_conn.close()

        # 按时间和站点排序
        if len(combined_result) > 0:
            combined_result = combined_result.sort_values(["datetime_str", "station_id"]).reset_index(drop=True)

        return combined_result

    def query_by_time_range_with_fill(
        self,
        start_time: str,
        end_time: str,
        station_ids: Optional[List[str]] = None,
        fill_zeros: bool = False,
        time_interval_minutes: int = 60,
    ) -> pd.DataFrame:
        """查询数据（由于已存储所有数据包括0值，通常不需要填充）

        Args:
            start_time: 开始时间
            end_time: 结束时间
            station_ids: 站点ID列表
            fill_zeros: 是否填充零值生成完整时间序列（默认False，因为数据库已包含0值）
            time_interval_minutes: 时间间隔（分钟）

        Returns:
            查询结果DataFrame
        """
        # 直接查询数据库中的所有数据（包括0值）
        stored_data = self.query_by_time_range(
            start_time=start_time,
            end_time=end_time,
            station_ids=station_ids,
            min_rainfall=0.0,  # 获取所有存储的数据，包括0值
        )

        # 如果需要填充（通常不需要，因为数据库已存储0值）
        if fill_zeros and len(stored_data) > 0:
            return self._fill_missing_timestamps(stored_data, start_time, end_time, station_ids, time_interval_minutes)

        return stored_data

    def _fill_missing_timestamps(
        self,
        existing_data: pd.DataFrame,
        start_time: str,
        end_time: str,
        station_ids: Optional[List[str]] = None,
        time_interval_minutes: int = 60,
    ) -> pd.DataFrame:
        """填充缺失的时间戳，为无降雨时间点补充零值

        Args:
            existing_data: 已存储的降雨数据
            start_time: 开始时间
            end_time: 结束时间
            station_ids: 站点ID列表
            time_interval_minutes: 时间间隔（分钟）

        Returns:
            包含完整时间序列的DataFrame
        """
        try:
            # 解析时间范围
            start_dt = pd.to_datetime(start_time)
            end_dt = pd.to_datetime(end_time)

            # 生成完整时间序列
            time_range = pd.date_range(start=start_dt, end=end_dt, freq=f"{time_interval_minutes}min")

            # 获取需要处理的站点列表
            if station_ids is None:
                # 如果没有指定站点，从元数据库获取所有站点
                meta_conn = duckdb.connect(str(self.meta_db_path))
                stations_result = meta_conn.execute("SELECT station_id, lng, lat, name, region FROM stations").df()
                meta_conn.close()
                target_stations = stations_result["station_id"].tolist()
            else:
                target_stations = station_ids
                # 获取站点信息
                meta_conn = duckdb.connect(str(self.meta_db_path))
                placeholders = ",".join(["?" for _ in target_stations])
                stations_sql = f"""
                    SELECT station_id, lng, lat, name, region
                    FROM stations
                    WHERE station_id IN ({placeholders})
                """
                stations_result = meta_conn.execute(stations_sql, target_stations).df()
                meta_conn.close()

            # 创建完整的时间x站点组合
            full_combinations = []
            for timestamp in time_range:
                for station_id in target_stations:
                    full_combinations.append(
                        {
                            "station_id": station_id,
                            "datetime_str": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                            "timestamp": int(timestamp.timestamp()),
                            "hour": timestamp.hour,
                            "minute": timestamp.minute,
                            "date": timestamp.strftime("%Y-%m-%d"),
                            "rainfall": 0.0,  # 默认填充零值
                            "status": "filled",  # 标记为填充数据
                            "source_file": "filled_data",
                            "source_sheet": "filled_data",
                        }
                    )

            full_df = pd.DataFrame(full_combinations)

            # 将已存在的数据合并到完整序列中（替换零值）
            if len(existing_data) > 0:
                # 为现有数据添加合并键
                existing_data["merge_key"] = existing_data["station_id"] + "_" + existing_data["datetime_str"]
                full_df["merge_key"] = full_df["station_id"] + "_" + full_df["datetime_str"]

                # 用现有数据更新完整序列
                full_df = full_df.set_index("merge_key")
                existing_df = existing_data.set_index("merge_key")

                # 更新存在的记录
                for key in existing_df.index:
                    if key in full_df.index:
                        full_df.loc[key, "rainfall"] = existing_df.loc[key, "rainfall"]
                        full_df.loc[key, "status"] = existing_df.loc[key, "status"]
                        full_df.loc[key, "source_file"] = existing_df.loc[key, "source_file"]
                        full_df.loc[key, "source_sheet"] = existing_df.loc[key, "source_sheet"]

                full_df = full_df.reset_index(drop=True)

                # 移除合并键列
                if "merge_key" in full_df.columns:
                    full_df = full_df.drop("merge_key", axis=1)

            # 添加站点信息
            if len(stations_result) > 0:
                full_df = full_df.merge(stations_result, on="station_id", how="left")

            # 按时间和站点排序
            full_df = full_df.sort_values(["datetime_str", "station_id"]).reset_index(drop=True)

            log_print(
                f"Generated complete time series: {len(full_df)} records (original: {len(existing_data)}, filled: {len(full_df) - len(existing_data)})",
                "debug",
            )

            return full_df

        except Exception as e:
            log_print(f"Error filling missing timestamps: {e}", "error")
            return existing_data

    def get_statistics(self) -> Dict:
        """获取分片数据库统计信息"""
        try:
            # 从元数据库获取站点统计
            meta_conn = duckdb.connect(str(self.meta_db_path))
            station_result = meta_conn.execute("SELECT COUNT(*) FROM stations").fetchone()
            station_count = station_result[0] if station_result else 0

            # 获取分片统计
            shard_stats = meta_conn.execute("""
                SELECT COUNT(*) as shard_count,
                       SUM(record_count) as total_records,
                       SUM(file_size_mb) as total_size_mb,
                       MIN(start_date) as earliest_date,
                       MAX(end_date) as latest_date
                FROM shard_index
            """).fetchone()

            meta_conn.close()

            # 获取分片详情
            shard_details = []
            for shard_id, shard_info in self._shard_index.items():
                shard_details.append(
                    {
                        "shard_id": shard_id,
                        "date_range": f"{shard_info['start_date']} to {shard_info['end_date']}",
                        "record_count": shard_info["record_count"],
                        "file_size_mb": round(shard_info["file_size_mb"], 2),
                    }
                )

            return {
                "station_count": station_count,
                "shard_count": shard_stats[0] if shard_stats and shard_stats[0] else 0,
                "total_records": shard_stats[1] if shard_stats and shard_stats[1] else 0,
                "total_size_mb": round(shard_stats[2], 2) if shard_stats and shard_stats[2] else 0,
                "date_range": (shard_stats[3], shard_stats[4])
                if shard_stats and shard_stats[3] and shard_stats[4]
                else (None, None),
                "shard_details": shard_details,
            }

        except Exception as e:
            log_print(f"Error getting statistics: {e}", "error")
            return {
                "station_count": 0,
                "shard_count": 0,
                "total_records": 0,
                "total_size_mb": 0,
                "date_range": (None, None),
                "shard_details": [],
            }

    def optimize_shards(self):
        """优化所有分片的性能"""
        log_print("Optimizing all shards...", "info")

        for shard_id, shard_info in self._shard_index.items():
            shard_path = Path(shard_info["file_path"])

            if shard_path.exists():
                try:
                    conn = duckdb.connect(str(shard_path))
                    conn.execute("ANALYZE rainfall_data")
                    conn.close()
                    log_print(f"Optimized shard {shard_id}", "debug")
                except Exception as e:
                    log_print(f"Failed to optimize shard {shard_id}: {e}", "warning")

        log_print("Shard optimization completed", "info")

    def cleanup_empty_shards(self):
        """清理空的分片文件"""
        removed_count = 0

        for shard_id, shard_info in list(self._shard_index.items()):
            if shard_info["record_count"] == 0:
                shard_path = Path(shard_info["file_path"])

                if shard_path.exists():
                    try:
                        shard_path.unlink()
                        log_print(f"Removed empty shard file: {shard_path}", "info")
                    except Exception as e:
                        log_print(f"Failed to remove shard file {shard_path}: {e}", "warning")
                        continue

                # 从索引中移除
                del self._shard_index[shard_id]

                # 从数据库中移除
                try:
                    meta_conn = duckdb.connect(str(self.meta_db_path))
                    meta_conn.execute("DELETE FROM shard_index WHERE shard_id = ?", [shard_id])
                    meta_conn.close()
                except Exception as e:
                    log_print(f"Failed to remove shard from database: {e}", "warning")

                removed_count += 1

        if removed_count > 0:
            self.save_shard_index()
            log_print(f"Cleaned up {removed_count} empty shards", "info")

    def meshgrid_rain(
        self,
        start_time: str,
        end_time: str,
        station_ids: list[str] | None = None,
        min_rainfall: float = 0.0,
        grid_width: int = 1200,
        grid_height: int = 900,
        target_proj: str = "epsg:4326",
        force_source_proj: str | None = None,
        interpolate: str | None = None,
        bounds: tuple[float, float, float, float] | None = None,
        verbose: bool = False,
    ) -> dict:
        """
        将四川省地图离散为网格，根据雨量站数据生成降雨量分布图

        使用与雷达/卫星数据相同的投影和网格系统，确保数据格式兼容性

        Args:
            start_time: 开始时间
            end_time: 结束时间
            station_ids: 指定站点ID列表，None表示使用所有站点
            min_rainfall: 最小降雨量阈值
            grid_width: 网格宽度 (默认1200，约1km分辨率)
            grid_height: 网格高度 (默认900，约1km分辨率)
            target_proj: 目标投影坐标系 (默认"epsg:4326")
            force_source_proj: 强制指定源投影
            interpolate: 插值方法 ("nearest", "linear", None)
            bounds: 自定义边界 (lon_min, lon_max, lat_min, lat_max)

        Returns:
            dict: 包含网格数据、坐标信息和元数据的字典
                - mapped_data: 形状为(h, w)的降雨量网格数组
                - x_mesh: X坐标网格
                - y_mesh: Y坐标网格
                - lon_mesh: 经度坐标网格
                - lat_mesh: 纬度坐标网格
                - bounds_target: 目标坐标系边界
                - bounds_lonlat: 经纬度边界
                - source_proj: 源投影
                - target_proj: 目标投影
                - statistics: 统计信息
        """
        import numpy as np

        # 导入投影工具
        try:
            from src.dataset.geo_utils import (
                create_display_coordinates,
                create_output_grid,
                detect_projection,
                setup_coordinate_transform,
            )
        except ImportError:
            log_print("Failed to import geo_utils, using simple grid method", "warning")
            return self._meshgrid_rain_simple(start_time, end_time, station_ids, min_rainfall, grid_width, grid_height)

        # 默认四川省边界
        default_bounds = (
            97.0,
            109.0,
            26.0,
            35.0,
        )  # (lon_min, lon_max, lat_min, lat_max)

        # 查询指定时间范围内的降雨数据
        rainfall_data = self.query_by_time_range(
            start_time=start_time,
            end_time=end_time,
            station_ids=station_ids,
            min_rainfall=min_rainfall,
        )

        if len(rainfall_data) == 0:
            log_print(f"No rainfall data found for the specified time range ({start_time}, {end_time})", "warning")
            # 创建空的输出网格
            final_bounds = bounds if bounds is not None else default_bounds
            x_mesh, y_mesh = create_output_grid(final_bounds, grid_width, grid_height)
            mapped_data = np.zeros_like(x_mesh)
            lon_mesh, lat_mesh = create_display_coordinates(x_mesh, y_mesh, target_proj)

            return {
                "mapped_data": mapped_data,
                "x_mesh": x_mesh,
                "y_mesh": y_mesh,
                "lon_mesh": lon_mesh,
                "lat_mesh": lat_mesh,
                "bounds_target": final_bounds,
                "bounds_lonlat": final_bounds,
                "source_proj": "epsg:4326",
                "target_proj": target_proj,
                "statistics": {
                    "station_count": 0,
                    "non_zero_grids": 0,
                    "max_rainfall": 0.0,
                    "mean_rainfall": 0.0,
                },
            }

        # 确保数据包含经纬度信息
        if "lng" not in rainfall_data.columns or "lat" not in rainfall_data.columns:
            log_print("Rainfall data missing longitude/latitude information", "error")
            raise ValueError("Rainfall data must contain 'lng' and 'lat' columns")

        # 按站点聚合降雨数据（计算时间段内的平均降雨量）
        station_rainfall = rainfall_data.groupby(["station_id", "lng", "lat"]).agg({"rainfall": "mean"}).reset_index()

        log_print(
            f"Processing {len(station_rainfall)} stations with rainfall data at time ({start_time}, {end_time})", "info"
        )

        if len(station_rainfall) == 0:
            # 没有任何站点在该时间范围内有有效降雨数据，返回全零网格
            final_bounds = bounds if bounds is not None else default_bounds
            x_mesh, y_mesh = create_output_grid(final_bounds, grid_width, grid_height)
            mapped_data = np.zeros_like(x_mesh, dtype=np.float32)
            lon_mesh, lat_mesh = create_display_coordinates(x_mesh, y_mesh, target_proj)

            return {
                "mapped_data": mapped_data,
                "x_mesh": x_mesh,
                "y_mesh": y_mesh,
                "lon_mesh": lon_mesh,
                "lat_mesh": lat_mesh,
                "bounds_target": final_bounds,
                "bounds_lonlat": final_bounds,
                "grid_width": grid_width,
                "grid_height": grid_height,
                "source_proj": "epsg:4326",
                "target_proj": target_proj,
                "statistics": {
                    "station_count": 0,
                    "non_zero_grids": 0,
                    "max_rainfall": 0.0,
                    "mean_rainfall": 0.0,
                },
            }

        # 提取坐标和降雨量
        longitudes = np.array(station_rainfall["lng"].values)
        latitudes = np.array(station_rainfall["lat"].values)
        rainfall_values = np.array(station_rainfall["rainfall"].values)

        log_print(
            f"Rainfall range: [{rainfall_values.min():.2f}, {rainfall_values.max():.2f}] mm",
            "debug",
        )

        # 检测源投影
        if force_source_proj:
            source_proj = force_source_proj
        else:
            source_proj = detect_projection(longitudes, latitudes)

        # 设置坐标转换和网格
        transformer, calculated_bounds, is_regular_grid = setup_coordinate_transform(
            source_proj, target_proj, longitudes, latitudes
        )

        # 使用指定边界或计算出的边界
        final_bounds = bounds if bounds is not None else calculated_bounds

        # 创建输出网格
        x_mesh, y_mesh = create_output_grid(final_bounds, grid_width, grid_height)

        # 创建稀疏矩阵（将雨量站数据映射到网格点）
        mapped_data = np.zeros_like(x_mesh)
        grid_counts = np.zeros_like(x_mesh)  # 记录每个网格中的站点数量

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

            # 在该网格点累积降雨值
            if y_idx < mapped_data.shape[0] and x_idx < mapped_data.shape[1]:
                mapped_data[y_idx, x_idx] += rainfall
                grid_counts[y_idx, x_idx] += 1

        # 计算每个网格的平均降雨量
        mask = grid_counts > 0
        mapped_data[mask] = mapped_data[mask] / grid_counts[mask]

        # 可选的插值处理
        if interpolate is not None:
            mapped_data = self._apply_interpolation(mapped_data, interpolate)

        # 创建显示坐标
        lon_mesh, lat_mesh = create_display_coordinates(x_mesh, y_mesh, target_proj)

        # 统计站点雨量范围
        rainfall_bins = {
            "0.01-0.1": int(np.sum((rainfall_values > 0.01) & (rainfall_values <= 0.1))),
            "0.1-0.3": int(np.sum((rainfall_values > 0.1) & (rainfall_values <= 0.3))),
            "0.3-0.6": int(np.sum((rainfall_values > 0.3) & (rainfall_values <= 0.6))),
            "0.6-1.0": int(np.sum((rainfall_values > 0.6) & (rainfall_values <= 1.0))),
            ">1.0": int(np.sum(rainfall_values > 1.0)),
        }

        # 计算统计信息
        non_zero_grids = np.sum(mapped_data > 0)
        max_rainfall = np.max(mapped_data)
        mean_rainfall = np.mean(mapped_data[mapped_data > 0]) if non_zero_grids > 0 else 0

        if verbose:
            log_print(f"Grid statistics:", "info")
            log_print(f"  Grid size: {grid_height}x{grid_width}", "info")
            log_print(
                f"  Non-zero grids: {non_zero_grids}/{grid_width * grid_height} ({non_zero_grids / (grid_width * grid_height) * 100:.2f}%)",
                "info",
            )
            log_print(f"  Max rainfall: {max_rainfall:.2f}mm", "info")
            log_print(f"  Mean rainfall (non-zero): {mean_rainfall:.2f}mm", "info")

        # 获取原始边界
        x_min_src = float(longitudes.min())
        x_max_src = float(longitudes.max())
        y_min_src = float(latitudes.min())
        y_max_src = float(latitudes.max())

        return {
            "mapped_data": mapped_data.astype(np.float32),
            "x_mesh": x_mesh,
            "y_mesh": y_mesh,
            "lon_mesh": lon_mesh,
            "lat_mesh": lat_mesh,
            "bounds_target": final_bounds,
            "bounds_lonlat": (x_min_src, x_max_src, y_min_src, y_max_src),
            "grid_width": grid_width,
            "grid_height": grid_height,
            "source_proj": source_proj,
            "target_proj": target_proj,
            "statistics": {
                "station_count": len(station_rainfall),
                "non_zero_grids": int(non_zero_grids),
                "max_rainfall": float(max_rainfall),
                "mean_rainfall": float(mean_rainfall),
                "rainfall_bins": rainfall_bins,
            },
        }

    def _apply_interpolation(self, data: np.ndarray, method: str) -> np.ndarray:
        """应用插值方法填充空白区域"""
        if method.lower() == "nearest":
            from scipy.interpolate import griddata
            from scipy.spatial.distance import cdist

            # 获取有数据的网格坐标和值
            y_coords, x_coords = np.where(data > 0)
            if len(y_coords) == 0:
                return data

            values = data[y_coords, x_coords]
            height, width = data.shape

            # 创建完整的网格坐标
            yi, xi = np.mgrid[0:height, 0:width]
            points = np.column_stack((y_coords, x_coords))
            target_points = np.column_stack((yi.ravel(), xi.ravel()))

            # 计算到最近有数据点的距离
            distances = cdist(target_points, points).min(axis=1)
            # 只对距离最近数据点不超过10个网格的区域进行插值
            interpolate_mask = distances <= 10

            if np.sum(interpolate_mask) > 0:
                # 使用最近邻插值
                interpolated = griddata(
                    points,
                    values,
                    target_points[interpolate_mask],
                    method="nearest",
                    fill_value=0,
                )

                # 将插值结果填回网格（只更新原本为0的位置）
                flat_data = data.ravel()
                mask_indices = np.where(interpolate_mask)[0]
                zero_mask = flat_data[mask_indices] == 0
                flat_data[mask_indices[zero_mask]] = interpolated[zero_mask]
                data = flat_data.reshape(height, width)

                log_print(
                    f"Interpolated {np.sum(zero_mask)} grid cells using {method}",
                    "debug",
                )

        return data

    def _meshgrid_rain_simple(
        self,
        start_time: str,
        end_time: str,
        station_ids: list[str] | None = None,
        min_rainfall: float = 0.0,
        grid_width: int = 1200,
        grid_height: int = 900,
    ) -> dict:
        """简单的网格方法（fallback，当geo_utils不可用时）"""
        import numpy as np

        # 四川省的经纬度范围
        lon_min, lon_max = 97.0, 109.0
        lat_min, lat_max = 26.0, 35.0

        # 查询数据
        rainfall_data = self.query_by_time_range(
            start_time=start_time,
            end_time=end_time,
            station_ids=station_ids,
            min_rainfall=min_rainfall,
        )

        if len(rainfall_data) == 0:
            mapped_data = np.zeros((grid_height, grid_width))
        else:
            # 简单的网格映射
            station_rainfall = (
                rainfall_data.groupby(["station_id", "lng", "lat"]).agg({"rainfall": "mean"}).reset_index()
            )

            mapped_data = np.zeros((grid_height, grid_width))
            grid_counts = np.zeros((grid_height, grid_width))

            for _, row in station_rainfall.iterrows():
                lng, lat, rainfall = row["lng"], row["lat"], row["rainfall"]

                if not (lon_min <= lng <= lon_max and lat_min <= lat <= lat_max):
                    continue

                grid_x = int((lng - lon_min) / (lon_max - lon_min) * grid_width)
                grid_y = int((lat_max - lat) / (lat_max - lat_min) * grid_height)

                if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                    mapped_data[grid_y, grid_x] += rainfall
                    grid_counts[grid_y, grid_x] += 1

            mask = grid_counts > 0
            mapped_data[mask] = mapped_data[mask] / grid_counts[mask]

        # 创建简单的坐标网格
        x_grid = np.linspace(lon_min, lon_max, grid_width)
        y_grid = np.linspace(lat_min, lat_max, grid_height)
        x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

        return {
            "mapped_data": mapped_data.astype(np.float32),
            "x_mesh": x_mesh,
            "y_mesh": y_mesh,
            "lon_mesh": x_mesh,
            "lat_mesh": y_mesh,
            "bounds_target": (lon_min, lon_max, lat_min, lat_max),
            "bounds_lonlat": (lon_min, lon_max, lat_min, lat_max),
            "source_proj": "epsg:4326",
            "target_proj": "epsg:4326",
            "statistics": {"non_zero_grids": int(np.sum(mapped_data > 0))},
        }


def process_shard(data_dir: str, shard_db_dir="data2/rainfall_shards", log_dir="data2/logs"):
    # 为每个进程创建独立的日志文件
    dir_name = data_dir.split("/")[-1]
    sink = f"{log_dir}/rainfall_importer_{dir_name}_{int(time.time())}.log"

    # 确保日志目录存在
    Path(log_dir).mkdir(exist_ok=True, parents=True)
    print(f"Process for {data_dir}: log file will be saved at {sink}")

    # 配置独立的logger
    logger.add(
        sink,
        format="{time:HH:mm:ss} - <level>[{level}:{file.name}:{line}]</level> - <level>{message}</level>",
        enqueue=True,
        backtrace=True,
        colorize=False,
    )

    try:
        # 为每个进程创建独立的分片数据库目录
        shard_db_dir = f"{shard_db_dir}_{dir_name}"
        importer = ShardedRainDataImporter(shard_db_dir)

        # 导入雨量站基础信息
        logger.info("Importing station information...")
        importer.import_station_info()

        # 批量导入指定目录下的Excel文件
        if os.path.exists(data_dir):
            logger.info(f"Importing Excel files from {data_dir}...")
            sheets_to_import = [f"{i}小时" for i in range(1, 25)]
            importer.import_directory(data_dir, "*.xlsx", sheets_to_import, include_zeros=False)

            # 获取导入统计
            stats = importer.get_statistics()
            logger.info(f"Import completed for {data_dir}:")
            logger.info(f"  Total records: {stats['total_records']:,}")
            logger.info(f"  Total shards: {stats['shard_count']}")
            logger.info(f"  Total size: {stats['total_size_mb']:.2f} MB")

            return {
                "status": "success",
                "data_dir": data_dir,
                "shard_db_dir": shard_db_dir,
                "records": stats["total_records"],
                "shards": stats["shard_count"],
                "size_mb": stats["total_size_mb"],
            }

        else:
            logger.error(f"Data directory not found: {data_dir}")
            return {
                "status": "error",
                "data_dir": data_dir,
                "error": f"Directory not found: {data_dir}",
            }

    except Exception as e:
        logger.error(f"Failed to process {data_dir}: {e}")
        return {"status": "error", "data_dir": data_dir, "error": str(e)}


def multiprocessing_import(rain_data_dirs: list[str] = ["data/rain/202310", "data/rain/202311"]):
    """多进程导入数据 - 修复版本"""

    # 验证目录存在性
    valid_dirs = []
    for data_dir in rain_data_dirs:
        if os.path.exists(data_dir):
            valid_dirs.append(data_dir)
            print(f"✅ Found directory: {data_dir}")
        else:
            print(f"❌ Directory not found: {data_dir}")

    if not valid_dirs:
        print("❌ No valid directories found for processing")
        return

    print(f"🚀 Starting multiprocessing import for {len(valid_dirs)} directories...")
    start_time = time.time()

    # 使用进程数等于目录数（但不超过CPU核心数）
    max_processes = min(len(valid_dirs), multiprocessing.cpu_count())

    with multiprocessing.Pool(processes=max_processes) as pool:
        try:
            # 异步执行并等待结果
            result_async = pool.map_async(process_shard, valid_dirs)
            results = result_async.get(timeout=3600)  # 1小时超时

            # 分析结果
            successful_imports = []
            failed_imports = []

            total_records = 0
            total_shards = 0
            total_size_mb = 0.0

            for result in results:
                if result["status"] == "success":
                    successful_imports.append(result)
                    total_records += result["records"]
                    total_shards += result["shards"]
                    total_size_mb += result["size_mb"]
                else:
                    failed_imports.append(result)

            # 打印汇总结果
            elapsed_time = time.time() - start_time
            print(f"\n{'=' * 60}")
            print(f"🎯 多进程导入完成!")
            print(f"{'=' * 60}")
            print(f"⏱️  总耗时: {elapsed_time:.2f} 秒")
            print(f"✅ 成功: {len(successful_imports)} 个目录")
            print(f"❌ 失败: {len(failed_imports)} 个目录")
            print(f"📊 总记录数: {total_records:,}")
            print(f"📦 总分片数: {total_shards}")
            print(f"💾 总大小: {total_size_mb:.2f} MB")

            if successful_imports:
                print(f"\n✅ 成功导入的目录:")
                for result in successful_imports:
                    print(f"  {result['data_dir']}: {result['records']:,} 条记录")

            if failed_imports:
                print(f"\n❌ 失败的目录:")
                for result in failed_imports:
                    print(f"  {result['data_dir']}: {result['error']}")

            log_print(
                f"Multiprocessing import completed: {len(successful_imports)} success, {len(failed_imports)} failed",
                "info",
            )

        except multiprocessing.TimeoutError:
            log_print("Multiprocessing import timed out after 1 hour", "error")
        except Exception as e:
            log_print(f"Multiprocessing import failed with error: {e}", "error")


def merge_shard_databases(db_save_dir: str = "data2/rainfall_shards", base_dir: str | Path = "data2"):
    """合并多个进程创建的分片数据库到主数据库（按月分片级别真正合并数据）"""
    import json

    main_importer = ShardedRainDataImporter(db_save_dir)

    # 查找所有进程创建的分片数据库
    base_dir = Path(base_dir)
    shard_dirs = list(base_dir.glob("rainfall_shards_*"))

    if not shard_dirs:
        print("❌ No shard databases found to merge")
        return

    print(f"🔄 Merging {len(shard_dirs)} shard databases...")

    total_shard_files = 0

    for shard_dir in shard_dirs:
        try:
            print(f"📂 Processing {shard_dir.name}...")

            # 1. 合并 shard_index.json
            json_file = shard_dir / "shard_index.json"
            if json_file.exists():
                source_json = json.load(json_file.open("r", encoding="utf-8"))
                target_json_path = main_importer.base_dir / "shard_index.json"
                if target_json_path.exists():
                    with open(target_json_path, "r", encoding="utf-8") as f:
                        existing_index = json.load(f)
                else:
                    existing_index = {}

                # 简单按 shard_id 覆盖/合并，统计值稍后统一用 _update_shard_stats 刷新
                existing_index.update(source_json)
                with open(target_json_path, "w", encoding="utf-8") as f:
                    json.dump(existing_index, f, ensure_ascii=False, indent=4)
                print(f"  ✅ Merged shard_index.json from {shard_dir.name} into main index")

            # 2. 合并每个 shard_*.duckdb 中的 rainfall_data 到主目录对应分片
            db_files = list(shard_dir.glob("shard_*.duckdb"))

            for db_file in db_files:
                if db_file.name == "metadata.duckdb":
                    continue

                shard_id = db_file.stem  # e.g. shard_202507
                total_shard_files += 1

                # 从子分片读取数据
                try:
                    src_conn = duckdb.connect(str(db_file))
                    src_df = src_conn.execute("SELECT * FROM rainfall_data").df()
                    src_conn.close()
                except Exception as e:
                    print(f"  ❌ Failed to read {db_file.name}: {e}")
                    continue

                if src_df.empty:
                    print(f"  ⚠️  {db_file.name} is empty, skipping")
                    continue

                # 目标分片路径
                target_file = main_importer.get_shard_path(shard_id)

                # 若目标分片不存在，则根据 shard_id 构造该月第一天创建分片
                if not target_file.exists():
                    try:
                        # shard_YYYYMM -> YYYY-MM-01
                        yyyymm = shard_id.split("_")[-1]
                        year = int(yyyymm[:4])
                        month = int(yyyymm[4:6])
                        date_str = f"{year:04d}-{month:02d}-01"
                        main_importer.create_shard(shard_id, date_str)
                        print(f"  ✅ Created target shard {shard_id} at {target_file}")
                    except Exception as e:
                        print(f"  ❌ Failed to create shard {shard_id}: {e}")
                        continue

                # 将子分片的数据合并到目标分片，依赖主键 (station_id, timestamp) 做去重/覆盖
                try:
                    dst_conn = duckdb.connect(str(target_file))
                    dst_conn.register("src_df", src_df)
                    dst_conn.execute(
                        """
                        INSERT OR REPLACE INTO rainfall_data
                        (station_id, timestamp, datetime_str, rainfall, status,
                         hour, minute, date, source_file, source_sheet)
                        SELECT station_id, timestamp, datetime_str, rainfall, status,
                               hour, minute, date, source_file, source_sheet
                        FROM src_df
                        """
                    )
                    dst_conn.close()
                    print(f"  ✅ Merged {len(src_df):,} rows from {db_file.name} into {target_file.name}")
                except Exception as e:
                    print(f"  ❌ Failed to merge {db_file.name} into main shard: {e}")
                    continue

        except Exception as e:
            print(f"❌ Error processing {shard_dir}: {e}")

    # 重新加载并刷新所有分片统计信息
    print("🔄 Rebuilding shard index and updating stats...")
    main_importer.load_shard_index()

    for shard_file in main_importer.base_dir.glob("shard_*.duckdb"):
        shard_id = shard_file.stem
        main_importer._update_shard_stats(shard_id)

    # 获取最终统计
    final_stats = main_importer.get_statistics()

    print(f"\n🎯 合并完成!")
    print(f"📂 处理分片文件数: {total_shard_files}")
    print(f"📊 总记录数: {final_stats['total_records']:,}")
    print(f"📦 总分片数: {final_stats['shard_count']}")
    print(f"💾 总大小: {final_stats['total_size_mb']:.2f} MB")
    print(f"📅 时间范围: {final_stats['date_range'][0]} 到 {final_stats['date_range'][1]}")


def multiprocessing_import_with_merge(
    db_save_dir: str = "data2/raw_2025_addtional_testset/rainfall_shards",
    rain_data_dirs: list[str] = ["data2/raw_2025_addtional_testset/rain"],
):
    """完整的多进程导入流程（包含合并）"""
    print("🚀 开始多进程导入...")

    # 1. 多进程导入
    multiprocessing_import(rain_data_dirs)

    # 2. 合并分片数据库
    print("\n" + "=" * 60)
    merge_shard_databases(db_save_dir)

    print("\n🎉 多进程导入和合并流程完成!")


def main_test():
    """主函数 - 分片数据库示例用法"""
    # 创建分片导入器实例
    importer = ShardedRainDataImporter("data/rainfall_shards")

    # 导入雨量站基础信息
    print("Importing station information...")
    importer.import_station_info()

    # 批量导入data目录下的Excel文件
    data_directory = "data/rain/202306"
    if os.path.exists(data_directory):
        print(f"Importing Excel files from {data_directory}...")
        sheets_to_import = [f"{i}小时" for i in range(1, 25)]
        importer.import_directory(data_directory, "*.xlsx", sheets_to_import, include_zeros=False)
    else:
        print(f"Data directory not found: {data_directory}")

        # 导入单个文件示例
        sample_file = "data/rain/分钟级雨量站数据_20230603.xlsx"
        if os.path.exists(sample_file):
            print(f"Importing sample file: {sample_file}")
            importer.import_excel_file(sample_file, ["1小时", "12小时"], include_zeros=False)

    # 显示统计信息
    print("\n=== Sharded Database Statistics ===")
    stats = importer.get_statistics()
    print(f"Total stations: {stats['station_count']:,}")
    print(f"Total shards: {stats['shard_count']}")
    print(f"Total records: {stats['total_records']:,}")
    print(f"Total size: {stats['total_size_mb']:.2f} MB")
    print(f"Date range: {stats['date_range'][0]} to {stats['date_range'][1]}")

    print(f"\nShard details:")
    for shard_detail in stats["shard_details"][:10]:  # 显示前10个分片
        print(
            f"  {shard_detail['shard_id']}: {shard_detail['record_count']:,} records, "
            f"{shard_detail['file_size_mb']} MB ({shard_detail['date_range']})"
        )

    # 查询示例
    print("\n=== Query Examples ===")
    try:
        import time

        # 测试查询所有数据（包括0值）
        print("\n1. 完整数据查询（不包括0值）:")
        t1 = time.time()
        all_data = importer.query_by_time_range("2023-06-03 00:00:00", "2023-06-03 23:59:59", min_rainfall=0.0)
        t2 = time.time()
        print(f"  查询耗时: {t2 - t1:.3f} 秒")
        print(f"  返回记录: {len(all_data)} 条")

        if len(all_data) > 0:
            print("  数据预览:")
            preview_cols = ["station_id", "datetime_str", "rainfall"]
            if "lng" in all_data.columns:
                preview_cols.extend(["lng", "lat"])
            print(all_data[preview_cols].head())

            # 统计降雨数据
            zero_count = len(all_data[all_data["rainfall"] == 0.0])
            rain_count = len(all_data[all_data["rainfall"] > 0.0])
            print(f"  零值记录: {zero_count} 条")
            print(f"  降雨记录: {rain_count} 条")
            print(f"  降雨量范围: {all_data['rainfall'].min():.2f} - {all_data['rainfall'].max():.2f} mm")

        # 测试仅查询有降雨的数据
        print("\n2. 仅降雨数据查询（min_rainfall > 0）:")
        t1 = time.time()
        rainfall_only_data = importer.query_by_time_range(
            "2023-06-03 00:00:00", "2023-06-03 23:59:59", min_rainfall=0.1
        )
        t2 = time.time()
        print(f"  查询耗时: {t2 - t1:.3f} 秒")
        print(f"  返回记录: {len(rainfall_only_data)} 条（仅降雨数据）")

        if len(rainfall_only_data) > 0:
            print("  降雨数据预览:")
            print(rainfall_only_data[["station_id", "datetime_str", "rainfall"]].head())

        # 测试特定站点查询
        print("\n3. 特定站点查询:")
        if len(all_data) > 0:
            t1 = time.time()
            station_list = all_data["station_id"].unique()[:3].tolist()
            station_data = importer.query_by_time_range(
                "2023-06-03 10:00:00",
                "2023-06-03 11:59:59",
                station_ids=station_list,
                min_rainfall=0.0,
            )
            t2 = time.time()
            print(f"  查询耗时: {t2 - t1:.3f} 秒")
            print(f"  查询站点: {station_list}")
            print(f"  返回记录: {len(station_data)} 条")

            if len(station_data) > 0:
                print("  站点数据预览:")
                print(station_data[["station_id", "datetime_str", "rainfall"]].head(10))

    except Exception as e:
        print(f"Query examples failed: {e}")

    # 优化分片
    print("\n=== Optimizing Shards ===")
    importer.optimize_shards()

    # 清理空分片
    print("=== Cleaning Up Empty Shards ===")
    importer.cleanup_empty_shards()


def compare_performance():
    """对比单文件和分片方案的性能"""
    import time

    # from src.tools.rain_station_excels_to_db import RainDataImporterDuckDB

    # 单文件方案
    # single_importer = RainDataImporterDuckDB("data/rainfall_database.duckdb")

    # 分片方案
    shard_importer = ShardedRainDataImporter("data/rainfall_shards")

    # 测试相同的查询
    test_cases = [
        ("2023-01-26 00:00:00", "2023-01-26 01:00:00"),  # 1小时
        ("2023-01-10 00:00:00", "2023-01-10 12:00:00"),  # 12小时
        ("2023-01-30 00:00:00", "2023-01-30 23:59:59"),  # 全天
    ]

    for start_time, end_time in test_cases:
        print(f"\n测试范围: {start_time} to {end_time}")

        # 单文件查询
        # t1 = time.time()
        # result1 = single_importer.query_by_time_range_fast(start_time, end_time, min_rainfall=0.1)
        # single_time = time.time() - t1

        # 分片查询
        t2 = time.time()
        result2 = shard_importer.query_by_time_range(start_time, end_time, min_rainfall=0.0)
        shard_time = time.time() - t2

        print(f"查询结果:")
        print(f"  分片  : {len(result2)} 条记录")
        print(f"  分片查询耗时: {shard_time:.3f}秒")

        # print(f"  单文件: {single_time:.3f}秒, {len(result1):,} 条记录")
        # print(f"  分片  : {shard_time:.3f}秒, {len(result2):,} 条记录")
        # print(f"  性能比: {single_time/shard_time:.2f}x")


def query_db():
    # 创建分片导入器实例
    importer = ShardedRainDataImporter("data2/rainfall_shards")

    # 导入雨量站基础信息
    print("Importing station information...")
    importer.import_station_info()

    import time

    for i in range(10):
        start_time = time.perf_counter()
        # res = importer.query_by_time_range(
        #     "2023-05-02 00:00:00",
        #     "2023-05-02 02:00:00",  # station_ids=[""]
        # )
        res = importer.meshgrid_rain(
            "2025-09-27 21:30:00",
            "2025-09-27 22:00:00",
            grid_width=256,
            grid_height=256,
            bounds=(97.0, 109.0, 26.0, 35.0),
        )
        end_time = time.perf_counter()
        if i == 0:
            print(res)
        print(f"Query {i} took {end_time - start_time:.4f} seconds")


if __name__ == "__main__":
    # main_test()

    process_shard(
        data_dir="data2/raw_2025_addtional_testset/rain",
        shard_db_dir="data2/raw_2025_addtional_testset/rain_shards",
        log_dir="data2/raw_2025_addtional_testset/logs",
    )
    # multiprocessing_import_with_merge()

    # compare_performance()
    #
    # query_db()

    # from pathlib import Path

    # import duckdb

    # shard = Path("data2/rainfall_shards") / "shard_202509.duckdb"
    # con = duckdb.connect(str(shard))

    # # 看整个 2025-09 的时间范围
    # min_dt, max_dt = con.execute("""
    #     SELECT MIN(datetime_str), MAX(datetime_str) FROM rainfall_data
    # """).fetchone()
    # print("time range in shard_202509:", min_dt, "->", max_dt)

    # # 看每天各有多少条，方便你对照原 Excel
    # day_counts = con.execute("""
    #     SELECT date, COUNT(*) AS n, MIN(rainfall) AS min_r, MAX(rainfall) AS max_r
    #     FROM rainfall_data
    #     GROUP BY date
    #     ORDER BY date
    # """).fetchdf()
    # print(day_counts.head(20))
