# 2026-04-16 Rain Interp Visualization And Ratio Filter

## 1) Rain 输出与插值导出能力增强

- 文件: `src/examination/output_nc.py`
- 新增/完善内容:
  - 增加公开可复用的雨量可视化函数: `rain_frame_to_image`、`draw_rain_spatial_before_after`、`draw_rain_temporal_before_after`。
  - 将时间插值规则补齐为 `exact_nx` 逻辑，支持“真实帧对齐 + 补尾帧”。
  - `tensor_output_to_xarray` 增加 `time_bound` 支持，按插帧规则生成均匀 `time` 坐标并写入 `nc`。
  - 完善 docstring 与示例用法，便于在 examination / trainer 侧复用。

## 2) 可视化脚本复用公共 API

- 文件: `src/examination/visualize_rain_interp_from_cfg.py`
- 改动内容:
  - 删除本地重复绘图实现，改为直接调用 `output_nc.py` 公共函数。
  - 导出 `nc` 时传入 `time_bound`，使插帧后时间轴在文件内可直接查看。

## 3) 测试补充

- 文件: `src/tests/examination/test_output_nc.py`
  - 增加 `exact_nx` 与 `frames` 下 `time_bound` 时间坐标测试。
- 文件: `src/tests/examination/test_cfg_train_val_iterable_samples.py`
  - 新增基于当前 cfg 的 train/val 样本统计测试（总样本与 drop_last 后可迭代样本）。
  - 输出统计报告到 `runs/examination/`。

## 4) 数据筛选与比例列扩展

- 文件: `src/dataset/rain_ts_litdata.py`
- 改动内容:
  - 调整 `_export_metadata_with_rain_ratio`：在覆盖写入时保留已有 `rain_ratio_gt_*` 列，再追加新阈值列，避免只保留单列。
  - 使用 `export_rain_filter_ratio` 批量为各月份 `metadata_rain_ratio.parquet` 增加 `rain_ratio_gt_0p02`。

## 5) 训练筛选配置调整

- 文件: `src/config/ts_rain_train/rain_trainer_ts_next_frame.yaml`
- 改动内容:
  - `rain_ratio_filter.column` 从 `rain_ratio_gt_0p1` 调整为 `rain_ratio_gt_0p02`。
  - `rain_ratio_filter.min_value` 调整为 `0.02`。
  - 保持 `mode: future_any`。

## 6) 结果验证摘要

- 关键验证:
  - `pytest -q src/tests/examination/test_output_nc.py` 通过。
  - `pytest -q src/tests/examination/test_cfg_train_val_iterable_samples.py -s` 通过。
- 最新统计（`rain_ratio_gt_0p02 + min_value=0.02`）:
  - train: `5222`（iterable `5222`）
  - val: `1258`（iterable `1256`）
