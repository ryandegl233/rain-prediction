# 2026-04-16 Dataset Aug + Time Embedding + Rollout Time

- 记录时间: 2026-04-16 01:42:23 CST

## 改动摘要

1. 新增独立数据增强模块，支持时序样本随机裁切（返回绝对/归一化裁切坐标）与时间反转增强。
2. 将数据增强接入 `RainTimeSeriesDataset` 与 dataloader 接口，并在 batch 中返回增强元信息。
3. 模型新增真实时间 embedding 输入通路（不使用 `time_reverse_flag`）。
4. trainer 的 train/val teacher-forcing 路径接入 `time_past/time_future`，透传到 `model.forward_ar`。
5. val rollout 路径完整接入时间透传：包括自回归 seed 构造、chunk rollout、after-roll-next 分支。
6. val 可视化支持 `viz_sample_index < 0` 时随机抽样，YAML 默认改为 `-1`。

## 具体修改文件

- `src/dataset/rain_ts_augmentation.py`
  - 新增 `RainTimeSeriesAugmentor`
  - 增强输出: `aug_crop_box_xyxy` / `aug_crop_box_norm_xyxy` / `aug_time_reversed`
- `src/dataset/rain_ts_litdata.py`
  - dataset/dataloader 新增 augmentation 参数
  - `__getitem__` 接入 augmentor 并返回增强元信息
- `src/config/ts_rain_train/rain_trainer_ts_next_frame.yaml`
  - 新增 `dataset.augmentation` 配置并透传到 train/val dataloader
  - `val.viz_sample_index` 改为 `-1`
- `src/networks/time_series/causal_patch_transformer_next_frame.py`
  - 新增 `use_time_embedding`
  - `forward/forward_ar` 新增时间输入参数
  - `_encode_tokens` 接入时间特征投影（sin/cos）
- `src/config/ts_rain_train/rain_prediction_model/causal_patch_transformer_next_frame.yaml`
  - 增加 `use_time_embedding: true`
- `src/trainer/rain_trainer_ts_next_frame.py`
  - `_build_next_pred_batch` 构造并透传 `context_time/target_seed_time`
  - train/val step 前向调用支持时间输入
  - `_prepare_val_inference_batch` 返回 `context_time/target_time`
  - rollout 全链路接入时间透传（含 `_build_self_rolled_seed`、`_rollout_predict_with_settings`、`_predict_next_after_roll_block`、`_val_inference_step`）
- `src/tests/dataset/test_rain_ts_augmentation.py`
  - 新增 augmentation 单测
- `src/tests/time_series/test_causal_patch_transformer_next_frame.py`
  - 新增 time embedding 前向单测
- `src/tests/trainer/test_rain_trainer_ts_next_frame.py`
  - 新增时间组装与 rollout 时间透传单测

## 验证

- `pytest -q src/tests/dataset/test_rain_ts_augmentation.py src/tests/dataset/test_rain_ts_litdata_window_index.py src/tests/trainer/test_rain_trainer_ts_next_frame.py`
  - 结果: `16 passed`
- `pytest -q src/tests/time_series/test_causal_patch_transformer_next_frame.py src/tests/trainer/test_rain_trainer_ts_next_frame.py`
  - 结果: `23 passed`
- `pytest -q src/tests/trainer/test_rain_trainer_ts_next_frame.py`
  - 结果: `11 passed`
- `pytest -q src/tests/time_series/test_causal_patch_transformer_next_frame.py`
  - 结果: `13 passed`

## 备注

- 当前实现按你的要求未引入 `time_reverse_flag` 到模型输入。
- rollout 时间已透传到 model，模型是否消费由 `use_time_embedding` 控制。
