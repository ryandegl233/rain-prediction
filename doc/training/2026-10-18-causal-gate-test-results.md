# 2026-10-18 因果训练配置验证记录

本轮新增独立配置 `src/config/ts_rain_train/gated_modality_rain_causal.yaml`，复用已有 2D decoder 与 I/O trainer，未修改原模型、trainer、baseline 或同学源码。测试文件为 `src/tests/gated_modality_rain/test_causal_training.py`。

## 定向检查

环境：Windows，Python 3.13.5，PyTorch 2.12.0+cpu。测试脚本另外完成 Python 3.12 语法解析；Ruff 检查通过。没有执行 ty。

```text
python -m pytest src/tests/gated_modality_rain/test_causal_training.py src/tests/gated_modality_rain/test_io_trainer.py -q -p no:cacheprovider --tb=short
16 passed, 2 skipped, 21 warnings in 140.21s
```

其中新增配置 8 项通过、1 项 CUDA BF16 检查跳过；旧 I/O 检查 8 项通过、1 项 CUDA 传输检查跳过。21 条警告来自依赖中的 torch.jit 弃用提示。两个跳过均不能计为 GPU 通过，也没有在本地验证真实数据或 448×448 的 batch2 显存。

覆盖内容：

- 改动后面的 seed 帧，前面的完整三模态预测保持不变。
- 单帧单独解码与多帧分别解码后合并一致；该检查用 FP64 和 1e-10 容差，避免不同批尺寸的 FP32 卷积舍入差异掩盖数学依赖关系。
- 主分支 context=4、seed=1，与实际自由递推第一步的输入和预测一致。
- 原 rollout loss 的梯度到达雷达、卫星、降雨 decoder 以及空间门控。
- 改变未来观测标签不会改变固定起报预测。
- 验证窗口序列跨轮保持一致。
- 实际小模型、activation checkpoint 开启，完成两次优化更新、一次验证、两次 checkpoint 保存；三种 decoder 和门控参数都有更新且有限。
- 现有入口通过 --config-name 正确解析新配置；旧 I/O prepared loader、损失/梯度/更新等价和 367 个同学快照文件完整性仍通过。

修复前先使用旧 `gated_modality_rain_io` 配置执行 5 个关键行为测试，5 个均按预期失败：未来影响前缀、单帧与多帧解码不同、context 只有 1 帧而非 4 帧、递推 decoder 零梯度、验证顺序随轮次变化。之后切换到独立修复配置，保留这些行为断言。小模型的 decoder_base_channels 使用 32 以满足已有 2D 残差块的 GroupNorm 分组要求；正式配置使用 128。

## 全仓检查限制

另执行了项目根目录下的 `python -m pytest -q -p no:cacheprovider --tb=short`。它在收集阶段中止，59 项错误，不能宣称全仓测试通过。

原因包括本地缺少 pyproj、duckdb、diffusers、dion，既有工具模块缺失，以及 third_party 同学副本与主仓测试同名导致 import file mismatch。本次没有安装这些无关依赖、清理缓存、删除文件或改写旧测试。完整输出保存在执行机 `C:/Users/13726/.codex/tmp/rain-causal-full-suite-20261018.log`。

收集错误逐项如下：

- `src/tests/test_comprehensive_features.py`
- `src/tests/test_duckdb.py`
- `src/tests/test_fm_rollout_high_order_solver.py`
- `src/tests/test_fm_rollout_kv_cache.py`
- `src/tests/test_minute_simple.py`
- `src/tests/test_minute_window.py`
- `src/tests/test_only_rain_db_save.py`
- `src/tests/test_rain_interfaces.py`
- `src/tests/test_time_windows_complete.py`
- `src/tests/tools/test_plot_ragion.py`
- `src/tests/trainer/test_rain_trainer_ts_diffusion_val_metrics.py`
- `src/utils/optim/dion/tests/test_newton_shultz.py`
- `test_plot_all_data.py`
- `third_party/dongchen_rain_20261013/src/tests/causal_forcing/test_causal_forcing_fm_alignment.py`
- `third_party/dongchen_rain_20261013/src/tests/dataset/test_rain_ts_augmentation.py`
- `third_party/dongchen_rain_20261013/src/tests/dataset/test_rain_ts_litdata_bytes.py`
- `third_party/dongchen_rain_20261013/src/tests/dataset/test_rain_ts_litdata_rain_ratio_filter_plot.py`
- `third_party/dongchen_rain_20261013/src/tests/dataset/test_rain_ts_litdata_sequence_plot.py`
- `third_party/dongchen_rain_20261013/src/tests/dataset/test_rain_ts_litdata_swinnet_bad_value.py`
- `third_party/dongchen_rain_20261013/src/tests/dataset/test_rain_ts_litdata_window_index.py`
- `third_party/dongchen_rain_20261013/src/tests/examination/test_cfg_train_val_iterable_samples.py`
- `third_party/dongchen_rain_20261013/src/tests/examination/test_output_nc.py`
- `third_party/dongchen_rain_20261013/src/tests/examination/test_rain_100_frames_gif.py`
- `third_party/dongchen_rain_20261013/src/tests/gated_modality_rain/test_framework_isolation.py`
- `third_party/dongchen_rain_20261013/src/tests/modules/test_reconstruction_causal.py`
- `third_party/dongchen_rain_20261013/src/tests/networks/test_gan_losses.py`
- `third_party/dongchen_rain_20261013/src/tests/networks/test_multimodal_spatial_frontend.py`
- `third_party/dongchen_rain_20261013/src/tests/networks/test_patchgan3d.py`
- `third_party/dongchen_rain_20261013/src/tests/residual_diffusion_stage/test_ablate_residual_scale.py`
- `third_party/dongchen_rain_20261013/src/tests/residual_diffusion_stage/test_metrics.py`
- `third_party/dongchen_rain_20261013/src/tests/residual_diffusion_stage/test_models.py`
- `third_party/dongchen_rain_20261013/src/tests/residual_diffusion_stage/test_robust_occurrence_corrector.py`
- `third_party/dongchen_rain_20261013/src/tests/test_5_hours.py`
- `third_party/dongchen_rain_20261013/src/tests/test_comprehensive_features.py`
- `third_party/dongchen_rain_20261013/src/tests/test_duckdb.py`
- `third_party/dongchen_rain_20261013/src/tests/test_fm_rollout_high_order_solver.py`
- `third_party/dongchen_rain_20261013/src/tests/test_fm_rollout_kv_cache.py`
- `third_party/dongchen_rain_20261013/src/tests/test_minute_simple.py`
- `third_party/dongchen_rain_20261013/src/tests/test_minute_window.py`
- `third_party/dongchen_rain_20261013/src/tests/test_only_rain_db_save.py`
- `third_party/dongchen_rain_20261013/src/tests/test_rain_interfaces.py`
- `third_party/dongchen_rain_20261013/src/tests/test_rain_map_opening_erosion.py`
- `third_party/dongchen_rain_20261013/src/tests/test_swinnet_cls.py`
- `third_party/dongchen_rain_20261013/src/tests/test_time_windows_complete.py`
- `third_party/dongchen_rain_20261013/src/tests/time_series/test_causal_patch_transformer_next_frame.py`
- `third_party/dongchen_rain_20261013/src/tests/time_series/test_spatial_modality_gate.py`
- `third_party/dongchen_rain_20261013/src/tests/tools/test_check_rain_learning.py`
- `third_party/dongchen_rain_20261013/src/tests/tools/test_diagnose_rain_transitions.py`
- `third_party/dongchen_rain_20261013/src/tests/tools/test_plot_ragion.py`
- `third_party/dongchen_rain_20261013/src/tests/trainer/test_gated_modality_rain_trainer.py`
- `third_party/dongchen_rain_20261013/src/tests/trainer/test_next_frame_rain_only_rollout.py`
- `third_party/dongchen_rain_20261013/src/tests/trainer/test_next_frame_transfer.py`
- `third_party/dongchen_rain_20261013/src/tests/trainer/test_rain_trainer_ts_diffusion_val_metrics.py`
- `third_party/dongchen_rain_20261013/src/tests/trainer/test_rain_trainer_ts_next_frame.py`
- `third_party/dongchen_rain_20261013/src/tests/trainer/test_rain_trainer_ts_next_frame_stage2.py`
- `third_party/dongchen_rain_20261013/src/tests/trainer/test_spatial_frontend_trainer.py`
- `third_party/dongchen_rain_20261013/src/tests/utils/test_plot.py`
- `third_party/dongchen_rain_20261013/src/tests/utils/test_plot_region_multimodal.py`
- `third_party/dongchen_rain_20261013/src/utils/optim/dion/tests/test_newton_shultz.py`
