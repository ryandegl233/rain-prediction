# 2026-04-16 Wan Causal Conv And Encoder Fix

- 记录时间: 2026-04-16 01:03:17 CST

## 改动摘要

1. 将 3D 卷积路径接入可选风格，新增 `wan_factorized`（空间 `Conv2d` + 时间因果卷积）以降低时空建模开销。
2. 保留并贯通 `causal` 行为，确保时间维仍是因果建模。
3. 修复 `encoder_type="resnet"` 且 `activation_checkpoint=False` 时 `resnet_encoder` 未初始化的问题。
4. 模型 YAML 增加 `encoder_conv_style: wan_factorized` 配置项。

## 具体修改文件

- `src/networks/modules/reconstruction.py`
  - 新增 `WanFactorizedCausalConv3d`
  - 新增 `build_3d_conv_layer(..., conv_style=...)`
  - `ResNetBlock3D`、`UpsampleConv3d`、`ResNetDecoder3D` 等 3D 路径接入 `conv_style`
- `src/networks/time_series/causal_patch_transformer_next_frame.py`
  - 新增模型参数 `encoder_conv_style`
  - resnet encoder 路径接入 `conv_style`
  - 3D decoder 路径透传 `conv_style`
  - 修复 `resnet_encoder` 初始化分支
- `src/config/ts_rain_train/rain_prediction_model/causal_patch_transformer_next_frame.yaml`
  - 增加 `encoder_conv_style: wan_factorized`

## 验证

- 执行命令:
  - `pytest -q src/tests/time_series/test_causal_patch_transformer_next_frame.py`
- 结果:
  - `12 passed`
