# 确定性降水演变诊断与微调

目标是让 t+1～t+4 的实际雨量场更接近 GT，分别关注位置、边界、强雨核心以及增强/消散。没有修改原 baseline、已有训练器或 diffusion；新增独立训练器、配置、启动脚本和诊断脚本。

## 完成的诊断

`src/residual_diffusion_stage/diagnose_coarse_evolution.py` 严格加载 baseline 和 smoke 各自的模型配置与权重，复用此前固定的 60 个验证窗口，并逐样本验证输入 SHA256。未来雷达、卫星和雨量均不作为 rollout 输入。对照包含直接重复最后一帧。

结果保存在 `runs/coarse_evolution_diagnosis_20261005_v2/evolution.json`。这些窗口有既有降雨面积筛选，不能代表完整测试集。首次配置加载失败的目录保留，不是有效结果。

| smoke 指标 | t+1 | t+2 | t+3 | t+4 |
|---|---:|---:|---:|---:|
| 活跃区域预测变化幅度 / GT 变化幅度 | 0.417 | 0.281 | 0.156 | 0.119 |
| GT ≥0.5 mm/min 区域预测雨量总和 / GT 总和 | 0.694 | 0.503 | 0.389 | 0.369 |

全场 MAE：persistence 0.015463、baseline 0.016571、smoke 0.016419 mm/min。高雨强区域比值同时受位移误差和强度误差影响，不是纯粹的峰值衰减测量。变化幅度增大本身不等于改善，必须同时检查变化误差、变化方向、空间重叠与雨量误差。

## 训练方法

新增 `src/trainer/rain_trainer_coarse_evolution.py`，继承现有训练器，采用 teacher-forced 与四步 free-rollout 联合训练。两条路径依次 forward/backward，在同一次 optimizer step 前累加梯度；保持同样的目标并减少同时存活的计算图。仅支持单 GPU、无 GAN。没有采用 LoRA，也没有新增运动网络。

`L = L_teacher + lambda(s) L_rollout`，lambda 在前 500 个优化步骤由 0.2 增至 0.4。每一步自回归使用自己的历史预测，未来真实模态仅用于构造监督目标；历史回填使用 detach，属于截断梯度的 rollout 训练。

新增配置 `src/config/ts_rain_train/rain_trainer_coarse_evolution.yaml`：

- 加权雨量 MSE：`w(y)=1+2*max(y-0.05,0)^0.7`，作用于物理雨量空间（当前未零中心化），增强真实强雨像素的误差贡献；并保留全场监督，不能只增加雨量来降低损失。
- 空间梯度 Charbonnier 误差，权重 0.05，监督预测与 GT 的水平/垂直边缘差异。
- 相邻帧雨量变化 MSE：比较 `(prediction[t]-prediction[t-1])` 和 `(GT[t]-GT[t-1])`，t+1 共同锚定最后观测。权重在 500 步内由 0.1 升至 0.35，保留有上限的 radar-guided 权重。
- 关闭 rollout 中额外的 delta-head MSE，避免与场变化监督叠加；保留 teacher-forced 原有 residual-delta 辅助监督和雷达/卫星重构监督。
- 关闭该实验的 rain-event BCE。没有为提高某个阈值分数额外扩张或裁剪雨区。
- 关闭时间倒放增强，保留其他训练增强。
- 学习率从 1e-5 下降至 3e-6。原调度器的 `min_lr` 实际是 LambdaLR 倍率，因此这里设为 0.3。

从 smoke 的 `checkpoint_4` 开始新的 1000 步实验，优化器重新初始化；batch=1、梯度累积=8、完整 448 分辨率、4 帧历史预测 4 帧。每 250 步验证并保存，保留 16 份 checkpoint。这是受控比较阶段，不表示训练充分或已修复全部演变问题。

固定验证在 step=0 和每 250 步执行，保存 CSI/POD/FAR、MAE/RMSE、强雨区域误差、FSS 和演变统计，以及相同前三个固定样本的雨量图和变化图。输出使用原始 checkpoint 参数，不使用 EMA。看图统一色标，不进行区域擦除或视觉修饰。验证窗口用于开发，最终仍需更完整的独立测试。

## 运行

```bash
cd /home/rainpred/RainPrediction
bash src/residual_diffusion_stage/train_coarse_evolution_4090.sh
```

脚本通过 GPU UUID 绑定 RTX 4090。正式运行前已用完整四步 rollout、8 次梯度累积完成一次真实参数更新；峰值 allocated 11.93 GiB、reserved 12.20 GiB。脚本要求至少 13.5 GiB 可用显存；其他进程的显存变化仍可能影响训练。

新增测试验证拆分反向传播与合并反向传播的梯度累积及参数更新等价；验证演变诊断可区分完美预测、移动雨区与 persistence，并对无雨事件的未定义指标输出 null。
