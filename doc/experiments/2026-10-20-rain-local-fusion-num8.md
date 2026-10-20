# num8 局部融合受控实验

入口：`src/trainer/rain_local_fusion_trainer.py`。三组从头训练，使用原 3D decoder、4→4、30 分钟数据、同样的 `rain_interpolated` 标签和 A loss。

| 组别 | YAML（位于 src/config/ts_rain_train） | 唯一结构区别 |
|---|---|---|
| A0 | rain_local_fusion_a0.yaml | 无新增支路 |
| R0 | rain_local_fusion_r0.yaml | 雨量专用 1×1 残差融合 |
| R1 | rain_local_fusion.yaml | 雨量专用 3×3 局部卷积残差融合 |

旧共享输入门控关闭，动态时效权重关闭；每个时效 q=0.25。A0 与原 A 的输出、loss、梯度和参数更新有精确等价性测试。R0/R1 的新增输出投影从零开始，初始预测与 A 一致；开始学习后本来就会产生不同轨迹。新增图中共享梯度按 FP32 数值容差核对，不承诺逐位相同；新增梯度也参与原全局裁剪，所以有支路时裁剪后的共享参数更新不保证相同。

R0/R1 不改变 decoder。分支参数只直接接收降雨 loss，主干仍被三个任务共同训练。这里的“降雨分支隔离”不是彻底消除共享梯度影响。局部 kernel=3 位于 patch=8 的 token 网格，采样中心最远 ±8 输入像素，不是已测得的物理位移校正。

## 1. 同步代码

在 num8 的 Bash 依次执行：

```bash
cd /home/yanjieyu/RainPrediction
```

```bash
git status --short --branch
```

存在本地改动时先检查保留。下面不使用自动 stash，也不覆盖改动；若拉取报错，保留报错处理，不能 reset/clean。

```bash
git -c merge.autostash=false pull --ff-only --no-rebase origin project
```

```bash
git log -1 --oneline
```

同步结果应包含交付消息中的提交。配置快照 `runs/.../config/config_total.yaml` 是历史记录，不能通过编辑它改变正在运行的实验。

## 2. tmux 和环境

```bash
nvidia-smi
```

确认 4090 没有其他训练占用后，建立会话：

```bash
tmux new -s rain-local
```

在会话内依次执行：

```bash
cd /home/yanjieyu/RainPrediction
```

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
```

```bash
conda activate rainpred
```

```bash
python -m src.trainer.rain_local_fusion_trainer --cfg job --resolve
```

默认 batch=1、累积=2、BF16、activation checkpoint；有效 batch=2。GPU 真实峰值必须由下一步测量，CPU 测试不保证 24GB 上一定够用。

## 3. 先做 R1 的 20 步 smoke

下面是一条 Bash 命令，完整复制：

```bash
python -m src.trainer.rain_local_fusion_trainer \
  train.max_steps=20 \
  train.save_every=20 \
  val.val_duration=20 \
  val.max_val_iters=1 \
  train.priority.gradient_diagnostics_every=5 \
  train.log.run_comment=num8_local_R1_smoke
```

检查正常结束、无 OOM/非有限值、出现 `[Val]`、checkpoint、CUDA peak allocated/reserved，并在对应 `validation/` 看到 JSON 和原始可视化。启动日志会打印 variant、kernel、added_parameters。训练日志会出现 `fusion/delta_rms`、`fusion/relative_rms`；初始残差为零，更新后通常不再为零。这只验证实现和资源，不代表预测性能。

如果安装了 pytest，可加跑小模型 CUDA BF16 测试：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest \
  src/tests/rain_local_fusion/test_trainer.py::test_cuda_bf16_four_step_rollout_backward \
  -q -p no:cacheprovider --import-mode=importlib
```

不需要因 litdata 的版本提示而升级现有环境。本轮保持环境一致。

## 4. 顺序运行三组 6000 步

smoke 通过后再运行。每条必须等前一条完成；同一张 GPU 上不并发，不加载 smoke 或旧 A/E 权重。

先 A0：

```bash
python -m src.trainer.rain_local_fusion_trainer \
  --config-name rain_local_fusion_a0
```

再 R0：

```bash
python -m src.trainer.rain_local_fusion_trainer \
  --config-name rain_local_fusion_r0
```

最后 R1：

```bash
python -m src.trainer.rain_local_fusion_trainer \
  --config-name rain_local_fusion
```

也可在 smoke 通过后用一条顺序队列，让任意一组失败时停止后续组：

```bash
python -m src.trainer.rain_local_fusion_trainer --config-name rain_local_fusion_a0 &&
python -m src.trainer.rain_local_fusion_trainer --config-name rain_local_fusion_r0 &&
python -m src.trainer.rain_local_fusion_trainer --config-name rain_local_fusion
```

每组默认 seed=2025，6000 个 optimizer step，每2000步全量验证并保存 checkpoint；scheduler 仍按100000步设置，与旧短实验口径一致。控制器保留可恢复状态但权重固定，不从验证集学习权重。

若统一改 batch=2，需要三组都加 `dataset.train.batch_size=2 accelerator.gradient_accumulation_plugin.num_steps=1`，并先重新 smoke；不要只改其中一组。

## 5. 查看与判定

输出：`runs/rain_local_fusion/日期/时间_组名/`。

- `config/config_total.yaml`：实际配置。
- `validation/step-00002000.json`（以及4000/6000）：各时效 rain 与 persistence 指标、事件计数。
- `validation/step-00002000-raw.png`（以及4000/6000）：同尺度原始预测图。
- `checkpoint-00002000`（以及4000/6000）：可恢复训练状态。

先比较相同步数和同一完整验证集的 +120分钟 CSI@0.3、整体 CSI@0.3；同时检查其他阈值、POD/FAR、MAE、雨区面积和 persistence。A0→R0 用于判断雨量专用支路是否有效；R0→R1 用于判断局部邻域是否额外有效。R0/R1 参数量不严格相等，日志会报告差异。

单种子6000步只用于筛选；不宣称最优 checkpoint、长期稳定收益或空间配准已被修复。验证可视化样本顺序允许变化，不能只比较随机抽到的两张图。保持验证全量，若要并排展示，应另取相同样本。鲁棒性位移实验和多种子重复在首轮有收益后再做。

实际抽查显示 `rain_interpolated` 与稀疏 `rain` 的均值及雨区面积差异很大；该因素三组一致，不能用这轮实验判断标签质量是否正确。网格/时间定义和原3D decoder的已知因果限制继续保留在结论边界中。

## 6. 离开与恢复终端

依次按 Ctrl+B、松开、D 脱离 tmux，之后可以关闭本地 SSH。

```bash
tmux attach -t rain-local
```

恢复训练需使用本组自己的 checkpoint、同一 YAML、batch、累积和模型参数。例如 R1：

```bash
python -m src.trainer.rain_local_fusion_trainer \
  train.resume_path=/home/yanjieyu/RainPrediction/runs/rain_local_fusion/实际日期/实际目录/checkpoint-00002000
```

上述 checkpoint 路径必须替换后再执行。模型/optimizer/scheduler/controller 可恢复，但原数据迭代器位置不保证逐位恢复；用于最终配对对照时记录任何中断。
