# num8 降雨优先门控实验

本实验使用独立入口 `src/trainer/rain_priority_trainer.py` 和配置 `src/config/ts_rain_train/rain_priority.yaml`。原 3D decoder、4 帧历史 / 4 帧未来、30 分钟数据、右移 7 个 teacher-forcing 目标均保留。新模型从头训练，不加载之前 2D 或旧门控实验的 checkpoint。

新方案的预测收益尚未验证。原 3D decoder 的时间填充/归一化因果性问题仍存在；本实验用于比较门控与 loss，不代表该问题已经修复。先通过 smoke，再做短对照，不直接承诺完整训练一定提升 CSI。

## 1. 同步代码

以下命令均在 **num8 的 Bash** 中逐段执行。先查看状态：

```bash
cd /home/yanjieyu/RainPrediction
git status --short --branch
git fetch origin
```

如果存在服务器本地改动，先保留并检查；下面的快进拉取如报冲突，停止处理冲突，不能用 `reset --hard` 或清理命令覆盖。

```bash
git -c merge.autostash=false pull --ff-only --no-rebase origin project
git log -1 --oneline
```

拉取后的提交应与交付消息一致，或为包含该提交的后续版本。`config_total.yaml` 是已运行实验的快照，修改它不会改变正在运行的训练；新实验使用上面列出的新 YAML。

## 2. 建立 tmux 会话并进入环境

```bash
tmux ls
nvidia-smi
```

同一张 4090 上不要同时启动旧训练和新训练。旧任务仍占用 GPU 时，先进入对应会话，等需要保留的 checkpoint 完成，再在该训练终端按 Ctrl+C 停止。同步代码本身不会停止旧进程，也不会把旧进程变成新框架。

```bash
tmux new -s rain-priority
```

进入新会话后：

```bash
cd /home/yanjieyu/RainPrediction
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate rainpred
```

检查配置能解析：

```bash
python -m src.trainer.rain_priority_trainer --cfg job --resolve
```

应看到 `decoder_type: 3d`、`n_past: 4`、`n_futures: 4`、`mixed_precision: bf16`、train batch 1、`num_steps: 2`。有效 batch 为 2；累积 2 次是两次 micro-batch，不是扩大单次显存占用。

## 3. 20 个 optimizer step 的 smoke

```bash
python -m src.trainer.rain_priority_trainer \
  train.max_steps=20 \
  train.save_every=20 \
  val.val_duration=20 \
  val.max_val_iters=1 \
  train.priority.horizon.warmup_steps=4 \
  train.priority.horizon.update_every=2 \
  train.priority.gradient_diagnostics_every=5 \
  train.scheduler.num_training_steps=100000 \
  train.log.run_comment=num8_priority_smoke
```

验收：正常退出，无 OOM/NaN；出现 `[Val][20]`、checkpoint 保存、`Training finished` 和 CUDA 峰值；`validation/` 中有 JSON 和原始可视化。此 smoke 特意缩短控制器窗口来覆盖动态更新。**正式实验从头开始，不续训这个 smoke checkpoint**。

程序会检查绝对时间戳间隔：正常样本应为 +1800 秒，原反向时间增强为 -1800 秒。出现此检查错误时保留输出检查数据，不能把检查关掉后继续跑。该要求避免重复或错位帧被当成正常时序差分。

有 CUDA 的环境还可以运行小模型 BF16 集成测试（需要 pytest）：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest \
  src/tests/rain_priority/test_trainer.py::test_cuda_bf16_routed_backward_and_checkpoint \
  -q -p no:cacheprovider
```

## 4. 先跑短对照，再延长

最低对照是 A（清理 loss、关闭门控、固定时效权重）与 E（完整新方案）。两组使用相同种子、数据、batch、步数和全验证集。先跑 A 能分辨收益是否只是清理 loss 产生的；不要仅把不同随机子集上的新 CSI 与旧日志比较。

**A，6000 步清理 loss 对照：**

```bash
python -m src.trainer.rain_priority_trainer \
  rain_prediction_model.spatial_modality_gate_enabled=false \
  train.priority.horizon.enabled=false \
  train.max_steps=6000 \
  train.save_every=2000 \
  train.scheduler.num_training_steps=100000 \
  train.log.run_comment=num8_priority_A_clean_6000
```

**E，6000 步完整新方案：**

```bash
python -m src.trainer.rain_priority_trainer \
  train.max_steps=6000 \
  train.save_every=2000 \
  train.scheduler.num_training_steps=100000 \
  train.log.run_comment=num8_priority_E_full_6000
```

这两条依次运行，不能在一张卡上并发。每 2000 步遍历全部验证窗口（允许随机顺序），单次验证可能比旧的 60 个 batch 更久。第一次全验证完成前不能按旧吞吐估计总时长。

每个新进程生成独立目录：`runs/rain_priority/日期/时间_备注/`，包含配置快照、日志、TensorBoard、checkpoint 和 `validation/step-xxxxxxxx.json` / `step-xxxxxxxx-raw.png`。

重点比较第 2000、4000、6000 步的：

- 各阈值的平均 CSI 及 +120 分钟 CSI，与相同验证集合的 A 和 persistence 对比。
- FAR/POD、干区预测均值/分位数、湿区面积，防止靠扩大雨区换取 POD。
- 门控来源分组与输入干湿区统计、门控梯度、共享参数梯度夹角，以及四个 `weight_used`。
- 吞吐和显存。CPU 检查不能保证 24GB GPU 上的峰值；当前默认 batch 1、累积 2、BF16、activation checkpoint。

E 的 6000 步结果有改善、FAR 未明显恶化时，可以续训到 100000。将下面路径替换为 **E 实验真正保存的 checkpoint 目录**：

```bash
python -m src.trainer.rain_priority_trainer \
  train.resume_path=/home/yanjieyu/RainPrediction/runs/rain_priority/实际日期/实际时间_num8_priority_E_full_6000/checkpoint-00006000 \
  train.max_steps=100000 \
  train.save_every=5000 \
  train.scheduler.num_training_steps=100000 \
  train.log.run_comment=num8_priority_E_resume
```

恢复时保持控制器配置、batch 和累积设置一致。模型、optimizer、scheduler、controller 均恢复；数据迭代器在数据集内的精确位置未保存，所以不承诺与不中断训练的完整轨迹逐位相同。给定相同后续 batch 的小模型参数更新一致性有测试。若改成 batch 2，应先重新 smoke，并把 `accelerator.gradient_accumulation_plugin.num_steps=1`；不能只把 batch 改大而无意翻倍有效 batch。

从头跑 100000 步的 E 命令（完成短对照后使用）：

```bash
python -m src.trainer.rain_priority_trainer \
  train.log.run_comment=num8_priority_E_100k
```

## 5. 完整消融矩阵

均使用同一个入口，在共同的 6000 步参数上追加以下覆盖；先 A/B/C，再 D/E/F 能逐项定位效果。

| 组 | 比较问题 | CLI 覆盖 |
| --- | --- | --- |
| A | 清理 loss 的无门控基线 | `rain_prediction_model.spatial_modality_gate_enabled=false train.priority.horizon.enabled=false` |
| B | 原逐帧空间门控、各任务共同训练门控 | `rain_prediction_model.gate_mode=spatial train.priority.rain_only_gate=false train.priority.horizon.enabled=false` |
| C | 门控仅接收降雨梯度的影响 | `rain_prediction_model.gate_mode=spatial train.priority.horizon.enabled=false` |
| D | 增加时序差分及输入来源条件 | `train.priority.horizon.enabled=false` |
| E | 完整新方案 | 无额外覆盖 |
| F | 无门控、仅动态时效 loss | `rain_prediction_model.spatial_modality_gate_enabled=false` |

全局梯度裁剪仍作用于整个模型，辅助任务可以通过整体裁剪系数间接影响门控步长；“雨量专属梯度”指裁剪前门控的直接梯度来源。辅助任务也仍影响共享主干，不能把它描述为完全消除跨任务影响。

## 6. 离线运行与返回

按 `Ctrl+B`，松开，再按 `D` 离开 tmux；随后可关闭本机 SSH。返回：

```bash
tmux attach -t rain-priority
```

只查看 TensorBoard 可在另一个终端运行（不要打断训练终端）：

```bash
tensorboard --logdir /home/yanjieyu/RainPrediction/runs/rain_priority --host 127.0.0.1 --port 6006
```

本地 Windows PowerShell 转发：

```powershell
ssh -p 1338 -L 6006:127.0.0.1:6006 yanjieyu@100.78.88.21
```

本机浏览器打开 `http://127.0.0.1:6006`。只访问 TensorBoard 时该 SSH 转发需保持连接，训练本身仍由 tmux 托管。
