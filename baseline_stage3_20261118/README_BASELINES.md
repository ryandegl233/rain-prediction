# Stage 3 baseline 运行说明

仓库已新增全部六个深度 baseline，最新的 Git 拉取、训练和评测命令见 [README_DEEP_BASELINES.md](README_DEEP_BASELINES.md)。下文保留当前方法与 Persistence/LK 的设置说明；最新交付使用仓库分支，不再依赖 ZIP。

本包依据提供的 `src.zip` 和最终 `rain_trainer_ts_next_frame_missing_modality_stage3_detach_off.yaml`，接通当前方法、Persistence、PySTEPS-LK 的公共评测。原始源码与最终 YAML 完整保留，新增实现位于 `src/baselines_stage3/`，使用独立配置 `baseline_stage3_train.yaml`。不要用上一版 `src.baselines.train` 命令启动本包。

## 已对齐的设置

| 设置 | 本包新入口 |
| --- | --- |
| 网络 | 原 `RainCausalPatchTransformerMissingModality`，44,162,136 参数 |
| 输入与预测 | 历史 4 帧，未来 4 帧；30 分钟间隔；448×448 |
| 历史缺失模态 | 第 1 帧全模态；其余逐帧 60% 全模态，其余六种非空组合各 40%/6 |
| sequence loss | 沿用原实现；雷达/卫星/降雨权重 1/1/2，context/future 权重 1/1 |
| rollout loss | 仅 rain MSE，外层权重 0.5 |
| 梯度 | `detach_history=true`，`detach_rain_anchor=false` |
| 辅助观测 | 训练和验证均 `use_gt_future_modalities=true` |
| 历史窗口 | 最多 7 帧 |
| 训练长度 | **70,000 optimizer steps**，Stage 3 从初始化权重开始计数；不是 70,000 个数据集 epoch |
| 初始化 | 保留最终 YAML 的 Stage 2 24k EMA 转换权重路径，可通过命令行指定实际文件 |
| 优化 | batch=2，累积=1，BF16，学习率 2e-5，warmup=300，scheduler.min_lr=1e-6（原 LambdaLR 的倍率，实际下限 2e-11） |
| 数据筛选 | `metadata_rain_ratio.parquet`，`rain_ratio_gt_0p02 >= 0.02`，`future_any` |

这套评测属于逐步更新观测的预测：+30 分钟预测完成后，才读入真实 +30 分钟雷达/卫星来预测 +60 分钟；后续同理。未来真实降雨始终只作标签。Persistence/LK 使用历史可见雨量，报告会明确各模型是否使用新增辅助观测，不能把这些结果与此前“纯历史固定起点”结果混标。

默认同时验证 `full` 和 `sampled` 两种历史输入。`sampled` 按数据中的日历时间与固定种子生成掩码，所有模型共用，改变 batch size 不改变掩码，也不消耗训练的随机数状态。原 dataset 用无时区日期生成 POSIX 时间戳；新评测在同一主机还原日历时间作为窗口标识，避免上海与 UTC 服务器生成不同掩码。报告同时保留原 POSIX 时间戳供审计。训练仍调用同学原有随机采样代码，包括 sequence loss 中未来 teacher-forcing 种子帧的缺失采样。

## 服务器拉取和准备

使用现有能训练同学源码的 Python 环境（Python >=3.12），在服务器仓库根目录更新分支：

```bash
git fetch origin
git switch ryan/baseline-evaluation
git pull --ff-only origin ryan/baseline-evaluation
cd baseline_stage3_20261118
```

以下 `ORIGINAL_PROJECT` 按服务器实际项目位置修改。目录用于引用已有数据，`INIT_WEIGHTS` 必须指向实际的 Stage 2 初始化权重：

```bash
ORIGINAL_PROJECT=/home/dongchenwang/RainPrediction
INIT_WEIGHTS="$ORIGINAL_PROJECT/runs/next_frame_final_stage3_detach_off/init/stage2_24k_ema/model.safetensors"
ln -s "$ORIGINAL_PROJECT/data2" data2
test -f "$INIT_WEIGHTS"

python -m pip install -r requirements-baselines.txt
python -m pytest src/tests/baselines_stage3 -q
```

数据和权重不提交到仓库。`data2` 符号链接只需首次建立；如已经正确连接，不重复执行 `ln -s`。pySTEPS 固定为 1.21.5，使用安装包，不把其 Git 主分支源码添加到 PYTHONPATH。若 pip 提示编译器缺失，需要服务器提供 C/C++ 编译工具。

新入口在训练前检查数据目录和过滤 metadata。缺少过滤文件时会明确报错，不采用原 trainer 自动关闭过滤的行为，以保证不同方法评测同一批窗口。直接覆盖 `dataset.train.inp_dirs` 或 `dataset.val.inp_dirs` 时，也会同步原 trainer 用于检查的顶层路径。

## 先短测，再训练 70,000 steps

先跑两批真实数据的传统 baseline，验证数据和 LK 环境：

```bash
python -m src.baselines_stage3.evaluate evaluation.max_batches=2
```

然后用真实初始化权重跑两个 optimizer steps，检查 448×448 前向、反向、显存和保存路径：

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.baselines_stage3.train \
  "train.init_model_path=$INIT_WEIGHTS" \
  train.max_steps=2 train.save_every=2 val.val_duration=2 \
  evaluation.max_batches=2 dataset.val.batch_size=1 \
  train.log.run_comment=stage3_baseline_smoke
```

通过后正式启动 70,000 steps。以下使用 `nohup` 在 SSH 断开后继续运行，并保存独立日志：

```bash
mkdir -p logs
RUN_LOG="logs/stage3_baseline_$(date +%Y%m%d_%H%M%S).log"
nohup env CUDA_VISIBLE_DEVICES=0 python -u -m src.baselines_stage3.train \
  "train.init_model_path=$INIT_WEIGHTS" > "$RUN_LOG" 2>&1 &
echo "PID=$! LOG=$RUN_LOG"
tail -f "$RUN_LOG"
```

新入口目前支持单进程、单 GPU。默认每 1,000 steps 验证与保存，训练中验证最多 60 批，标记 `complete=false`。最终主表请使用下面的完整评测命令。不要用短测的 `train.max_steps=2` 命令作为正式训练命令。

## 结果、完整评测与恢复

训练结果在 `runs/baseline_stage3/<日期>/<时间>_stage3_baselines_70k/`。每次公共评测输出 `baseline_validation/step_XXXXXXXX/metrics.json`、`metrics.csv` 和 `windows.json`；包含逐时效与总体 MAE/MSE/CSI/POD/FAR、事件计数、参数量、样本数量、完整配置及窗口/掩码 SHA-256。无事件且分母为零的指标为 null。雨量沿用原数据数值，不未经核实标为 mm/h。

checkpoint 位于 `<run>/checkpoints/step_XXXXXXXX/`，包含原训练状态、`model.safetensors`、`ema.pt` 和 `metadata.json`。遵循项目禁止批量删除的要求，**新入口保留全部 checkpoint，不自动执行同学源码中的目录清理**。`checkpoints/latest.json` 指向最近的完整保存。

完整评测某个 checkpoint，`CKPT` 替换为实际目录：

```bash
CKPT=/absolute/path/to/run/checkpoints/step_00070000
CUDA_VISIBLE_DEVICES=0 python -m src.baselines_stage3.evaluate \
  "evaluation.reference_checkpoint=$CKPT" evaluation.device=cuda \
  evaluation.max_batches=-1 dataset.val.batch_size=1
```

默认评测在线权重。评测相同目录中的 EMA 时追加 `evaluation.checkpoint_kind=ema`。模型权重加载使用 strict 模式。只评测 Persistence/LK 时省略 checkpoint 参数即可，不需要初始化权重或 GPU。

训练内验证与独立评估共用 `evaluation.mixed_precision`，默认继承 BF16；可显式设为 `no` 或 `fp16`，比较指标时应保持一致。报告记录每个方法的实际设备和精度设置；Persistence/LK 使用 CPU float32 输入，LK 内部沿用 pySTEPS 数值运算。

断点恢复可以传具体 step 目录或包含 `latest.json` 的 checkpoints 目录：

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.baselines_stage3.train \
  train.resume_path=/absolute/path/to/run/checkpoints
```

提供 `resume_path` 后，使用 checkpoint 恢复模型、优化器、调度器、EMA 和步数，不再要求 Stage 2 初始化文件。默认目标仍是总计 70,000 steps，不是在恢复步数上另加 70,000。

若 checkpoint 已达到目标步数，新入口直接结束，不额外训练一步。原数据加载器没有保存每个 worker 的迭代位置，因此恢复可以继续优化，但不保证与不中断训练具有逐位相同的采样轨迹。

需要逐种缺失组合评测时追加：

```bash
'baseline.scenarios=[full,sampled,radar+satellite,radar+rain,satellite+rain,radar,satellite,rain]'
```

这些固定组合只约束第 2–4 帧，第 1 帧始终全模态；未来逐步到达的雷达/卫星仍完整。因此这里的 `satellite` 不是严格 satellite-only 的 NPM 专项设定。

## 传统方法与验证范围

Persistence 重复最近一次可见降雨。LK 在完整历史上用四帧估计速度；雨量历史有缺失时用最近两次观测、按间隔归一速度，并从最后观测时刻外推到四个目标时刻。仅有首帧降雨时，使用零运动外推；报告的 `availability_counts.<scenario>.rain_observations[1]` 给出这类样本数量。

`SOURCE_MANIFEST.json` 记录 398 个未修改源文件和原 ZIP 的 SHA-256。完整网络参数统计在 `artifacts/reference_profile.json`，可以用 `python -m scripts.profile_stage3` 重测。`src/baselines_stage3/metrics.py` 复用此前项目指标文件。

本地测试使用缩小模型与合成数据，覆盖输入隔离、跨时区/批次的掩码复现、LK 不规则时间间隔、原 trainer 与新推理包装的一致性、降雨锚点梯度、loss 组合、训练更新、在线/EMA 权重加载及断点恢复。没有在本机运行真实气象数据或完整尺寸 GPU 训练。ConvLSTM、PredRNN-V2、SimVP、Earthformer、AlphaPre、NPM 已新增独立训练入口，详见 README_DEEP_BASELINES.md。
