# Baseline 公共评测与服务器运行

这次完成 Persistence、PySTEPS-LK 和指定 `rain_trainer_ts_next_frame.yaml` 对应网络的统一评测；新训练入口继续训练该参考网络，并在验证阶段同时计算三者的指标。Persistence/LK 没有可训练参数。ConvLSTM、PredRNN-V2、SimVP、Earthformer、AlphaPre、NPM 的源码已登记，训练适配尚未在本阶段实现。

## 统一后的接口

- 输入：`HistoryBatch`，包含 `[B,12,4,H,W]` 历史、三组模态可用性以及可选历史时刻。通道顺序 radar=1、satellite=10、rain=1。模型接收的对象没有未来真值。
- 输出：`[B,1,4,H,W]`，对应固定起点后的 30、60、90、120 分钟。参考配置空间尺寸 448×448。
- 缺失模态：原始历史清零；参考网络使用既有 mask token，之后只回填自己的预测。缺失 rain 时残差初值为零，Persistence/LK 标记为 skipped。
- 指标：复用项目 `RainLeadMetrics`，在相同完整验证窗口上累计事件数，输出各时效与总体 MAE/MSE/CSI/POD/FAR，阈值继承 `[0.1,0.3,0.5]`。无事件时分母为零的指标保存为 null。
- 量纲：沿用未零中心化的 `rain_interpolated` 原始数值；没有把它未经核实标为 mm/h。
- 来源：参考网络保持 44,162,136 参数；两个传统方法为 0 参数。配置、依赖版本、源码登记、样本数量与时间窗口 SHA-256 写入每次报告。

LK 使用已安装的 `pysteps==1.21.5`。用四帧降雨的 dB 场估计稠密 LK 光流，在最后一帧原始降雨量上进行半拉格朗日外推；阈值 0.02、形态学 opening=3、双线性插值、区域外填零。可通过 `baseline.lk` 配置调整，并随报告记录。`third_party/baselines/pysteps` 的主分支是源码参考，其接口已不同，不要将它加入 PYTHONPATH。

## 拉取和环境

在服务器项目根目录、现有训练环境中执行。分支为 `ryan/baseline-evaluation`；以下命令会将该分支快进合并到当前分支，如服务器有本地分叉，Git 会拒绝而不会覆盖。

```bash
git pull --ff-only origin ryan/baseline-evaluation
python -m pip install -r scripts/baselines/requirements.txt
python -m pytest src/tests/baselines -q
```

沿用已有 PyTorch/CUDA、Accelerate、Hydra、litdata 等项目训练依赖。pySTEPS 含编译扩展；Linux 环境需要可用的 C/C++ 编译器。数据路径仍继承原 YAML 的 `data2/litdata_train_2025/litdata_interval_30/`，并沿用各月 `metadata_rain_delta_filter.parquet` 过滤。

源码仓库不随主项目重复提交。需要在服务器恢复五个固定版本源码时执行：

```bash
python scripts/baselines/fetch_sources.py
```

该脚本只在目录不存在时克隆并检出登记的提交；已有目录仅核对 remote、commit 和干净状态，不覆盖源码、不删除文件。Persistence/LK 运行本身不需要执行此步。

## 先运行两批真实验证数据

```bash
python -m src.baselines.evaluate evaluation.max_batches=2
```

这不构造训练模型，不消耗 GPU。成功后进行完整传统 baseline 评测：

```bash
python -m src.baselines.evaluate
```

结果默认写入 `runs/baseline_evaluation/<日期>/<时间>/`，包括 `metrics.json`、`metrics.csv`、`windows.json` 和 `config_resolved.yaml`。`evaluation.max_batches=-1` 才是正式完整验证，并检查样本数等于验证集长度；有限批次报告标记 `complete=false`。

## 训练当前参考网络

先运行两个 optimizer step，检查真实数据、显存和保存路径：

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.baselines.train \
  train.max_steps=2 train.save_every=2 val.val_duration=2 \
  evaluation.max_batches=2 train.log.run_comment=baseline_smoke
```

成功后正常训练：

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.baselines.train
```

入口显式继承用户指定的 `rain_trainer_ts_next_frame.yaml`，沿用原模型、损失、优化器、调度器、EMA 与模态 dropout。新增配置关闭训练 rollout 和验证中的未来真实辅助模态回填。训练中的 next-frame/sequence teacher forcing 仍是原训练目标，应与纯历史推理协议分别记录。

默认单进程单 GPU、448×448、BF16，micro batch=1、梯度累积=8，保持有效 batch=8；总 optimizer steps=100000。不要使用多进程 `torchrun` 或多 GPU Accelerate 启动此入口。完整尺寸的显存和真实数据吞吐需要以上服务器 smoke 确认，本机验证使用 CPU 和缩小模型。

训练目录在 `runs/baseline_reference/`。每次验证将 reference、persistence、lk 的公共报告写入 `baseline_validation/step-XXXXXXXX/`，并记录到 TensorBoard。默认评测在线模型，可用 `evaluation.checkpoint_kind=ema` 评测 EMA。原 `val.max_val_iters` 不控制此入口，评测批次数统一由 `evaluation.max_batches` 控制。

checkpoint 在 `checkpoint-XXXXXXXX/`，模型权重和 `meta.json` 同目录；EMA 沿用原 trainer 的 `<run>/ema/ema.pt`。新配置关闭自动 checkpoint 命名和旧权重自动清理，不删除以往 checkpoint。继续同一训练时指定：

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.baselines.train \
  train.resume_path=/absolute/path/to/checkpoint-00001000
```

## 统一重评估已有模型

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.baselines.evaluate \
  evaluation.device=cuda \
  evaluation.reference_checkpoint=/absolute/path/to/checkpoint-00001000
```

加载使用 strict 模式，模型结构必须与 checkpoint 一致。评测 EMA 时传实际 EMA 文件，并附 `evaluation.checkpoint_kind=ema`。旧 checkpoint 如果训练曾使用未来真实雷达/卫星，应记录这个训练条件；新评测不会继续提供这些观测。

一次评测所有七个非空模态组合，可给评测或训练命令追加：

```bash
'baseline.combinations=[[radar,satellite,rain],[radar,rain],[satellite,rain],[rain],[radar,satellite],[radar],[satellite]]'
```

默认仅全模态。七组会增加参考网络验证开销；相同雨量历史下 Persistence/LK 的预测在组间复用，不含 rain 的三组明确跳过。各模型的 `window_sha256`、验证配置和 `complete` 应一致后再汇总比较。

## 本地验证边界

自动测试涵盖真实 pySTEPS 静止场/平移、未来输入隔离、缺失模态、尾批次与指标聚合，以及小型原网络的四步 rollout、训练更新、报告保存和 checkpoint 严格重载。CLI 另用 3 个合成窗口验证完整报告生成。合成测试不代表真实气象数据上的预测性能；服务器完整 448×448 训练尚需执行上述 smoke。

2026-11-18 本地验证结果：19 项 pytest 全部通过；Ruff 检查通过。原网络使用 16×16、dim=8、depth=1 的测试配置，执行两次 micro batch、一次 optimizer update，验证完整三样本报告与保存/重载后的预测一致。正式 YAML 仍保留 448×448、dim=384、depth=6。测试环境为 Windows / Python 3.13 / PyTorch 2.12 CPU，依赖在项目外隔离安装；服务器继续使用原训练环境。
