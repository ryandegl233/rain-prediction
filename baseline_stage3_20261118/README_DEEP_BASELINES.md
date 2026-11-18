# 八个 baseline 的仓库运行入口

表中八个 baseline 均有独立实现入口：Persistence、PySTEPS-LK，以及本次新增的 ConvLSTM、PredRNN-V2、SimVP、Earthformer、AlphaPre、NPM。原方法通过 `src.baselines_stage3.train` 训练；六个深度 baseline 通过 `src.baselines_stage3.train_deep` 训练，共用数据读取与指标计算。

本目录保留同学提供的完整源码，避免覆盖仓库根目录的另一版 trainer。**以下 Python 命令均在 `baseline_stage3_20261118/` 内运行。** 模型运行所需上游代码已随独立适配目录纳入仓库，服务器不需要另克隆 OpenSTL、Earthformer、AlphaPre 或 NPM，也不依赖本机 `third_party/baselines` 中被 Git 忽略的检出。

## 拉取与准备

在服务器仓库根目录更新本次分支：

```bash
git fetch origin
git switch ryan/baseline-evaluation
git pull --ff-only origin ryan/baseline-evaluation
cd baseline_stage3_20261118
```

使用已经能训练原方法的 Python >=3.12 环境，安装增量依赖。若父目录已有 `data2`，首次建立数据链接；数据与权重不会提交到 Git。

```bash
python -m pip install -r requirements-deep-baselines.txt
test -e data2 || ln -s ../data2 data2
python -m pytest src/tests/baselines_stage3 -q
python -m scripts.profile_deep_baselines
```

训练前检查数据目录与 `metadata_rain_ratio.parquet`。缺少 metadata 会报错，不会静默取消数据过滤。两个与独立原始 checkout 比较的测试在服务器没有原始 checkout 时会 skip；随仓库携带的文件哈希及全部模型训练/推理测试仍会执行。

## 六种方法的配置

| CLI 选择 | 方法 | 总参数（含公共 mask token） | 相对当前方法 |
| --- | --- | ---: | ---: |
| `baseline_model=convlstm` | ConvLSTM | 44,115,132 | 0.9989× |
| `baseline_model=predrnn_v2` | PredRNN-V2 | 44,110,764 | 0.9988× |
| `baseline_model=simvp` | SimVP，原始 IncepU | 43,830,168 | 0.9925× |
| `baseline_model=earthformer` | Earthformer | 43,994,260 | 0.9962× |
| `baseline_model=alphapre` | AlphaPre | 44,702,307 | 1.0122× |
| `baseline_model=npm` | NPM 卫星预报器 + 卫星到雨量翻译器 | 45,431,153 | 1.0287× |

当前方法为 44,162,136 参数。容量按总参数规模对齐，结构层数按各方法保留：不同架构的层数、FLOPs、显存不能互相等同。ConvLSTM 官方 cell 注册了前向未使用的模块；其有效参数为 34,942,332（含 mask token），总数与有效数同时记录，不将闲置参数算作有效计算容量。NPM 参数包含两个阶段，分别为 5,306,122 和 40,125,031。

## 训练协议与迁移边界

五种通用深度 baseline 使用历史四帧预测未来四帧；第 1 帧三模态完整，其余帧按 60% 全模态、其余六种组合各 40%/6 训练。固定七个时间槽容纳逐步增长的历史，起始不足七帧时左侧补零。三种模态通道数为 1/10/1，原始分辨率 448×448，间隔 30 分钟。

沿用最终 Stage3 配置：sequence loss 的雷达/卫星/降雨权重 1/1/2，历史三次转换与未来四次转换分别取均值，再按 context/future 权重 1/1 归一；rollout rain MSE 外乘 0.5。历史输入 `detach_history=true`，独立残差锚点 `detach_rain_anchor=false`。每个预测完成后才读入该时刻真实雷达/卫星，预测过程中始终不读未来真实 rainfall。teacher forcing 在训练监督分支使用先前标签，与预测评测分开。

PredRNN-V2 保留系数 0.1 的解耦损失。AlphaPre 保留相位、幅度及分支监督；为对齐共享的雨量残差输出，其雨通道幅度分支采用线性激活，辅助监督使用同一个残差目标，其他通道仍采用原 sigmoid。该变更位于独立适配器，原 AlphaPre 文件逐字节保持不变。AlphaPre 频域内核固定 FP32，以支持 448 分辨率的 FFT；其他部分可使用 BF16。

NPM 自动采用单独 `satellite_only` 场景：全部四帧仅卫星，不读任何历史雷达/降雨或未来观测，固定起点预测四帧。它保留公开代码的两段模型，十通道适配，日历时间条件使用历史帧的公历年内日与小时小数。原数据雨量没有统一上限，因此翻译器末端改为线性输出。迁移训练使用卫星 MSE×1 + 雨量 MSE×2，区别于原论文/代码的分阶段训练；它是公开模型在本数据上的统一迁移比较，不能称为论文完整训练方案复现，不能与 GT 更新的主表直接混合排名。详细差异见 `doc/npm-adaptation.md`。

六种方法均从随机初始化开始训练 **70,000 optimizer steps**，不加载当前方法的 Stage2 权重。默认继承最终 YAML 的 AdamW、学习率 2e-5、warmup 300、`scheduler.min_lr=1e-6`、batch 2、EMA。注意原调度器将 `min_lr` 用作 LambdaLR 的倍率，实际学习率下限是 **2e-11**，不是 1e-6；本次保持原行为，并在日志记录实际 lr。当前方法另有 Stage2 预训练，论文需同时报告初始化与训练预算。若另行调学习率，用独立实验命名记录。

## 先短测，再启动训练

每种模型先用真实数据跑两步检查显存。以 SimVP 为例：

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.baselines_stage3.train_deep \
  baseline_model=simvp train.max_steps=2 train.save_every=2 \
  val.val_duration=2 evaluation.max_batches=2 dataset.val.batch_size=1 \
  train.log.run_comment=simvp_smoke
```

正式训练单种模型：

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m src.baselines_stage3.train_deep baseline_model=simvp
```

一张 GPU 顺序跑完六种方法，使用 Bash；任一任务失败即停止，避免继续启动后掩盖错误：

```bash
mkdir -p logs
nohup bash -c '
  set -e
  for model in convlstm predrnn_v2 simvp earthformer alphapre npm; do
    CUDA_VISIBLE_DEVICES=0 python -u -m src.baselines_stage3.train_deep "baseline_model=$model"
  done
' > "logs/six_baselines_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

多张 GPU 可在各终端分别指定 GPU 与方法。训练入口支持单进程单 GPU，精度为 BF16 或 `accelerator.mixed_precision=no`；不支持 FP16 训练。默认启用核心模型激活检查点。若 batch 2 显存不足，可用 `dataset.train.batch_size=1 accelerator.gradient_accumulation_plugin.num_steps=2` 保持有效 batch 2，并在该运行的所有恢复命令保持一致。AlphaPre 默认宽度的 GPU 内存与速度需以服务器短测为准。

## 完整评估与恢复

每种运行保存在 `runs/deep_baselines/<method>/<日期>/<时间>_<说明>/`，包含 resolved config、逐步 JSONL loss、模型与 EMA 权重、优化器/调度器/RNG 状态、统一指标及窗口掩码。默认训练中每 1,000 步评估最多 60 批；主表用完整评估：

```bash
CKPT=/absolute/path/to/run/checkpoints/step_00070000
CUDA_VISIBLE_DEVICES=0 python -m src.baselines_stage3.evaluate_deep \
  baseline_model=simvp "evaluation.reference_checkpoint=$CKPT" \
  evaluation.max_batches=-1 dataset.val.batch_size=1
```

追加 `evaluation.checkpoint_kind=ema` 可评测 EMA；所有方法比较时使用相同权重选择、精度及评测窗口。NPM 将方法换为 `baseline_model=npm`，自动采用自己的独立协议。Persistence/LK 仍可直接执行 `python -m src.baselines_stage3.evaluate evaluation.max_batches=-1`。

中断后从完整 checkpoint 继续到原目标步数：

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.baselines_stage3.train_deep \
  baseline_model=simvp train.resume_path=/absolute/path/to/run/checkpoints
```

恢复必须保持模型、目标步数、优化器/调度器、EMA、loss、缺失采样与累积设置一致；程序会核对训练签名。评估也核对模型和雨量残差解释。达到目标步数的恢复不会额外更新。原数据加载器没有保存 worker 位置，恢复不保证逐位相同的数据采样轨迹。所有 checkpoint 均保留，不自动删除；长训练需为多次完整保存准备磁盘空间。

## 来源与验证

- [OpenSTL](https://github.com/chengtan9907/OpenSTL)，固定提交 `eecf8a3078f0a178dbc7b28723da20f94ce36985`。
- [Earthformer](https://github.com/amazon-science/earth-forecasting-transformer)，固定提交 `7732b03bdb366110563516c3502315deab4c2026`。
- [AlphaPre](https://github.com/linkenghong/AlphaPre)，固定提交 `64d5195984c72db2692266acc800efa34ff8a4bf`。
- [NPM](https://github.com/seominseok0429/Data-driven-Precipitation-Nowcasting-Using-Satellite-Imagery)，固定提交 `52ca97702f9d9b7e314cf0ce9e053d3aa06c5bea`。

各 port 的来源、原文件 SHA-256、导入/接口改造说明及上游提供的许可文件随代码保存。测试覆盖真实小模型前后向、优化更新、EMA/断点恢复、完整评测、输入隔离、因果顺序和原代码数值一致性。参数统计使用最终完整配置。NPM 另外验证默认 448 前向；AlphaPre 的 448 FFT 检查使用小宽度。尚未执行六种完整配置的真实数据 GPU 训练，不把本地测试当作训练完成。
