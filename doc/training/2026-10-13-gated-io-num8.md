# num8 空间门控与 I/O 训练

本轮实验使用 `GatedModalityRainModel` 的空间门控，从头训练。独立 I/O trainer 继承我们自己的 `GatedModalityRainTrainer`，修正训练循环使用的 prepared DataLoader，并将每个 batch 的模型输入统一转换为设备上的 float32，供主损失和 rollout 分支共享。原模型、原 trainer、baseline 和同学的源码快照均不修改。

入口为 `src/trainer/gated_modality_rain_io_trainer.py`，配置为 `src/config/ts_rain_train/gated_modality_rain_io.yaml`。其余设置继承已有空间门控配置：30 分钟数据、448×448 输入、4 帧历史和 4 帧未来、原 loss、原优化器及固定起报 rollout。面向 RTX 4090 的 24GB 显存，配置显式启用 BF16 和 activation checkpointing，训练 batch size 和验证 batch size 均为 1。`resume_path` 与 `init_model_path` 均为空，CCC、跨模态 adapter 和局部窗口模块本轮关闭，代码和后续启用能力保留。

## 更新服务器代码

在 num8 的终端执行，每个代码块单独执行。

```bash
tmux new-session -A -s gate-io
```

```bash
conda activate rainpred
```

```bash
cd /home/yanjieyu/RainPrediction
```

```bash
git status --short --branch
```

若有服务器上的源码改动，先保存并确认它们；工作区干净时继续：

```bash
git -c merge.autostash=false pull --ff-only --no-rebase origin project
```

```bash
git log -1 --oneline
```

## 确认 GPU、数据与测试

```bash
nvidia-smi --query-gpu=index,name,memory.free --format=csv
```

```bash
df -h ~
```

数据路径是相对于仓库的，必须从 `/home/yanjieyu/RainPrediction` 启动。检查全部 9 个月及配置要求的元数据：

```bash
python - <<'PY'
from pathlib import Path
from hydra import compose, initialize_config_dir

with initialize_config_dir(config_dir=str(Path('src/config/ts_rain_train').resolve()), version_base=None):
    cfg = compose(config_name='gated_modality_rain_io')

missing = []
for directory in [*cfg.dataset.train_inp_dirs, *cfg.dataset.val_inp_dirs]:
    root = Path(directory)
    required = [root / 'index.json', root / 'metadata.parquet']
    if cfg.dataset.rain_ratio_filter.enabled:
        required.append(root / cfg.dataset.rain_ratio_filter.file_name)
    absent = [str(path) for path in required if not path.is_file()]
    missing.extend(absent)
    print(f'{directory}: {"OK" if not absent else "MISSING"}')

if missing:
    raise SystemExit('Missing data files:\n' + '\n'.join(missing))
print('Data preflight passed; resume_path:', cfg.train.resume_path, 'init_model_path:', cfg.train.init_model_path)
PY
```

如果检查发现缺少 `metadata_rain_delta_filter.parquet`，先核对学校服务器对应目录是否存在此文件，并补充同步；不要为了启动训练直接改变样本筛选配置。

```bash
python -m pytest -q src/tests/gated_modality_rain/test_io_trainer.py
```

这是 CPU 与 CUDA 的小尺寸功能检查，CUDA 可用时应执行全部 9 项测试。它不等同于 448×448 真实数据训练通过；如果环境未安装 pytest，可以先运行下一节的真实数据测试，并反馈缺失依赖，不要直接升级迁移后的训练环境。

## 真实数据：5 步训练与一次验证

```bash
python -m src.trainer.gated_modality_rain_io_trainer \
  train.max_steps=5 \
  val.val_duration=5 \
  val.max_val_iters=1 \
  val.save_visuals=false \
  train.log.log_every=1 \
  train.log.run_comment=num8_io_smoke
```

启动日志应显示 `spatial_modality_gate=True`，`cross_modal_adapter=False`、`local_window_refiner=False`、`consensus_calibration=False`，以及 prepared loader、`non_blocking=True`、`shared_batch_conversion=True`。结束时应有第 5 步训练、step=5 的验证、checkpoint 保存和 `Training finished.`；确认 loss 与验证指标有限，无 CUDA OOM 或 traceback。

训练入口还会在结束或训练异常后输出 `[CUDA memory] peak allocated=... GiB, peak reserved=... GiB`，统计模型初始化后的训练、验证及保存阶段。该数字是当前进程的 PyTorch 显存峰值；CUDA context 和其他进程的占用仍需结合 `nvidia-smi` 判断。24GB 是否足够以实际测试为准。如果 OOM，保留完整报错及显存输出，再调整独立实验配置的分辨率或模型规模；改变分辨率后需采用相同设置做对照实验。

可在另一个 SSH 终端监控整张 GPU：

```bash
watch -n 1 nvidia-smi
```

日志、解析后的配置、TensorBoard 和 checkpoint 位于该次运行的 `runs/gated_modality_rain_io/日期/时间_num8_io_smoke/` 内。检查保存空间：

```bash
du -sh runs/gated_modality_rain_io
```

```bash
df -h ~
```

本配置关闭 checkpoint 自动清理，每 10000 步保存一次，结束时也保存。正式训练 100000 步会有 10 个定期 checkpoint；相同末步的最终保存使用同一路径，EMA 另存一份。应根据本次完整 checkpoint 的实际大小，为全部 checkpoint、EMA 和日志预留磁盘空间。不要批量删除已有实验或数据。

## 正式训练

确认上述测试、真实验证和磁盘预算通过后，在同一 tmux 窗口中启动新的正式 run：

```bash
python -m src.trainer.gated_modality_rain_io_trainer \
  train.log.run_comment=num8_gate_io_train
```

默认预算为 100000 步，每 2000 步验证，验证上限 60 个 batch。正式 run 仍从头训练，不加载 5 步测试的权重；日志和 checkpoint 与测试 run 分开。

运行中按 `Ctrl+B`，松开后按 `D`，可离开 tmux，再关闭本机 SSH 或电脑。重新登录后执行：

```bash
tmux attach -t gate-io
```

## 结果解释

本改动验证数值行为和减少重复输入转换；实际吞吐变化需要在 num8 上测量，不能仅凭配置宣称提速。固定起报 rollout 不注入真实未来雷达或卫星，但现有模型的 3D decoder 仍有已记录的时间因果性限制，见 [原验证记录](../superpowers/validation/2026-09-03-gated-modality-rain-trainer.md)。本次 I/O 接入不改变该架构；训练中的 teacher-forcing loss 不能直接当作严格因果预测性能，实验比较应统一预测协议。

## 本地验证记录

2026-10-13，在独立 Windows 测试环境中使用 Python 3.13.5 与 PyTorch 2.12.0 CPU 验证：新增测试 8 项通过，1 项 CUDA 测试跳过；真实小尺寸门控模型的 loss、梯度和一次参数更新与原独立 trainer 完全一致，比较容差为零。主损失与 rollout 对原始 float64 模型输入的转换次数从每字段 2 次降为 1 次；已经准备好的 float32 输入不产生额外转换。DataLoaderShard 和 IterableDataset 对应的 DataLoaderDispatcher、GAN 训练循环、纯门控及全部创新模块同时启用的训练路径也完成小尺寸检查。新 Python 文件通过 Python 3.12 语法检查和 Ruff 检查，同学快照的 367 个文件 SHA256 一致。

既有门控、trainer 和独立框架回归检查有 60 项通过、6 项跳过，另有 `test_original_baseline_sources_are_unchanged` 失败。该测试固定了更早版本的源码指纹，当前 baseline 指纹与本任务开始前的 `c25ec88` 学校服务器提交一致；本次不改动该源码或旧测试。全仓 `src/tests` 检查在收集阶段因本地缺少 `pyproj`、`duckdb`、`diffusers` 出现 5 项错误，尚未完成全仓回归。

上述结果不表示新服务器的 CUDA、真实数据或正式训练已通过；需执行前述服务器检查与 5 步验证。

用户补充 24GB 显存约束后，配置中的 BF16 与 activation checkpointing 改为显式设置，数值仍与原继承值一致；配置契约测试通过，入口的 CPU 训练和最终 checkpoint 保存检查通过。新增 CUDA 峰值输出需在 num8 上确认。
