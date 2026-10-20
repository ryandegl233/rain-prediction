# 旧模型四川降雨案例：4 帧历史，滚动预测 12 帧

本工具为独立入口 `src/tools/gated_sichuan_case.py`。不修改训练框架、模型、decoder 或原绘图脚本，复用 `GatedModalityRainTrainer` 的逐帧、增长历史滚动方法及雨量残差还原。只实例化模型和一个案例的数据，不创建优化器或启动训练。

## 权重选择依据

用户上传的 `log.log` 截至 82000 步，全日志最高 CSI@0.5 为 12000 步的 0.310673。但该次训练每 5000 步保存，12000 步没有保存记录。已保存且有对应验证记录的 checkpoint 中，80000 步最高，为 0.287054；用户已确认 `checkpoint-00080000/model.safetensors` 存在。

这不代表每个 checkpoint 都在相同样本上重评后的全局最优：旧验证会抽样，并且 5000、15000 等步数没有对应验证指标。这里使用已知记录可支持的候选。旧验证直接使用 `self.model`，故加载普通权重；不使用 run 根目录中可能已被后续训练覆盖的 EMA。

## 时间与数据

- 4 帧历史：2025-08-10 21:30、22:00、22:30、23:00。
- 12 帧预测：2025-08-10 23:30 至 2025-08-11 05:00，间隔 30 分钟，共 6 小时。
- 读取训练保存的完整 `config_total.yaml`，保留原预处理、门控、decoder。
- 按 metadata 和 dataset window 两次核对时间；缺帧、重复或没有连续窗口时停止，不擅自换案例或插值。
- 模型输入只包含历史四帧及之后自己的预测。未来真值仅用于绘图对照。
- 真实标签沿用 `rain_interpolated`。原始预测保存为 NPZ，不做真值强度匹配、额外高斯平滑或 0.99 上限截断。
- 统一使用原项目色阶，最近邻显示；地理范围沿用预处理默认 `(97.3,108.4,26.1,34.25)`，北在上。范围并非从每个 TIFF 的地理元数据独立复核。
- 图上为数据标称北京时间。仓库雨量预处理查询 `[t,t+interval]`，因此这是一项历史回放，不应直接宣称所有输入在标称起报时刻都已实时可得。
- 第 5–12 个预测时效超过训练的四帧未来范围，属于长时滚动外推，须与真值一起查看。

## num8 执行

进入仓库并同步。若 fast-forward 被本地改动阻止，保留改动并检查，不执行 reset。

```bash
cd /home/yanjieyu/RainPrediction

git pull --ff-only origin project

conda activate rainpred
```

可选创建 tmux 会话：

```bash
tmux new -s sichuan-case
```

在 tmux 内再次激活环境，然后执行：

```bash
conda activate rainpred

cd /home/yanjieyu/RainPrediction

RUN=/home/yanjieyu/RainPrediction/runs/gated_modality_rain_io/2026-10-13/20-45-28_num8_spatial_gate_bs2

OUT="$HOME/RainPrediction/vis_show/sichuan_20250810_step80000_$(date +%Y%m%d_%H%M%S)"

python -m src.tools.gated_sichuan_case \
  --run-dir "$RUN" \
  --step 80000 \
  --first-future "2025-08-10 23:30:00" \
  --future-frames 12 \
  --output-dir "$OUT"

echo "$OUT"
```

同一时间只运行这一组推理，避免与训练争抢 GPU 显存。脱离 tmux：先按 Ctrl+B，松开后按 D；重新进入：`tmux attach -t sichuan-case`。

默认从原绘图程序的阿里云 DataV 地址读取四川市级边界，并把 GeoJSON 缓存到输出目录。若服务器无法连接该地址，可以用 `--boundary-source /path/to/510000_full.json` 指定同一份本地边界文件。

## 输出

- `comparison_grid.png`：12 帧预测与真值对照，按每四个时刻一组排列，上一行预测、下一行真值。
- `prediction_grid.png`：12 张预测总图。
- `history_grid.png`：4 张历史雨量图。
- `prediction/`、`truth/`：逐帧 PNG，文件名含时效与时间。
- `comparison.gif`：左预测、右真值的动画。
- `forecast_raw.npz`：未修饰的预测、雨量历史和真值、时间。
- `case.json`：配置和权重 SHA256、代码提交、时间范围、绘图口径等。

这些图片是特定日期的案例展示；若该月参与训练，应标为训练期案例，不能当作独立测试集结果。具体归属以原 run 保存的 train/val 输入目录为准。

## 本地验证范围

测试覆盖跨日时间、连续窗口匹配、12 步预测回灌、残差累计、模型最大长度检查、原始数组不被绘图修改，以及小尺寸真实 GatedModalityRainModel 的 12 步运行与前 4 步一致性。真实数据和 80000 步权重位于 num8；最终 GPU 推理及实际案例图片由上面的服务器命令生成。
