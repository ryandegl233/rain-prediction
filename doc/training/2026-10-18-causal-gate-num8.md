# num8：修复因果性与递推监督后的空间门控实验

独立配置为 `src/config/ts_rain_train/gated_modality_rain_causal.yaml`。复用已有 `GatedModalityRainIOTrainer` 和 `GatedModalityRainModel`，通过已有配置接口选择逐帧 2D decoder；没有改写原模型、trainer、共享 decoder、同学副本或旧实验配置。

## 修复内容与实验边界

1. `decoder_type: 2d`：每帧分别进行空间卷积与归一化，消除 3D decoder 的时间 reflect 和跨时间 GroupNorm 依赖。时序 Transformer 与空间门控继续处理历史信息；模型仍使用时序输入。
2. `sequence_loss.enabled: false`：主分支用 4 帧历史预测下一帧，与自由递推第一步使用相同的 context/seed 结构。后续 4 帧未来仍由 rollout 分支监督，并非只训练第一帧、忽略其余未来。
3. `rollout_branch.loss_on: all`：雷达、卫星、降雨 decoder 都获得递推分支的直接监督。沿用原全模态权重 radar=1、satellite=1、rain=2；因此相对于原 rain-only 分支，其降雨权重也从内部覆盖值 1 恢复为 2。rollout 总权重仍为 0.2，保留 detach_history=true 控制显存。
4. 验证使用固定顺序、无增强、不打乱、batch1、不丢弃尾批，最多 415 批；与当前日志中的 415 个筛选后窗口对应，覆盖 202508/202509 当前验证集合。以后更换数据或 val batch size 时需重新核对该数量。

保持 30 分钟数据、448×448、4 帧历史和 4 帧未来、纯空间门控、BF16、activation checkpoint、rain residual 和既有损失参数。没有顺带启用 GAN、CCC、额外雨区加权或新的纹理损失。`mode=mse` 下原 rain_region_weight 仍不生效，本轮不将它计作已启用的创新点。

这是新的训练协议与 decoder，必须从头训练；不要使用旧 3D 模型 checkpoint 作为 resume 或强行加载为新模型初始化。新配置的 resume_path/init_model_path 均为空。旧配置的缺陷不因增加新配置而自动消失，启动时必须带 `--config-name gated_modality_rain_causal`。

默认训练 batch2、验证 batch1，每 2000 步验证并输出可视化，每 5000 步保存 checkpoint，预算 100000 步。checkpoint 不自动删除。新权重大小需重新实测，不能沿用旧 460M 的数值。

## 1. 结束旧实验并更新代码

先确认旧实验最近一次 checkpoint 已保存；在旧训练终端按 `Ctrl+C` 结束旧程序。未保存的步数不会因此自动保存。不要同时在同一张 4090 上运行新旧训练，也不要停止其他用户的进程。

如在旧 tmux 中，按 `Ctrl+B`，松开，再按 `D` 离开旧会话。从普通 SSH 终端执行：

```bash
cd /home/yanjieyu/RainPrediction
```

```bash
git status --short --branch
```

```bash
git -c merge.autostash=false pull --ff-only --no-rebase origin project
```

本次仅新增文件，不覆盖你已修改的旧 `gated_modality_rain_io.yaml`。如果 Git 报本地文件冲突，保留报错和本地改动，不执行 reset、clean 或强制覆盖。

```bash
git log -1 --oneline
```

```bash
tmux new-session -A -s gate-causal
```

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
```

```bash
conda activate rainpred
```

```bash
cd /home/yanjieyu/RainPrediction
```

```bash
nvidia-smi
```

```bash
df -h ~
```

## 2. 配置与回归检查

```bash
python -m src.trainer.gated_modality_rain_io_trainer --config-name gated_modality_rain_causal --cfg job --resolve
```

确认 decoder_type=2d、sequence_loss.enabled=false、rollout_branch.loss_on=all、两个 use_gt_future_modalities 均为 false、train batch2、val batch1、max_val_iters=415、resume_path/init_model_path=null。

```bash
python -m pytest src/tests/gated_modality_rain/test_causal_training.py -q -p no:cacheprovider
```

本机无 GPU 的结果不能代替该服务器检查。CUDA 可用时，该文件包含的小尺寸 BF16/activation-checkpoint 训练、验证和保存测试应执行而非跳过。若缺少 pytest，请先反馈缺失依赖；不要为测试整体升级迁移过来的 PyTorch/CUDA 环境。

## 3. 真实数据：batch2，20 步训练与两次验证

```bash
python -m src.trainer.gated_modality_rain_io_trainer \
  --config-name gated_modality_rain_causal \
  train.max_steps=20 \
  train.scheduler.num_training_steps=100000 \
  val.val_duration=10 \
  val.max_val_iters=2 \
  train.log.log_every=1 \
  train.log.run_comment=num8_causal_gate_smoke_bs2
```

`num_training_steps=100000` 保留正式训练的学习率进度，避免 20 步冒烟把整个衰减过程跑完。此处的验证数量只为检查执行路径，不能用它估计正式指标。

通过条件：训练到 20 步、10/20 步验证完成、checkpoint 保存、`Training finished.`，所有 loss 有限，无 OOM/traceback。应看到 spatial_modality_gate=True、prepared loader，以及 meta/sequence_loss=0、meta/context_frames=4、meta/target_frames=1；rollout 分支仍启用并预测未来 4 帧。观察最后的 CUDA peak allocated/reserved 与整卡 nvidia-smi 占用。

若真实数据 batch2 OOM，使用同一命令加 `dataset.train.batch_size=1`，并改为新的 run_comment；正式命令也必须使用同样的 batch1。不删除旧输出，不以降低分辨率或关闭验证掩盖显存问题。

## 4. 正式启动

仅在上述真实数据检查通过后执行。启动新 run，从头训练，不接续 20 步的冒烟 checkpoint。

```bash
python -m src.trainer.gated_modality_rain_io_trainer \
  --config-name gated_modality_rain_causal \
  train.log.run_comment=num8_causal_gate_bs2
```

配置已包含 batch2、100000 步、每 5000 步保存一次，不需要再编辑旧 YAML。运行目录为：

```text
/home/yanjieyu/RainPrediction/runs/gated_modality_rain_causal/日期/时间_num8_causal_gate_bs2/
```

解析后的实际配置保存在 `config/config_total.yaml`，日志为 `log.log`，拼图在 `val_viz/step_XXXXXXXX/rollout_sample0_timeline.jpg`。固定验证顺序使每次选中的首个样本保持一致。保存结果后可测量本次 checkpoint 大小并据此预算 20 次 checkpoint 的磁盘开销。

运行后按 `Ctrl+B`，松开，再按 `D`，可关闭 SSH；重新进入：

```bash
tmux attach -t gate-causal
```

## 结果判断

本轮修复因果性、训练/推理输入结构、递推梯度覆盖和验证可比性，不保证某个步数的 CSI 必然提高，也不保证卫星纹理立即清晰。应比较本轮固定集合上 2000/4000/6000/10000 步的趋势，避免再次只靠持续增加步数。

旧 run 每轮随机抽 60 个窗口，新 run 使用当前全部 415 个固定窗口，不能把两轮日志的 CSI 直接作为门控提升量。评价改进需要在同一验证集合上重评已有权重，并加入持久性基线。门控消融可在同一新配置下额外使用 `rain_prediction_model.spatial_modality_gate_enabled=false`，作为另一个独立实验；不要在一张卡上同时启动。

测试命令、覆盖范围和全仓检查限制见 [验证记录](2026-10-18-causal-gate-test-results.md)。
