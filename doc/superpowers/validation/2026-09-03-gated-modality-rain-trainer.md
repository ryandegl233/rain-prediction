# gated_modality_rain_trainer：验证与服务器检查

## 🎯 版本边界

这是用户批准的第一版结构改动：雷达与卫星独立空间门控，训练逻辑复用 `RainTSNextFrameTrainer`。不新增 loss、空间 mask 或干预训练；不自动启动训练，也不据此宣称 CSI 提升。结构见[批准设计](../specs/2026-09-03-spatial-modality-gate-design.md)。

部署目标是 CUDA GPU，不是 CPU 专用实现。门控是普通 PyTorch 模块，运算随模型及输入的 device／dtype；不在 forward 中创建 CPU 常量张量或强行转移设备。沿用 Accelerate 和 bf16 配置；CUDA／bf16 测试提供前向、反向和恒等初始化检查。GPU 测试通过也不等同于真实数据、448×448 输入或多卡训练已经验证。

## 🧪 本机验证环境

Windows，Python 3.13.5，PyTorch 2.12.0+cpu；`torch.cuda.is_available() == False`。

测试环境版本：pytest 8.3.4、Hydra 1.3.6、Accelerate 1.14.0、einops 0.8.2、timm 1.0.29、transformers 5.16.1、safetensors 0.8.0。这是本次验证记录，不要求改动服务器现有环境版本。

在项目内独立、Git 忽略的 venv 补齐测试依赖，复用本机已有 CPU PyTorch，没有修改全局 Python 包。旧 `.pytest_cache` 无写入权限，测试用 `-p no:cacheprovider` 避开，不删除或改变原缓存。

改动前，原模型和 trainer 的 43 项测试通过。控制器在实现提交 `527d990` 上独立重跑新旧模型和 trainer：**98 passed、2 skipped**，75.57 秒，退出码 0。2 项跳过都是 CUDA 专项测试，原因为本机无 GPU；不计为 GPU 通过。21 条第三方 `torch.jit.script`／`script_method` 弃用警告在生产代码修改前已存在。

新增 Python 文件通过 Ruff 检查，全部本轮 Python 文件通过语法检查；未运行 ty。CPU 上另做启用 activation checkpoint 的 bf16 autocast 前向／反向检查，输出 dtype 为 bfloat16、输出和梯度均有限；该检查也不替代 CUDA 验证。

| 验证内容 | 本轮结果 |
| --- | --- |
| 初始编码特征、三模态完整预测、AR 与 mask token | 与同权重 baseline 精确一致 |
| 非单位门控、bias／降水直接贡献、逐帧空间门控、梯度 | 通过 |
| baseline／门控权重初始化与损坏权重拒绝 | 通过，拒绝前不部分写入模型 |
| Hydra 两份配置及脚本／模块入口 | 通过 |
| train／val 配置驱动的四帧固定起报 | 改变未来标签，预测逐元素不变 |
| CUDA FP32／bf16（含 activation checkpoint） | 已编写测试；本机未执行 |
| 多卡 DDP、真实数据、448×448 吞吐／显存、预测指标 | 未验证 |

stem384／hidden32 时门控实测新增 **37,282** 个参数。分模态贡献卷积与特征保存会增加计算及显存，参数量小不表示没有运行开销。

## 🖥️ 服务器检查命令

在服务器已有项目环境中运行，先确认使用 CUDA 版 PyTorch 且硬件支持当前配置的 bf16：

```bash
cd ~/RainPrediction
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(), 'CUDA is unavailable'; print(torch.cuda.get_device_name(0)); assert torch.cuda.is_bf16_supported(), 'Current bf16 configuration requires supported hardware'"
```

然后运行小尺寸 CUDA 专项测试，确认结果确实通过而非全部跳过：

```bash
python -m pytest src/tests/time_series/test_spatial_modality_gate.py -k cuda -q -rs -p no:cacheprovider
```

再运行相关完整回归和新配置检查。下面的 `--cfg job` 只输出配置，不开始训练：

```bash
python -m pytest src/tests/time_series/test_spatial_modality_gate.py src/tests/time_series/test_causal_patch_transformer_next_frame.py src/tests/trainer/test_gated_modality_rain_trainer.py src/tests/trainer/test_rain_trainer_ts_next_frame.py -q -rs -p no:cacheprovider
python -m src.trainer.gated_modality_rain_trainer --cfg job
python -m src.trainer.gated_modality_rain_trainer --config-name rain_trainer_ts_next_frame_fixed_origin --cfg job
```

两份配置分别是门控实验和固定起报 baseline。两者均不注入真实未来雷达、卫星；数据、loss、优化器、训练预算相同，输出目录分别隔离。旧 `rain_trainer_ts_next_frame.yaml` 保持原样，不应用它原先开启 GT forcing 的指标直接与新协议比较。

旧 baseline 权重只能作为 `train.init_model_path` 初始化新架构；`train.resume_path` 保持空，不能拿旧 optimizer 状态续训。需要先由用户确定实际服务器 checkpoint；本版本不内置个人 checkpoint 路径。

受检初始化支持已有 Accelerate 的完整 bin／safetensors 文件、目录及分片索引。先检查完整键集合与形状，再执行加载，因此会多读一次权重；不重命名参数，也不适配任意嵌套 checkpoint 字典。

## ⚠️ 原 baseline 的时间信息泄漏

本轮独立诊断发现，原 3D decoder 不是严格时间因果的。这是门控关闭时就存在的问题，不能因配置中出现 `causal=true` 就认为通过验证。

测试设置：随机种子 2025，小尺寸 baseline（32×32 输入，patch4、stem16、dim32、depth1），保留原 decoder 的 `wan_factorized`、`reflect`、kernel7、causal 配置。使用 1 帧 context 和 7 帧右移 target seed，只替换最后 2 帧 seed，比较前 4 帧结果。

| 观察位置 | 前 4 帧最大绝对差 |
| --- | ---: |
| decoder 输入（Transformer 后） | 0 |
| decoder 第一个卷积后 | 0.5688492 |
| decoder 第一个 GroupNorm 后 | 2.7221808 |
| 雷达输出 | 0.5774781 |
| 卫星输出 | 0.6653512 |
| 降水输出 | 0.6473166 |

这些是随机输入下的数值差异，不是降水物理量或模型指标。其用途是定位依赖关系，不代表真实样本的影响幅度。

两个机制都在原 [reconstruction.py](../../../src/networks/modules/reconstruction.py) 中：`SpatialPadConv3d` 用 reflect 填充左侧时间边界，会把后续时间位置反射到早期；3D `GroupNorm` 的统计量覆盖时间维。在独立保持 GroupNorm 输入前缀相同、只改变后两帧的检查中，其输出前缀仍变化（最大绝对差 0.1081345）。

可用以下小尺寸检查重现端到端现象；默认使用可用 GPU，否则 CPU，因此 GPU 上具体差值可以与本机记录不同：

```bash
python - <<'PY'
import torch
from src.networks.time_series.causal_patch_transformer_next_frame import RainCausalPatchTransformerNextFrame

torch.manual_seed(2025)
torch.set_num_threads(2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RainCausalPatchTransformerNextFrame(
    input_size=32, patch_size=4, stem_channels=16, dim=32, depth=1,
    num_heads=4, dropout=0.0, drop_path=0.0, max_frames=16,
    decoder_base_channels=16, stem_pad_mode='reflect',
    decoder_pad_mode='reflect', decoder_k_size=7, decoder_causal=True,
    encoder_conv_style='wan_factorized',
).to(device).eval()
context = torch.randn(1, 12, 1, 32, 32).to(device)
seed = torch.randn(1, 12, 7, 32, 32).to(device)
changed = seed.clone()
changed[:, :, -2:] = torch.randn(1, 12, 2, 32, 32).to(device) * 3.0
features: list[torch.Tensor] = []

def capture(module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
    features.append(args[0].detach().clone())

handle = model.rain_decoder.register_forward_pre_hook(capture)
with torch.no_grad():
    original = model.forward_ar(context, seed, strict_target_isolation=False)
    perturbed = model.forward_ar(context, changed, strict_target_isolation=False)
handle.remove()
print('decoder input prefix difference:', (features[0][:, :, :4] - features[1][:, :, :4]).abs().max().item())
print('rain output prefix difference:', (original['rain'][:, :, :4] - perturbed['rain'][:, :, :4]).abs().max().item())
PY
```

本次不修改 decoder。门控自身逐帧、固定起报 rollout 不注入未来真值，与原 decoder 的序列训练泄漏是不同检查；前两项通过并不能消除后一项。正式实验前需要单独确定 decoder 修复范围并重跑对照。

## 📋 实现取舍记录

| 取舍 | 原因 | 如需改变的成本 |
| --- | --- | --- |
| 在现有 `D:/rain-prediction` 工作区实现，不创建额外 worktree | 用户批准在当前项目修改；保留无关改动，只提交明确文件 | 若希望隔离，可把本轮提交转到独立分支 |
| 一名实现者负责模型、配置、加载的完整单元；控制器独立诊断与验证 | 减少共享文件冲突，并保持接口一致 | 范围过大时由独立审查拆出针对性修复 |
| 沿用 `train.log.run_comment`，同时隔离 Hydra 和 Accelerate 输出目录 | 与项目现有配置、checkpoint 保存路径一致 | 输出路径要求改变时只需调整继承配置 |
| 加载检查放在 `src/utils/gated_checkpoint.py` | 明确只服务门控初始化，避免引入不必要的包导入副作用 | 后续扩大初始化功能时重新评估模块归属 |

遵守禁止批量删除要求，保留独立测试环境和诊断记录；不做自动清理，不改用户 `.gitignore`，不推送代码。
