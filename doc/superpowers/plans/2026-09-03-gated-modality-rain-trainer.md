# gated_modality_rain_trainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现用户批准的第一版雷达／卫星空间门控，提供 `gated_modality_rain_trainer` 入口和可公平对比的固定起报配置。

**Architecture:** 原 patch embedding 保留，复用权重切片计算三模态贡献，以逐帧空间门控修正雷达和卫星贡献。统一接入 `_encode_tokens`，保留 ConvStem、Transformer、decoder、trainer 和现有返回值。门控模型加载旧权重时严格核对键，optimizer 不做旧架构续训。

**Tech Stack:** Python >=3.12、PyTorch、pytest、Hydra/OmegaConf、Accelerate；不引入新生产依赖。

**Spec:** `doc/superpowers/specs/2026-09-03-spatial-modality-gate-design.md`

## Global Constraints

- 第一版名称为 `gated_modality_rain_trainer`；不接入新 loss 或新增 mask；不启动训练。
- 保留原 `patch_embed.weight` 和 `patch_embed.bias` 的名称、形状与参数对象；不复制成三个独立可训练卷积核，也不执行 detach。
- 门控只读取模型当前输入中的模态特征，不增加未来真值或标签输入。
- 保留现有模型返回值：三模态字典或降水 tensor。不新增返回键、trainer 依赖或持久化的中间 gate 缓存。
- 关闭门控默认不增加参数；开启仅支持 `encoder_type=patch`、`frame_patch_size=1`。
- `train.next_pred.rollout_branch.use_gt_future_modalities=false`、`val.rollout_use_gt_future_modalities=false`。
- 两份新配置除门控和输出目录外，保持相同的数据、loss、优化器和训练预算；不启用 cross-modal adapter 或 local-window refiner。
- 不删除数据、不批量删除文件、不修改现存 `.gitignore` 改动、不自动推送 GitHub。
- 新 Python 代码使用现代类型标注和英文注释；测试放在 `src/tests/`；不运行 ty。

## 📂 文件职责

| 文件 | 职责 |
| --- | --- |
| `src/networks/time_series/causal_patch_transformer_next_frame.py` | 私有空间门控模块、开关和编码接入 |
| `src/trainer/gated_modality_rain_trainer.py` | 同名 Hydra 入口，复用现有 trainer |
| `src/trainer/rain_trainer_ts_next_frame.py` | 只在模型初始化加载处调用受检加载；不改 loss／rollout 算法 |
| `src/utils/checkpoint.py` | 如确需独立模块：门控权重初始化核验，不做格式兼容框架 |
| `src/config/ts_rain_train/rain_trainer_ts_next_frame_fixed_origin.yaml` | 公共固定起报对照配置 |
| `src/config/ts_rain_train/gated_modality_rain_trainer.yaml` | 继承对照配置，只开门控和改输出目录 |
| `src/tests/time_series/test_spatial_modality_gate.py` | 小尺寸真实模型结构、等价、梯度、加载测试 |
| `src/tests/trainer/test_gated_modality_rain_trainer.py` | 配置、入口、初始化和固定起报集成测试 |

## 🛠️ 实现与验收

### Task 1: 可测试的门控版本与固定起报入口

**Files:** 创建／修改上表文件；原 baseline YAML 保持不变；仅按实际需要创建 checkpoint 工具。

**Interfaces:**

- 消费现有 `RainCausalPatchTransformerNextFrame`、`RainTSNextFrameTrainer` 与 baseline Hydra 配置。
- 新构造参数 `spatial_modality_gate_enabled: bool = False`、`spatial_modality_gate_hidden_channels: int = 32`。
- 新模型属性 `spatial_modality_gate` 为可选私有门控模块，全部新增参数名以 `spatial_modality_gate.` 开头。
- 门控模块 `forward(z_radar: torch.Tensor, z_satellite: torch.Tensor, z_rain: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]`，每个输出形状 `[B,1,T,H,W]`。
- `src/trainer/gated_modality_rain_trainer.py` 默认配置名 `gated_modality_rain_trainer`；直接脚本运行与 `python -m` 均能解析配置。

- [ ] **Step 1: 建立小尺寸模型测试并验证 RED。**

独立测试文件中的工厂使用真实模型，`input_size=16, patch_size=4, stem_channels=16, dim=32, depth=1, num_heads=4, decoder_base_channels=16, dropout=0, drop_path=0, max_frames=8`。输入使用有限随机 `[1,12,4,16,16]`。先加以下测试，运行后必须因新增参数尚不存在而失败：

```python
def test_neutral_gate_preserves_complete_predictions() -> None:
    baseline = build_model().eval()
    gated = build_model(spatial_modality_gate_enabled=True).eval()
    result = gated.load_state_dict(baseline.state_dict(), strict=False)
    assert not result.unexpected_keys
    assert result.missing_keys
    assert all(key.startswith("spatial_modality_gate.") for key in result.missing_keys)
    x = torch.randn(1, 12, 4, 16, 16)
    with torch.no_grad():
        original = baseline(x, return_modality_dict=True)
        updated = gated(x, return_modality_dict=True)
    for name in ("radar", "satellite", "rain"):
        torch.testing.assert_close(updated[name], original[name], rtol=0, atol=0)
```

命令：测试环境 Python `-m pytest src/tests/time_series/test_spatial_modality_gate.py -q`。

- [ ] **Step 2: 实现最小门控与编码接入，转为 GREEN。**

在现有模型文件添加私有模块，按以下核心操作实现，模块成员名由语义确定，不添加额外状态缓存或测试专用方法：

```python
self.net = nn.Sequential(
    nn.Conv2d(3 * channels, hidden_channels, 1),
    nn.SiLU(),
    nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, groups=hidden_channels),
    nn.SiLU(),
    nn.Conv2d(hidden_channels, 2, 1),
)
nn.init.zeros_(self.net[-1].weight)
nn.init.zeros_(self.net[-1].bias)
```

forward 拼接三贡献，将 `b d t h w` 排列为 `(b t) d h w`，预测后还原 `b 2 t h w`，返回两个 `1 + tanh(logit)`。模块放在原模块初始化之后以保留 baseline 参数初始化的随机数顺序。

`_encode_tokens` patch 分支先执行原 `encoded = self.patch_embed(x)`。开门控时，用 `F.conv3d` 和原权重的三份通道切片计算不带 bias 的贡献；stride、padding、dilation 取原卷积属性。随后：

```python
gate_radar, gate_satellite = self.spatial_modality_gate(z_radar, z_satellite, z_rain)
encoded = encoded + (gate_radar - 1) * z_radar + (gate_satellite - 1) * z_satellite
```

新增参数正值、模态通道正值及总和、encoder 和时间 patch 验证应在开启时尽早执行并给出明确 `ValueError`。关门控走原数值路径，不破坏已有 resnet 模式。

- [ ] **Step 3: 补齐门控行为测试，各测试先观察失败或确认其对应实现已由 Step 1 驱动。**

断言中使用手工构造值，避免调用待测 helper 计算期待结果。以 ConvStem 的前向 pre-hook 观察融合输入：将权重设置为已知常数，最后 gate head 设置常数 logit `atanh(0.5)` 和 `atanh(-0.5)`，预期融合为 `1.5*z_radar + 0.5*z_satellite + z_rain + bias`（允许非中性拆分求和的 FP32 误差）。改变仅降水输入时验证直接贡献系数为 1；旧 bias 只加一次。

覆盖非单位门控的空间差异和时间独立性：只扰动最后帧贡献，前几帧两个 gate 必须精确不变。比较初始 `_encode_tokens` 和完整预测；检查关门控的严格旧 state dict 加载；检查 `forward_ar` 和 mask token 与 baseline 等价。

梯度测试：门控 head 初始梯度和原 patch 权重梯度有限且非零；用 optimizer 只更新 head 一步，再反传，早期 gate 层梯度应有限且非零。新增模型 state dict 经 `torch.save`/`torch.load(weights_only=True)` 后严格重载、预测一致。非法 encoder、时间 patch、hidden 和模态通道配置均有参数化测试。

命令：测试环境 Python `-m pytest src/tests/time_series/test_spatial_modality_gate.py src/tests/time_series/test_causal_patch_transformer_next_frame.py -q`。

- [ ] **Step 4: 配置与权重加载测试先 RED，然后实现同名入口与受检初始化。**

用 Hydra compose 真实加载两份新配置，解析后比较 dataset、train、val、optim 设置；除门控和输出设置外不得有差异。两份均满足固定起报 false 开关，`resume_path` 为空；基配置的其他 loss 原样保留。

固定起报对照配置：

```yaml
defaults:
  - rain_trainer_ts_next_frame
  - _self_
hydra:
  run:
    dir: ./runs/next_frame_fixed_origin/${now:%Y-%m-%d}/${now:%H-%M-%S}_${comment}
comment: next_frame_fixed_origin
train:
  resume_path: null
  next_pred:
    rollout_branch:
      use_gt_future_modalities: false
val:
  rollout_use_gt_future_modalities: false
```

门控配置：

```yaml
defaults:
  - rain_trainer_ts_next_frame_fixed_origin
  - _self_
hydra:
  run:
    dir: ./runs/gated_modality_rain_trainer/${now:%Y-%m-%d}/${now:%H-%M-%S}_${comment}
comment: gated_modality_rain_trainer
rain_prediction_model:
  spatial_modality_gate_enabled: true
  spatial_modality_gate_hidden_channels: 32
```

直接脚本参考原 trainer 的项目路径注入，但入口只 instantiate `RainTSNextFrameTrainer(cfg)` 并 `run()`；不复制训练类，不添加空子类或 alias。

权重初始化优先复用现有 Accelerate 加载：先研究实际调用与格式，使用其受支持的文件／目录选择和 state dict 读取，在加载前比较所有 expected/actual keys、shape。仅门控模型接受完整旧 baseline 或完整 gated 权重，允许缺失集合为完整门控参数集合或空；部分损坏的门控 state 必须拒绝。非门控模型保持原初始化行为。新架构初始化仍使用 `train.init_model_path`，不自动提供路径；`resume_path` 留空。测试要使用真实临时保存文件，包含完整旧权重、新权重、旧键缺失、意外键、错误 shape、部分门控键，不只验证 mock 调用。

命令：测试环境 Python `-m pytest src/tests/trainer/test_gated_modality_rain_trainer.py -q`，初次因缺新配置／入口或未拒绝不完整权重而 RED，实现后 GREEN。

- [ ] **Step 5: 固定起报与原接口集成验证。**

使用现有 trainer 测试的轻量构造方式，不创建真实 dataset/训练任务。用真实小尺寸 gated 模型执行 `_rollout_predict`（以现有函数签名为准）；同样历史输入配两份不同未来雷达／卫星／降水标签，在 `use_gt_future_modalities=False` 下预测逐元素相同。用 train rollout 与 val 配置驱动开关，不仅检查 YAML 文本。

实际入口只执行配置输出，不触发训练：

```powershell
python src/trainer/gated_modality_rain_trainer.py --cfg job
python -m src.trainer.gated_modality_rain_trainer --cfg job
```

运行原 trainer 与模型的相关完整回归；CUDA 可用时增加 bf16 前向及反传，硬件不可用明确记录未验证。记录额外参数量 `sum(p.numel() for p in model.spatial_modality_gate.parameters())`，stem384/hidden32应为37,282，不推断吞吐或显存。

- [ ] **Step 6: 自审、提交和报告。**

运行 `git diff --check`、新增代码语法检查与上述 pytest；只 stage 本任务文件，不包含 `.gitignore`。提交信息 `feat: add gated modality rain trainer v1`。实现报告列出 RED/GREEN 命令和结果、文件、checkpoint 格式选择、边界与未验证项，不声称预测指标提升。

## 🔬 控制器独立诊断与完成检查

控制器准备隔离依赖环境，运行改动前相关测试，并独立做 baseline 全链路前缀扰动诊断；该诊断不写入失败的门控单元测试，不修补原 decoder。结果如有非因果性，应带具体数值向用户报告，明确与门控本身的时间独立性区分。

Task 1 完成后分别做任务级 spec／质量 review 与最终整体验收，控制器重跑最终测试。无数据、无 CUDA 时不冒充完成服务器验证，不推送、不启动实验。
