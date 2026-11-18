# Earthformer 与 AlphaPre 适配说明

本实现位于 `src/baselines_stage3/deep/earth_alpha.py`。模型实现以私有正常 Python 包形式保存于相邻的 `earth_alpha_port/`，运行时不依赖仓库外路径，不创建伪造模块、不替换上游对象、不修改原始 baseline。

## 来源与结构

| 模型 | 官方代码 | 固定提交 | 本地保留内容 |
| --- | --- | --- | --- |
| Earthformer | https://github.com/amazon-science/earth-forecasting-transformer | `7732b03bdb366110563516c3502315deab4c2026` | 官方 `CuboidTransformerModel`、cuboid self/cross attention、patterns、utils、registry，以及 Apache-2.0 LICENSE/NOTICE |
| AlphaPre | https://github.com/linkenghong/AlphaPre | `64d5195984c72db2692266acc800efa34ff8a4bf` | 官方 `models/alphapre.py` 的完整模型文件 |

`earth_alpha_port/SOURCE_MANIFEST.json` 逐文件记录上游路径、提交和 SHA-256。全部 10 个上游文件均按字节复制，没有更改代码或导入。AlphaPre 该固定提交没有独立许可证文件；清单保留其来源，不为上游代码补写授权声明。

运行依赖仅为 `torch`、`numpy`、`einops`。测试另外需要 `pytest`。私有包的上游代码维持原始类型标注与格式，新的适配层使用现代 Python 类型标注。

## 统一接口

```python
core = build_earth_alpha("earthformer", options)
prediction, auxiliary = core(history)
```

输入为 `[B, 12, 7, H, W]`，输出为 `[B, 12, 1, H, W]`。`forward` 不接受标签。历史不足七帧时的左侧零填充、模态缺失标记、目标雨量残差锚点、teacher forcing 与 rollout 均由上层统一框架处理。

Earthformer 的 `auxiliary` 为空。AlphaPre 的 `auxiliary` 包含 `[B, 1, 12, H, W]` 的 `amplitude_prediction` 和 `[B, 1, 12, H, W//2+1]` 的 `phase_prediction`，它们是待监督的分支张量，不能作为标量损失直接累加。调用 `core.supervised_losses(auxiliary, target, step)` 获得已经加权的 `phase_loss`、`amplitude_loss`、`anet_loss`，不包含主预测 MSE，也不重算网络前向。

频谱监督公式保持官方 `predict` 实现：低频掩码上的相位余弦距离、振幅 FFT 模长 MSE、振幅分支像素 MSE。默认系数分别为 0.01、0.01 和 0.1。振幅系数随传入的零起始优化器 `step` 线性衰减，在 `step=9999` 归零。该调度不受验证前向或重复 loss 查询的次数影响；官方原实现则在每次调用 `predict(compute_loss=True)` 时修改 Python 状态。

## 模型身份与默认规模

| 模型 | 配置 | 参数量 | 相对 44,162,136 |
| --- | --- | ---: | ---: |
| Earthformer | `base_units=288`, `depths=[1,1]`, `num_heads=4`, `num_global_vectors=8`, `stem_dims=[36,144,288]`, `stem_scales=[4,2,2]` | 43,994,248 | −0.38% |
| AlphaPre | `hidden_dim=496`, `spec_num=20`, 官方三个幅度时间单元 | 44,702,295 | +1.22% |

Earthformer 沿用官方 SEVIR 的两层 encoder/decoder、axial cuboid self attention、`cross_1x1` cross attention、全局向量和卷积降采样/上采样。为整除 448，stem 从 SEVIR 的 `[3,2,2]` 改为 `[4,2,2]`，attention 分辨率为 28×28 与 14×14。decoder 直接预测一步，输入输出通道均为 12。

AlphaPre 原有 AmpliNet、PhaseNet、AlphaMixer 均完整保留，包含幅度时间频域映射、相位加法/乘法/局部分支、低频掩码混合与可学习输出融合。`n_layers` 在官方实现的构造调用中不控制 AmpliNet 的层数，适配层保持官方默认三个单元，不将其描述成可调深度。

AlphaPre 在禁用 autocast 的上下文中将输入转换为 float32，并以 float32 执行整个官方内核及监督损失，规避非二次幂分辨率的低精度 FFT 限制。调用方应保留 float32 主权重，使用外部 autocast 管理混合精度，而不是将 AlphaPre 权重整体转换为 BF16。44.7M 配置在 448 原生分辨率运行，计算和显存开销较高；本次没有对该默认宽度执行 GPU 显存或吞吐量验收。

## 验证结果

`src/tests/baselines_stage3/test_earth_alpha.py` 共 9 项测试通过，CPU、单线程用时 17.54 秒。测试覆盖：

- 两个模型在 batch=2、七帧、16×16 下的前向、有限梯度反向及 AdamW 参数更新。
- 两个默认配置的真实参数计数，均位于目标 ±5% 内。
- 所有 vendored 源文件 SHA-256 与上游原件逐字节一致。
- 两个适配模型加载相同权重后，输出与原始上游实现逐元素完全一致；Earthformer 原实现通过独立子进程导入。
- AlphaPre 的三项分支监督与上游原始损失一致，并检查振幅权重归零时刻。
- AlphaPre 在 28×28 与 448×448、BF16 输入且 CPU BF16 autocast 开启时输出有限 float32 结果；该空间验证使用 `hidden_dim=8`，不代表默认 496 宽度的吞吐测试。

验证命令：

```powershell
$env:PYTHONPATH='C:/Users/13726/.codex/tmp/baseline-eval-extra-deps;C:/Users/13726/.codex/tmp/baseline-audit-20261117-python-deps;D:/rain-prediction/baseline_stage3_20261118'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
D:/Anaconda3/python.exe -m pytest src/tests/baselines_stage3/test_earth_alpha.py -q -o cache_dir=.pytest_cache
```

上游 `torch.meshgrid` 未指定 `indexing` 导致 PyTorch 兼容提醒，未修改原源码消除该提醒。
# Shared residual protocol update

The final integration YAML enables `rain_linear_amplitude=true`: the independent adapter calls the original AmpliNet, PhaseNet and AlphaMixer, preserving sigmoid for the eleven radar/satellite channels and using a linear rain amplitude. Auxiliary rain supervision uses the same native rain residual target as the shared prediction objective. This accommodates signed residuals without changing the byte-preserved original AlphaPre source. The core default remains false for exact source-equivalence tests; the final experiment configuration explicitly enables the adaptation. Parameter counts are unchanged.

