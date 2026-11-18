# OpenSTL 独立适配说明

本适配以 OpenSTL 的 `eecf8a3078f0a178dbc7b28723da20f94ce36985` 为来源，分别实现 ConvLSTM、PredRNN-V2 和 SimVP 2022 的 IncepU 版本。上游地址为 <https://github.com/chengtan9907/OpenSTL>，许可证为 Apache-2.0。运行所需代码均包含在独立项目中，仅依赖 PyTorch，不需要导入 OpenSTL、Lightning 或 timm。

`src/baselines_stage3/deep/openstl_port/PROVENANCE.json` 记录来源 commit、原始文件 SHA-256 和适配范围，同目录保留完整许可证。两个 recurrent cell 是官方文件的逐字节副本。`simvp_incepu.py` 原样选取官方 IncepU 相关类和函数，只整理了局部导入，将初始化函数导入改为 PyTorch 的 `torch.nn.init.trunc_normal_`。其他 SimVP 变体不在此适配范围内。参考文件目录 `reference/` 保留官方模型及相关工具的逐字节副本，用于离线复现数值对照，不参与运行时模型导入。没有修改上游目录，也没有对原框架进行运行时替换。

## 输入、输出和滚动规则

入口为 `build_openstl(method, options)`，支持 `convlstm`、`predrnn_v2` 和 `simvp`。共同默认值为 `image_size=448`、`context_frames=7`、`in_channels=12`。前向只接受 `[B, 12, 7, H, W]` 历史序列，返回原始 12 通道单步预测 `[B, 12, 1, H, W]` 和辅助损失字典。核心模型不接收未来标签，不处理降水残差锚点、缺失模态、teacher forcing 或滚动窗口；这些由外层共同协议完成。

ConvLSTM 和 PredRNN-V2 保留官方 patch 重排、各层状态演化和最后的投影。每次调用从零初始化 recurrent 状态，依次读入全部历史帧，以最后状态产生下一帧。只保留所需的最终输出，省去不会影响状态的中间输出投影计算。采用固定已观测历史的单步调用，因此官方内部 scheduled sampling/reverse scheduled sampling 分支不启用；外层负责未来时段的自回归或 teacher-forced roll-in。

PredRNN-V2 保留共享时空记忆、共享 adapter，以及每个历史时刻、每一层的 `delta_c`/`delta_m` 解耦项。前向返回 `auxiliary_losses["decoupling"]`，计算方式与官方一致：adapter 投影后进行空间维归一化，取逐通道余弦相似度绝对值，再对批次、通道、时间和层平均。它是未加权的标量，训练配置应使用 `aux_weight.decoupling=0.1`，与官方 `decouple_beta=0.1` 对齐。解耦项没有丢弃，也没有混入未来标签的监督误差。

SimVP 保留原始空间 encoder/decoder 和多尺度分组卷积 IncepU translator，生成与输入长度一致的未来序列，再取第一张未来帧。该裁剪规则与官方 `openstl/methods/simvp.py` 在 `aft_seq_length < pre_seq_length` 时一致。

## 默认模型规模

以 44,162,136 参数为匹配目标，下列默认结构均在其 1% 内。统计只包含核心模型；外层可学习缺失标记等参数需由完整适配器另计。

| 方法 | 结构选项 | 注册参数量 | 相对目标偏差 |
| --- | --- | ---: | ---: |
| ConvLSTM | `num_hidden=[210,210,210,210]`, `patch_size=4`, `filter_size=5`, `layer_norm=False` | 44,115,120 | -0.1065% |
| PredRNN-V2 | `num_hidden=[164,164,164,164]`, `patch_size=4`, `filter_size=5`, `layer_norm=False` | 44,110,752 | -0.1164% |
| SimVP IncepU | `hid_S=64`, `hid_T=448`, `N_S=4`, `N_T=8`, spatial kernels 3 | 43,830,156 | -0.7517% |

ConvLSTM 官方 cell 注册了未在前向使用的 `conv_o` 和 `conv_last`。为保留官方结构及 checkpoint 键，本适配原样保留这些模块，未将其当作额外有效能力。除去这些完整闲置模块后，ConvLSTM 为 34,942,320 参数；公平性比较应同时报告注册参数与该差异。其余两个模型没有这种完整闲置子模块，但单步 SimVP 与官方短预测规则一样，只监督未来序列的第一帧。

## 轻量验证配置与结果

小尺寸验证固定 `in_channels=12`、`context_frames=7`，测试 `image_size=16` 和 `32`，批量为 2。

| 方法 | 选项 |
| --- | --- |
| ConvLSTM | `num_hidden=[8,8]`, `patch_size=2`, `filter_size=3`, `layer_norm=False` |
| PredRNN-V2 | `num_hidden=[8,8]`, `patch_size=2`, `filter_size=3`, `layer_norm=False` |
| SimVP | `hid_S=8`, `hid_T=16`, `N_S=2`, `N_T=2` |

验证文件为 `src/tests/baselines_stage3/test_openstl.py`。17 项测试通过，覆盖输出形状和有限值、历史输入梯度、全部有效参数梯度、一次优化器更新、PredRNN adapter 的非零梯度、patch 通道/像素顺序、错误历史长度拦截、默认模型精确参数量和源文件 hash。

数值对照在独立测试命名空间中运行未修改的官方函数/类定义，装载相同 state dict，未替换任何全局模块。ConvLSTM 和 PredRNN-V2 在有/无 LayerNorm 时均与官方最终预测逐位一致；PredRNN-V2 的解耦数值和 adapter 梯度也通过容差比较。SimVP 与官方 IncepU 的第一张未来帧逐位一致。小尺寸结果验证结构和训练连通性，不代表已完成 448 分辨率、默认宽度的完整数据集训练。

运行命令：

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python -m pytest src/tests/baselines_stage3/test_openstl.py -q
```
