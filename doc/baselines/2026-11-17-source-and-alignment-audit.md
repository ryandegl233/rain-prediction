# Baseline 源码与迁移核查

日期：2026-11-17。本文记录第一阶段的源码拉取、接口核查与参考网络参数统计；下文的“尚未实现”指该阶段快照。第二阶段的 Persistence/LK、公共评测与训练接入见 [服务器运行说明](2026-11-17-evaluation-server.md)。其余深度 baseline 的适配及容量容差仍属于建议。

用户指定参考配置为 `src/config/ts_rain_train/rain_trainer_ts_next_frame.yaml`，并确认主表统一采用固定预报起点、只使用历史观测的 4→4 预测协议。

## 已拉取的源码

仓库放在 `third_party/baselines/`，均为独立 Git 浅克隆，上游文件未修改。完整地址、分支、commit 和许可证文件记录在同目录 `UPSTREAMS.json`。本次未执行上游安装脚本、下载训练数据集或预训练权重；仓库本身跟踪的资源随 Git checkout 保留。

| 仓库 | 覆盖方法 | 固定提交 | 关键入口 |
| --- | --- | --- | --- |
| [OpenSTL](https://github.com/chengtan9907/OpenSTL) | ConvLSTM、PredRNN-V2、SimVP | `eecf8a3078f0a178dbc7b28723da20f94ce36985` | `openstl/models/{convlstm,predrnnv2,simvp}_model.py` |
| [Earthformer](https://github.com/amazon-science/earth-forecasting-transformer) | Earthformer | `7732b03bdb366110563516c3502315deab4c2026` | `src/earthformer/cuboid_transformer/cuboid_transformer.py` |
| [AlphaPre](https://github.com/linkenghong/AlphaPre) | AlphaPre | `64d5195984c72db2692266acc800efa34ff8a4bf` | `models/alphapre.py` |
| [NPM](https://github.com/seominseok0429/Data-driven-Precipitation-Nowcasting-Using-Satellite-Imagery) | NPM | `52ca97702f9d9b7e314cf0ce9e053d3aa06c5bea` | `model.py`、`sat2rdr/models/pix2pixhd.py` |
| [pySTEPS](https://github.com/pySTEPS/pysteps) | PySTEPS-LK | `81b819debc1eef054464a990165bf99165856b98` | `pysteps/motion/lucaskanade.py`、`pysteps/nowcasts/extrapolation.py` |

Persistence 不需要第三方仓库，直接重复最后一帧历史降雨即可。

NPM 当前 README 声明迁移到 [SAR2RDR-Dataset](https://huggingface.co/datasets/minseok96/SAR2RDR-Dataset)。核查该数据仓库 revision `8270366c9bd82e7cdd22a3762237ece47d6eeb47`，文件索引中 Python 文件仅有 `processing_rdr.py`；完整模型源文件仍能从上表 GitHub checkout 获取。不能仅依据迁移说明认为数据仓库包含可运行的完整训练框架。

## 参考网络：实测 44.16M 参数

配置目标是 `src.networks.time_series.causal_patch_transformer_next_frame.RainCausalPatchTransformerNextFrame`，由 `src/trainer/rain_trainer_ts_next_frame.py` 实例化。

| 项目 | 当前配置 |
| --- | --- |
| 历史／未来 | 4 帧／4 帧，30 分钟间隔，评测 +30、+60、+90、+120 分钟 |
| 图像尺寸 | 448×448 |
| 模态 | 雷达 1 通道、卫星 10 通道、降雨 1 通道 |
| 项目张量布局 | `[B, C, T, H, W]` |
| 编码 | patch encoder，空间 patch=8，时间 patch=1，token 网格 56×56 |
| 主干 | dim=384、depth=6、heads=8、MLP ratio=4 |
| 解码 | 三个独立 3D decoder，base channels=128，kernel=7，wan_factorized |
| 其他 | 模态 mask token、时间嵌入、空间／时间 RoPE、activation checkpoint |

`encoder_spatial_downsample_stages=4` 和 `encoder_temporal_downsample_stages=1` 是配置字段，但当前选择 `encoder_type=patch`；不能据此把当前网络称为四级 ResNet 编码器。

CPU 真实构造模型，累计 `model.parameters()` 得到：

| 部分 | 参数数 |
| --- | ---: |
| Patch embedding | 295,296 |
| Stem | 1,328,832 |
| 时间嵌入投影 | 148,992 |
| 六个时空 Transformer block | 21,295,872 |
| 最后归一化 | 768 |
| 雷达 decoder | 6,627,137 |
| 卫星 decoder | 6,627,722 |
| 降雨 decoder | 6,627,137 |
| 顶层位置嵌入与 mask 参数 | 1,210,380 |
| **总计／requires_grad=True** | **44,162,136** |

原始结果在 `artifacts/baseline-audit-20261117/reference_profile.json`，包含解析后的模型配置和关键源文件 SHA-256。统计脚本为 `scripts/baselines/profile_reference.py`。在项目依赖齐全的环境、项目根目录运行：

```powershell
python -m scripts.baselines.profile_reference --config-name rain_trainer_ts_next_frame --output artifacts/baseline-audit-20261117/reference_profile.json
```

本次使用 CPU PyTorch 2.12.0；没有执行该网络完整前向、训练、FLOPs 或 GPU 显存测量。临时补充的统计依赖独立存放于 `C:/Users/13726/.codex/tmp/baseline-audit-20261117-python-deps`，未改项目依赖文件或全局 Python 包。

## 已确认的评测协议及需要单独覆盖的配置

原 YAML 的 `train.next_pred.rollout_branch.use_gt_future_modalities` 和 `val.rollout_use_gt_future_modalities` 均为 true。trainer 在每次预测后将对应时刻的真实雷达／卫星写入下一轮历史；frame 模式下第一步用历史输入，第二步及后续预测会获得预报起点之后的观测。

用户已确认主表只用历史观测。后续新建独立配置，同时将这两个字段设为 false。对原网络也使用同一协议，不能把历史条件 baseline 与使用未来真实辅助观测的原结果混在一张主表中。已有 checkpoint 可以先按该协议重评估，但应记录其训练曾使用的条件；正式受控比较需统一训练条件。训练时用于监督的未来标签与评测时喂入未来观测应区分，训练阶段的 teacher forcing／scheduled sampling 另行记录。

trainer 底部 Hydra 装饰器当前默认 `config_name="rain_trainer_ts_next_frame_delta_filter"`。启动指定配置必须显式给出 `--config-name rain_trainer_ts_next_frame`，最终以运行输出的 resolved config 为准。

还需在独立评测配置中统一以下内容：

- 使用相同月份划分、`metadata_rain_delta_filter.parquet` 筛选和相同窗口清单；当前 train 为 202305–202309、202506、202507，val 为 202508、202509。此划分没有独立 test 配置，不能把调参验证集直接称为测试集。
- 当前 val 为 shuffle=true、drop_last=true、最多 60 batch。建议生成固定完整验证窗口，shuffle=false、drop_last=false；所有方法使用同一清单。
- 当前监督来自 `rain_interpolated`，雷达除以 60、卫星除以 300；`modality_zero_centering=false` 时降雨不执行配置中的均值／方差标准化。雨量单位需要从数据元数据进一步确认，不能擅自标为 mm/h。
- 统一雨量评测域、阈值 `[0.1, 0.3, 0.5]` 和逐时效聚合。CSI/POD/FAR 建议汇总事件计数后计算，并报告 MAE/MSE；不混用各上游仓库的阈值和归一化。
- 训练预算需比较 optimizer step、有效 batch、见过的样本数、种子和模型选择规则。当前 train batch=8、累积=1、100000 steps；显存不足时用梯度累积保持有效 batch，并记录真实耗时。

## 容量怎样对齐

建议保留模型特有结构，优先调隐藏宽度，使包含输入／输出适配部分的总参数落在 **39.75M–48.58M**，即参考 44.16M 的 ±10%。这个范围是首轮建议，尚未完成各 baseline 的容量搜索。原始官方宽度作为另一个明确标注的参考配置保留；调整宽度后的模型应称容量匹配版本。

不要求 ConvLSTM、SimVP、Earthformer 都变成六层，也不要求它们使用同一个 decoder。循环层、卷积翻译器和 Transformer block 的层数没有直接等价关系。输入分辨率和输出时效必须一致；内部 patch/downsample、隐藏宽度和原模型核心结构逐项记录。

参数量相近仍不代表开销相近。正式运行补测同一输入、同一硬件上的推理 FLOPs/MACs、延迟和训练峰值显存，并报告完整预测链路的参数。需要同时留意注册但未参与 forward 的参数，例如当前 OpenSTL ConvLSTMCell 中 `conv_o`、`conv_last` 被定义而未在其 forward 使用，不能仅靠这些参数凑预算。Persistence 和 PySTEPS-LK 参数为 0，不进行容量匹配。

## 各方法接入与关键约束

| 方法 | 接入方式建议 | 容量调整／需要保留的机制 |
| --- | --- | --- |
| Persistence | 读取最后一帧历史 rain，重复 4 次 | 不训练；仅用于包含历史 rain 的组合 |
| PySTEPS-LK | 历史 rain 估计 LK 速度场，最后一帧做半拉格朗日外推 4 步 | LK+确定性外推，不使用随机 STEPS；固定无雨阈值、边界和无特征时的行为 |
| ConvLSTM | 包装 `ConvLSTM_Model`，转换 patch 和张量轴，评测禁止未来真值 sampling | 首先调 `num_hidden`，再讨论层数；patch 是可逆重排，不能以降低输出分辨率替代 |
| PredRNN-V2 | 包装 `PredRNNv2_Model`，对齐序列、mask_true 和返回值 | 保留 memory decoupling loss 和训练采样策略；调 `num_hidden`；共享 memory 要求兼容的隐藏宽度 |
| SimVP（2022） | 包装 `SimVP_Model`，显式 `model_type='IncepU'`，4→4 | 调 `hid_S`、`hid_T`，必要时调 `N_S/N_T`；默认 `gSTA` 是另一版本，不能混标为 2022 原版 |
| Earthformer | 调用 `CuboidTransformerModel`，输入输出使用 `[B,T,H,W,C]` | 调 `base_units`、`block_units`、`enc_depth/dec_depth`；保留 Cuboid attention |
| AlphaPre | 调用 `predict(frames_in, frames_gt, compute_loss)`，使用 `[B,T,C,H,W]` | 调 `hidden_dim`；保留相位／幅值／融合分支及对应 loss，不只保留像素 MSE |
| NPM | 单独接两阶段链路：卫星未来预测，再卫星→降水映射 | 总参数统计两阶段之和；仅 satellite-only 专项，显式记录是否提供 DEM |

### PySTEPS

本次拉到的主分支 `dense_lucaskanade` 和 `nowcasts.extrapolation.forecast` 接口已经使用 **xarray.Dataset**；不能照旧版 ndarray 示例直接调用。后续基于固定 commit 实现 Dataset 包装，或另行固定并记录稳定发行版，避免接口随安装版本漂移。运动输入需有至少两帧，使用本项目四帧历史；30 分钟为一个预测步，不把 4 步误写成 4 分钟。

### OpenSTL

ConvLSTM／PredRNN-V2 的核心 forward 使用 `[B,T,H_patch,W_patch,C_patch]`；SimVP 使用 `[B,T,C,H,W]`。循环模型原实现输入输出通道相同，适合固定 12 通道预测，再按模态拆分。评测未来位置只能填占位并关闭未来采样，不能把真实未来张量作为可访问条件。PredRNN-V2 训练损失中包含 `decouple_beta` 项，迁移时保留其含义并记录权重。

`layer_norm=True` 使用 `[hidden_channels,H_patch,W_patch]` 的可训练 LayerNorm，448 网格下会显著改变参数量，不能直接引用 Moving MNIST 的参数数。需实际构造计数。

### Earthformer

官方 SEVIR 配置在 `scripts/cuboid_transformer/sevir/cfg_sevir.yaml`。原始配置为 13→12、384×384、1 通道，stack-conv downscale 为 `[3,2,2]`。接本项目要显式配置 `[4,448,448,12]` 的历史输入和相应目标形状。建议核查 `[2,2,2]` 等适配 448 的内部下采样方案；这是待验证的候选，不能把直接替换分辨率视为适配完成。输入输出通道支持分别配置，可选择多任务输出或单 rain 输出，但比较时必须标明输出任务差异。

### AlphaPre

源码支持输入帧数、输出帧数、空间尺寸、通道数，但完整运行器仍有硬编码：`run.py` 将 `dim` 固定为 64。模型构造时 `n_layers` 没有传给 `AmpliNet`，其默认深度仍为 3；`PhaseNet` 接受 `n_layers` 却没有据此创建可变层数。容量搜索优先直接传 `hidden_dim`，不能声称只改命令行 layers 就完成深度对齐。

幅值分支包含 sigmoid，需检查多模态量纲和雨量范围。448 不是 2 的幂且涉及复数 FFT，首轮在 FP32 中执行 FFT 路径并测试反向，再决定混合精度边界。`itr`、衰减的 `amp_weight` 是普通 Python 状态，原 forward 在推理也递增 itr；独立 trainer 需明确训练步调度、验证隔离和 checkpoint 恢复，不能只保存张量 state_dict 后假定调度完整恢复。

### NPM

`model.py::NPM` 的输出是 `Y[:, :, :3, :, :]`，预测的是三个卫星通道；降水结果还需要 `sat2rdr` 的 Generator。原 dataloader 使用六帧历史／六帧未来，拼接 DEM，存在绝对数据路径。直接把第一阶段输出当雨量会比较错任务。

本项目有十个卫星通道，必须核实对应原文 IR10.5、WV6.3、WV7.3 的实际波段索引。不能把前三个数组通道直接视作这三种物理通道。若使用全部十通道、移除 DEM 或把映射目标改为本项目插值降雨，均需标为项目迁移版本。上游 `dem_new.npy` 是其区域资源，不能直接当作四川地形。

`Predictor` 的 `channel_hid` 未实际使用，`hid_T` 不能当作已验证的容量旋钮；`N_T` 所构建的层数与 forward 遍历范围也需核对。保持上游不变，在独立实现中明确所采用的逻辑与偏离点。

## 推荐的独立框架边界

后续新增 `src/baselines/`（本次尚未建立），分离输入适配、各方法包装器、训练和评测；独立配置放在 `src/config/baselines/`，pytest 放在 `src/tests/baselines/`。使用原始类的公开接口进行组合，不改 upstream 文件，不做运行时替换，也不将参考网络的三路 decoder 强行装进每个 baseline。

建议统一对外接收历史张量和模态可用性 mask，返回 `[B,1,4,448,448]` 的降雨预测及可选辅助输出。对需要相同输入输出通道的架构，可内部固定 12 通道、输出 12 通道再取 rain；不能把多任务 loss 与单雨量 loss 的差异隐藏在包装器中。

缺失模态建议先复用 radar/satellite/rain 三组随机 dropout 训练，再以七个非空模态组合固定 mask 评测。Persistence／LK 只覆盖含 rain 的四组。原网络的 dropout 比例为 0.15/0.15/0.05、至少保留一组；baseline 需采用可比较的缺失定义。关闭某组时，历史输入、残差锚点及 rollout 回填都不得偷偷读取该组真实值；未来真实观测在任何组均不可用。

原训练目标含多任务 MSE、降雨残差、时序差分、雨区加权等。建议先建立相同 rain 主监督的受控轨道，同时保留 PredRNN-V2、AlphaPre、NPM 的必要方法损失；多任务辅助监督作为显式配置统一报告。是否把原网络的完整专用损失用于所有 baseline 尚待确定，不能把任一选项称为已完成的公平对齐。

实现顺序建议为 Persistence／LK 验证公共评测链路，随后 OpenSTL 三个模型，再 Earthformer／AlphaPre，最后 NPM 两阶段专项。每个可学习模型先做输出 shape、有限值、反向梯度和缺失模态测试，再在真实数据上做短 smoke。增加“修改全部未来真值不影响推理输出”的测试，用来直接验证已确认的历史条件协议。

## 本次验证范围

五个上游仓库已逐一核对 commit，`git status --porcelain` 均为空。参考参数脚本已成功执行，输出完整模型参数拆分。本次未声称任一 baseline 已训练通过、已在 GPU 上运行、已达到 44.16M 预算或取得预测性能结果。
