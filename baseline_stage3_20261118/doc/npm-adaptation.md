# NPM：严格卫星输入的独立迁移协议

此实现为 AAAI 2025《Data-driven Precipitation Nowcasting Using Satellite Imagery》的公开代码迁移版。完整预测链路是历史卫星 → 未来卫星 → 未来降水。它保留公开代码中的卫星预报网络和 Sat2Rdr ResNet 生成器，不是仅输出卫星的预报器，也不是另加一个通用 CNN 雨头。实验名称应注明 **NPM code-port / satellite-only / joint-MSE adaptation**，不得将本协议结果标为论文原设置或原 GAN 训练结果。

## 来源与可追溯性

作者：[Young-Jae Park、Doyi Kim、Minseok Seo、Hae-Gon Jeon、Yeji Choi](https://github.com/seominseok0429/Data-driven-Precipitation-Nowcasting-Using-Satellite-Imagery)。固定 Git revision 为 `52ca97702f9d9b7e314cf0ce9e053d3aa06c5bea`。参考文件为 `model.py`、`modules.py`、`sat2rdr/models/pix2pixhd.py`，原文件未改写；本实现只包含执行这两段网络所需的独立 PyTorch 移植，不注入替代模块或运行时补丁。

原 README 于 2025-10-10 宣布迁移到 [SAR2RDR-Dataset](https://huggingface.co/datasets/minseok96/SAR2RDR-Dataset/tree/8270366c9bd82e7cdd22a3762237ece47d6eeb47)。2026-11-18 核验该 revision 的顶层只有数据目录、README、雷达处理脚本及可视化 notebook，没有新版预报或翻译网络，因此网络仍以已固定的 Git 源码为依据。逐文件 SHA-256 保存在 `src/baselines_stage3/deep/npm_port/SOURCE_MANIFEST.json`。

## 实际保留的网络

卫星预报器保留逐帧卷积编码器、跳跃连接、把时间折叠进通道的时序预测器、LKA 瓶颈、带 day/hour 条件的 ConvNeXt 块、PixelShuffle 解码器及最终 1×1 readout。`hid_T` 在原代码中没有参与计算，不在新 API 中提供无效选项。

翻译器对应 `sat2rdr/train.py` 使用的 `models.pix2pixhd.Generator` 的 ResNet 分支：7×7 输入卷积、4 次 stride-2 下采样、9 个双 3×3 卷积残差块、4 次转置卷积上采样、7×7 输出卷积。保留训练入口指定的 ReLU 和 BatchNorm2d。它逐个处理预测出的未来卫星帧；两阶段之间没有 detach，雨损失可传回预报器。

### 公开代码与论文的差异

公开源码与论文并不完全一致，不能默默替二者补齐对方缺失的组件：

| 项目 | 固定 Git 中实际执行 | 论文或补充材料 |
|---|---|---|
| 卫星预报块数 | `main.py` 指定 `N_T=8` | 正文写 3 个 ST 块，补充材料表 1 写 translator×6 |
| 时空模块 | 首个 LKA 瓶颈加 ConvNeXt；未实现单独 TKA | 正文有 TKA 和 LKA |
| 卫星目标 | `exp.py` 只有 MSE | 正文含权重 0.1 的 temporal consistency KL |
| 翻译器 | pix2pixHD 风格 ResNet，默认 4 次下采样、9 残块 | 补充材料为 StegoGAN 双 G/D、2 次下采样、6 残块 |
| 翻译训练 | `GANLoss` 使用 LSGAN、feature matching 与可选 CCC | 正文写 GAN+MSE，采用 StegoGAN paired 配置 |

本移植忠实于可执行源码，未凭论文示意图另造 TKA、StegoGAN 或 temporal-KL 实现。`sat2rdr/eval_10.py` 还存在 VAN/SiLU 的另一组默认值；本协议采用训练入口及 README 的 ResNet/ReLU 变体，并明确记录，不混用这两组 checkpoint 配置。

## 输入、输出及必要改动

`build_npm(options)` 返回模块；调用形式：

```python
result = model(satellite, time)
# satellite: [B, 10, 4, H, W]
# time: [B, 4, 2], calendar day of year and fractional hour
# result["satellite"]: [B, 10, 4, H, W]
# result["rain"]: [B, 1, 4, H, W]
```

时间来自历史观测时间戳，最后轴是 `(day_of_year, hour + minute/60 + second/3600)`，范围分别为 `1..366` 和 `[0,24)`。只将最后一帧的时间送入原有两个 64 维 sinusoidal+MLP 编码器。缺少时间直接报错。官方 loader 忽略闰日且丢弃分钟，本协议修正为公历日序及小数小时。转换必须与数据集原时间戳使用同一时区语义，不能把当地时间再次错误转换为 UTC。

适配仅发生在独立命名空间：

1. 原来的 6→6 帧改为本任务固定 4→4；3 个卫星通道加 DEM 改为全部 10 个卫星通道。不注入 DEM、历史雷达、历史降雨或未来观测。
2. 原 `NPM.forward` 最后硬裁 `:3` 改为保留全部 10 个预测卫星通道，全部进入翻译器。
3. 原 `Predictor` 创建 `N_T+1` 个模块，却只执行 `N_T` 个。移植仅保留实际执行的块，删除的是新实现中的死注册项，计算图不变，也不会把无梯度参数计入容量。
4. 原翻译器 Tanh 适配为线性输出。原 Sat2Rdr 雨量按 `[0,100]→[-1,1]` 编码，而本项目 `rain_interpolated` 目标保留原值且无上限；保留 Tanh 或硬映射到 `[0,1]` 会限制有效雨量。新输出不 clamp，保持 MSE 梯度，允许训练早期出现负预测，评估使用统一框架规则。
5. `timm.trunc_normal_` 和 `einops.rearrange('b c -> b c 1 1')` 分别使用等价的 `torch.nn.init.trunc_normal_` 和张量广播，不需要额外模型库。非所选分支及训练脚本导入不带入移植包。

## 参数与容量

默认配置如下。缩小翻译器宽度使整个降水预测管线进入参照模型容量范围，未省略翻译器后再报告参数量。

| 选项 | 默认 |
|---|---:|
| `image_size` | 448 |
| `context_frames` / `forecast_frames` | 4 / 4 |
| `in_channels` | 10 |
| `hidden` | 64 |
| `spatial_depth` | 4 |
| `temporal_depth` | 8 |
| `translator_width` | 30，原训练入口为 64 |
| `translator_downsamples` | 4 |
| `translator_residual_blocks` | 9 |
| `translator_norm` | BatchNorm2d |

默认卫星预报器 **5,306,122** 参数，翻译器 **40,125,031** 参数，完整管线 **45,431,153** 参数，比参照 **44,162,136** 多 **2.87%**。所有注册参数都在训练图内，tiny 测试检查全部参数都有梯度。参数量不包含训练专用判别器，因为本迁移训练不使用 GAN。

## 原训练与本实验训练

固定 Git 原入口分别训练两个模型。卫星预报器使用 Adam `lr=0.002`、OneCycleLR、2000 epochs、MSE，使用月份分组采样。翻译器使用 Adam `lr=0.0002, betas=(0.5,0.999)`、200 epochs，100 epoch 后多项式衰减，交替更新生成器和判别器；默认目标是 LSGAN 加 feature matching×10 和 CCC×5，并非当前联合目标。

本实验为独立比较协议：从随机初始化开始，以本项目同一数据拆分和目标单位训练两段网络，默认 **70,000 optimizer steps**。损失为 `2*MSE(rain_prediction, rain_future) + 1*MSE(satellite_prediction, satellite_future)`。未来降水和卫星只用于损失监督，绝不进入模型 forward。没有 GAN 判别器、分阶段预训练、原论文的 temporal-KL 或原月份采样；这些差异必须随指标一起报告。调用者负责执行上述步数、优化器和损失，模型本身不读取数据集或目标。

NPM 只在 `satellite_only` 组报告。其他多模态方法若在 rollout 使用真实未来辅助观测，信息条件不同，不应将两者包装为同条件总榜。

## 已执行验证

- `src/tests/baselines_stage3/test_npm.py`：10 项通过，包括 shape、有限性、两段网络真实 optimizer 更新、仅雨损失也能传回卫星预报器、翻译器只消费预测卫星、严格输入及未知配置拒绝、线性目标范围和完整参数预算。
- 真实默认 `448×448`、batch=1 完整管线前向通过；两种输出尺寸正确且全部 finite。CPU 单线程含初始化约 8.14 秒，该数字是本机 smoke timing，不是论文速度对比。
- 默认全部 45,431,153 个参数和完整深度不变、仅空间尺寸缩到 `32×32` 的训练步也通过：全部梯度存在且 finite，SGD step 后参数确实更新。
- 在独立 Python 进程直接读取原源码，加载相同权重后，tiny 原 NPM 的前三通道与移植预报器对应输出在 `rtol=atol=1e-6` 下相等；唯一未匹配 state keys 是原来的末尾未执行块。
- 原 Generator 与 `tanh(移植线性生成器输出)` 在相同权重及 tiny 输入下相等，全部 state keys 匹配。原始计算图与输出激活适配被分别核验。

完整分布式训练和真实数据集精度仍需在目标训练环境完成；以上验证不代表完成 70,000 steps 的科研实验。
