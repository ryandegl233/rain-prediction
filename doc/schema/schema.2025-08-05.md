# 空间超分辨率


## 方案设计
两种方案：

1. MLP直接学习空间**点**上的映射关系，也就是$x\in X$, $X$大小为$H\times W\times 3$，$x$的大小为$1\times 1\times 3$，即为一个像素点。网络输出为$\tilde y=F_\theta (x)$，学习目标为*有雨量站的雨量数据*，学习目标为真实Y上的像素$y$进行监督。
2. CNN （朱捷宜现在实现为一个U-Net网络），学习空间映射关系，网络输出为$\tilde Y=F_\theta (x)$，学习目标为*有雨量站的雨量数据*，通过四川省雨量站的mask在真实Y上进行监督学习。

## 目前进展
1. MLP方案尚未尝试；
2. 使用CNN进行训练，但是由于监督信号过于系数，导致需要将有降雨量的地方进行加权，导致训练出来的图模糊
    - 缓解方法：调整加权；使用其他的CNN网络；使用其他的loss函数。

## 下一步方案
- [] 尝试方案一 （assign to Jieyi）；
- [] 进一步调整CNN的训练配方。


---


# 雨量预测

## 两步走——非时序

跟龙昊一样，暂时不使用时序数据，直接使用成对的卫星/雷达数据进行训练：
    $$\tilde Y = F_\theta(X_{radar}, X_{satellite}),$$
其中，$\tilde Y$是真实的雨量数据（暂时为经过第一轮空间超分辨率的处理，但是可以crop到雨量站分布集中的区域：经度99.20~106.80， 纬度26.00~33.66），这里不存在时序的问题，所有的数据都是成对的。

> 这一部分的数据是整理好的，提供webdataset tar文件供流式读取。

- 既然不考虑时序，那么网络$F$的选取较为随意，可以使用各种dense-predition的网络，例如DepthAnything中的Dino+DPT，Unet等；

- 训练目标设置为l1或者为l2，可能也需要进行有雨和无雨的加权

- 进一步可以尝试将雨量进行量化成离散值，然后loss可以使用现成的分割损失，直接使用focal loss或者dice loss，进行训练。




## 两步走——时序

暂时龙昊也没有尝试进行时序数据的训练。

涉及到时序的数据准备过程和网络配置较为复杂：

### 数据
首先存在几个问题，1）输入，输出为
$$
    \{\tilde Y_{p}, \tilde Y_{p+1}, \dots, \tilde Y_{p+n} \} = F_\theta(X^{radar}_1, \dots X_p^{radar}; X^{sate}_{1}, \dots, X^{sate}_{p}; \bar Y_1,\dots, \bar Y_{p-1}),$$
其中$\bar Y$可以是之前预测的$\tilde Y$或者真实值$Y$，这里对应到不同的时序预测方法，另外$n,p$的配置需要仔细权衡。由于需要设置不同的$n,p$导致webdataset的流式读取不再可用，现给出两种数据读取方案：

a. 直接将所有的数据使用都存成图像进行操作，维护一个table(使用parquet文件)存储数据的元信息进行索引，例如：

| dir | time | file index | file name |
| --- | --- | --- | --- |
| data/radar | 2025-08-05 09:00:00 | 1 | radar_1.png |
| data/radar | 2025-08-05 09:00:00 | 2 | radar_2.png |
| data/radar | 2025-08-05 09:10:00 | 3 | radar_3.png |
| data/satelite | 2025-08-05 09:00:00 | 1 | satellite_1.png |
| data/satelite | 2025-08-05 09:00:00 | 2 | satellite_2.png |
| data/satelite | 2025-08-05 09:10:00 | 2 | satellite_2.png |
| data/rain | 2025-08-05 09:00:00 | 1 | rain_1.png |
| data/rain | 2025-08-05 09:00:00 | 2 | rain_2.png |
| data/rain | 2025-08-05 09:10:00 | 2 | rain_2.png |

或者offline配置读取对为json文件，例如
```json
{
    // 这里的n=2, p=3，假设时间间隔是10分钟每帧
    "history_start_time": "2025-08-05 09:10:00",
    "history_end_time": "2025-08-05 09:30:00",
    "prediction_start_time": "2025-08-05 09:30:00",
    "prediction_end_time": "2025-08-05 09:40:00",
    "history_radar_data": [
        "data/radar/radar_1.png",
        "data/radar/radar_2.png",
        "data/radar/radar_3.png",
    ],
    "history_satellite_data": [
        "data/satellite/satellite_1.png",
        "data/satellite/satellite_2.png"
        "data/satellite/satellite_3.png"
    ],
    "history_rain_data": [ 
        "data/rain/rain_1.png",
        "data/rain/rain_2.png",
        //注意这里没有rain_3.png，需要预测的
    ],
    "gt_rain_data": [
        "data/rain/rain_3.png",
        "data/rain/rain_4.png"
    ]
}

```
所有的数据读取逻辑都是torch dataset中进行处理，此时的n,p配置较为灵活，在dataset中的
> 缺点：这种数据读取处理方式难以扩展，并且涉及到随机访问，对于大数据集，效率可能会下降（需要测试）。

> 雨量站也需要存成图吗,duckdb如果无法多线程访问这一点存疑？

> xarray有太多无用的元信息存储，另一方面没有成熟的decoder，也无法进行高效的压缩，读取速度较慢，建议全部转换为tiff文件进行读取，不要使用xarray。

> json文件的处理过程中需要filter/平衡有雨和无雨的数据，json文件的缺点在于需要自行组织n,p，不同的n，p设置需要使用不同的json文件，但是不需要额外组织文件，这也是时序数据的通用方法，从这一点上来看，这个方案较为实用。

b. 为了解决数据读取效率问题，仍然可以使用webdatase的index版本wids，进行流式读取，但是需要对数据进行预处理成为tar文件，这样解决了随机访问的问题，使得读取效率加快。wids仍然将数据处理成为一个tar文件，仍然需要一个parquet文件进行索引，此方案需要额外设置torch dataloader的sampler进行index，较为复杂。

c. ~~龙昊给出的解决方案无法适用于时序数据，他们在成对的数据上使用xarray进行索引。~~


### 方法

在整理完数据之后，对于时序预测的方法较多，下面给出两种方案进行时序雨量预测：

1. autoregressive next-prediction model (也就是teacher forcing)：这里就是常说的AR model，输入为上述的$3p-1$帧数据，这里需要仔细地实现attention mask,考虑时序的causal mask，也就是上一帧不能看到下一帧的数据（不能泄露历史），但是输入帧可以看到以前的帧（包括同一模态和不同模态的）；帧之间的像素是双向注意力，跨帧的就是causal注意力，这一点需要额外的注意，不然推理的时候无法rollout。优点在于可以使用kv cache，缺点是，mask比较复杂

2. 转向diffusion方案，重点看diffusion video生成相关的文章，将雨量站作为diffusion model输入，历史雷达和卫星帧编码之后全部作为diffusion model的condition，生成新的雨量帧。diffusion处理不同雨量帧之间需要遵顼causal原则，重点看cogvideoX，Wan2.1, Wan2.2的文章；优点在于结构简化，causal mask没那么复杂，缺点在于无法复用kv cache，推理较慢；

3. 从teacher force和diffusion转向diffusion force，作为一个新的diffusion时序方法，其噪声调度稍微复杂，但是（据说，从文章上看）长时间预测效果好。diffusion force有两篇文章，直接可以搜到。

- 额外需要注意positional embedding，可以暂时先使用sine-cosine，无需考虑RoPE。