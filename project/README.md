# RainPrediction 实时推理部署说明

本说明用于交付「实时读取 NC 数据并进行  推理」能力。

## 1. 功能概览

当前链路分为两段：

1. `realtime_nc_to_stream.py`
- 持续监听新到的 `radar/satellite/rain` NC 文件
- 进行空间映射/对齐与标准化落盘
- 输出到统一帧流目录（`runtime_stream/<timestamp>/...`）

2. `realtime_stream_infer.py`
- 持续监听帧流目录
- 取最近 `n_past` 帧组窗（当前来自模型配置，默认 5）
- 加载 EMA 权重执行推理
- 输出 `json` 和可视化 `jpg`

---

## 2. 文件组织

请按如下结构组织工程（相对项目根目录）：

```text
RainPrediction/
├─ src/
│  ├─ tools/
│  │  ├─ ai_model_invoke.py
│  │  ├─ demo_process_ai_predict_rainfall.py
│  │  ├─ realtime_nc_to_stream.py
│  │  └─ realtime_stream_infer.py
│  ├─ networks/
│  │  ├─ SwinNet.py
│  │  └─ modules/
│  │     ├─ SwinTransformer.py
│  │     └─ ttt_memory.py
│  ├─ dataset/
│  │  ├─ read_nc_file_mapped.py
│  │  ├─ rain_ts_litdata.py
│  │  └─ geo_utils.py
│  ├─ data_preprocessors/
│  │  └─ all_to_figs.py
│  └─ utils/
│     ├─ visualization/
│     │  ├─ plot.py
│     │  └─ color.py
│     └─ metrics/
│        └─ compute_metrics_cls.py
├─ runs/
│  └─ swinnet_cls_10min_AR/
│     └─ 2026-05-09_23-55-33_rain_train_pasts_n=5_future_n=5/
│        ├─ config/
│        │  └─ config_total.yaml
│        └─ ema/
│           └─ rain_model/
│              └─ model.safetensors
├─ test_data/
│  ├─ achn/
│  ├─ sat/clip/
│  └─ rain/
└─ results/
```

说明：
- `config_total.yaml` 和 `ema/rain_model/model.safetensors` 是运行推理必需。
- `test_data` 仅用于演示；生产环境替换为实时落地 NC 目录。

---

## 3. 环境要求

- Python >= 3.12
- PyTorch（CUDA 环境按需）
- accelerate
- kornia
- h5py
- tifffile
- xarray（若走 mapped 全能力建议安装）
- netcdf4 / h5netcdf（建议安装，避免 xarray 后端错误）

建议额外安装：

```bash
pip install netcdf4 h5netcdf
```

---

## 4. 关键参数位置

模型输入输出帧数来自：

- `runs/swinnet_cls_10min_AR/2026-05-09_23-55-33_rain_train_pasts_n=5_future_n=5/config/config_total.yaml`

其中：
- `rain_prediction_model.n_past`：输入历史帧数
- `rain_prediction_model.output_frames`：输出未来帧数

当前为 `n_past=5`、`output_frames=5`。

---

## 5. 启动步骤

### 步骤 A：启动实时预处理（NC -> 标准流）

```bash
python src/tools/realtime_nc_to_stream.py \
  --radar-dir test_data/achn \
  --sat-dir test_data/sat/clip \
  --rain-dir test_data/rain \
  --out-dir runtime_stream \
  --img-size 256
```

说明：
- 脚本会持续轮询目录，不会自动退出。
- 终端输出 `wrote frame ...` 表示一帧成功产出。

### 步骤 B：启动实时推理（标准流 -> 结果）

另开一个终端：

```bash
python src/tools/realtime_stream_infer.py \
  --stream-dir runtime_stream \
  --out-dir results/realtime_infer \
  --poll-seconds 5
```

输出：
- `results/realtime_infer/<timestamp>.json`
- `results/realtime_infer/<timestamp>.jpg`

---







