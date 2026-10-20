# 四川降雨简洁对照图

入口：`src/tools/render_sichuan_case.py`。读取已有 `forecast_raw.npz`、`case.json` 和缓存的四川边界，不重新运行模型。

每张图左侧 GT、右侧 Pred，白底、细红色行政边界，顶部仅显示 `frame=05 | time=2025-08-11 02:00:00`。使用原有统一色阶，不平滑、不调整预测数值。模型来源、原始数组哈希和完整时间信息保存在 `render.json`。

在 num8 执行：

```bash
conda activate rainpred

cd /home/yanjieyu/RainPrediction

git pull --ff-only origin project

PLOT_OUT="$PWD/vis_show/compact_3h_$(date +%Y%m%d_%H%M%S)"

python -m src.tools.render_sichuan_case \
  --cases vis_show/sichuan_* \
  --max-lead-minutes 180 \
  --dpi 200 \
  --output-dir "$PLOT_OUT"

echo "$PLOT_OUT"
```

`--cases` 可以改为一个具体推理输出目录，或多个目录。默认从各目录末个历史时刻计算预报时效，保留未来 180 分钟内所有帧，不按真值评分筛选。原本输出 12 帧的结果会截取前 6 帧，已有 6 帧的结果直接重画。

输出按原推理目录名分子目录，每个子目录包含逐帧 PNG/PDF、`comparison.gif` 和 `render.json`。起报时间为 2025-08-10 23:00 时，3 小时后的图是 `frame_05_20250811_0200.png`。帧号从 0 开始，与原序列索引一致。

本地验证使用合成数组检查 GT/Pred 顺序、简洁标题、原始数值和文件哈希不变、跨日 3 小时时效筛选以及 GIF 帧数。真实数据绘制需在保存推理结果的服务器执行。
