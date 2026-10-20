# 四川案例选图与展示排版

入口：`src/tools/select_sichuan_cases.py`。读取既有出图目录中的 `forecast_raw.npz`、`case.json` 和 `sichuan_boundary.geojson`，无需重新推理或 GPU。

目标是在同一起始时间的多个 checkpoint 结果中挑选表现较好的案例，用统一色标、高分辨率 PNG/PDF 展示。保留原始预测，不抹除小雨、不扩大雨区、不按真值修正预测。原训练、推理和绘图程序均不修改。

## 选图口径

1. 所有输入的历史时间、未来时间、地理范围、真值及历史雨量数组必须相同，否则拒绝合并排名。
2. 对每组结果的每个时效分别计算 CSI/POD/FAR@0.1/0.3/0.5、TP/FP/FN、MSE、MAE，以及复制最后一帧的 CSI@0.5。
3. 默认按 CSI@0.5 降序选三帧；同分按 CSI@0.3 降序、MSE 升序。选择不同有效时刻，同一时刻只取最优候选。因此可能混合 checkpoint，各图明确标出模型与步数。
4. 预测和真值都没有 >=0.5 事件时，CSI@0.5 未定义，不把这种空场记为满分，也不参与选图。
5. 这是依据此案例真值进行的事后筛选，不能作为验证集整体性能，也不能据此宣称某个 checkpoint 全局最佳。图内保留相应说明，完整候选表同时输出。
6. 使用现有原始色阶和最近邻显示，统一提高 DPI、整理边界和排版。不会锐化、平滑、阈值遮盖或改变预测值。保存原始文件 SHA256 以供核查。

## num8 命令

先更新：

```bash
cd /home/yanjieyu/RainPrediction

git pull --ff-only origin project

conda activate rainpred
```

比较此前已生成的这个时间点的案例目录：

```bash
SELECT_OUT="$HOME/RainPrediction/vis_show/case_selection_20250810_$(date +%Y%m%d_%H%M%S)"

python -m src.tools.select_sichuan_cases \
  --cases "$HOME"/RainPrediction/vis_show/sichuan_20250810_* \
  --top-k 3 \
  --dpi 300 \
  --output-dir "$SELECT_OUT"

echo "$SELECT_OUT"
```

通配符会收集已完成的旧模型 80000、50000 步及 A6000 的导出目录。没有完成原始数组导出的目录会打印跳过提示，并记录在 JSON 中；未运行过的模型不会凭空加入。若重复跑过某个模型，会保留其候选记录，但展示帧不会重复选同一时刻。

如果只想从一个模型挑图，把 `--cases` 后替换为该次实际输出目录即可。

## 输出与解释

- `selected_comparison.png` / `.pdf`：筛选出的预测与真值总图，上预测、下真值。
- `selected_01_step..._lead...png/.pdf` 等：单独的预测和真值配对图，含模型、步数、有效时刻、时效及 CSI。
- `all_frame_metrics.csv`：所有候选帧的完整指标，不仅是选中的三帧。
- `selection.json`：选图规则、结果、输入来源和文件哈希。

`csi05_gain_vs_persistence` 为模型相对于“复制最后一帧”的 CSI@0.5 差值，负值表示该帧没有超过这个简单对照。如果最佳候选的 CSI 仍然很低，应如实标为效果有限的案例；排版改善不会提高预测能力。
