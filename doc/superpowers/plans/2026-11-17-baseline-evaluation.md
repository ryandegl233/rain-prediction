# 独立 baseline 评测与训练接入

依据：`doc/baselines/2026-11-17-source-and-alignment-audit.md`。用户已批准先实现 Persistence／LK 与统一接口，并接入指定 YAML 对应训练流程。

约束：只增加独立文件；原 trainer、模型、上游源码不修改。4→4，30 分钟，默认 448×448，输入顺序 radar/satellite/rain=1/10/1。推理只获得历史输入。沿用项目原始 rain_interpolated 监督量纲。

执行：本轮直接完成实现、验证、独立审查和服务器交付；在 `ryan/baseline-evaluation` 分支仅提交本任务文件。

## 接口与文件

1. `src/baselines/contracts.py`：`HistoryBatch` 和 `prepare_history`，将数据 batch 转换为历史条件，缺失模态清零并保留三组 availability。未来张量不进入该对象。
2. `src/baselines/classical.py`：`Persistence`／`LucasKanade`，统一 `forward(history) -> [B,1,4,H,W]`，缺失 rain 拒绝预测。LK 固定安装稳定发行版 `pysteps==1.21.5`，使用其 ndarray 光流和半拉格朗日外推接口；已下载主分支保留作源码参考。
3. `src/baselines/reference.py`：包装原网络公开 `forward_ar`，单步自回归四次，仅回填自身预测，处理原 rain residual 和已知时刻。
4. `src/baselines/evaluation.py`、`evaluate.py`：一遍数据评测多个模型／模态组合，复用 `RainLeadMetrics`，记录逐时效和全局事件计数、样本数、窗口摘要、配置与来源，保存 JSON/CSV。
5. `src/baselines/trainer.py`、`train.py` 与 `src/config/ts_rain_train/baselines_*.yaml`：独立 subclass 沿用原训练与 loss，在验证阶段运行公共 evaluator；配置与原 YAML 同目录以继承现有 Hydra defaults；本轮明确支持单进程单 GPU。
6. `scripts/baselines/fetch_sources.py`、独立运行依赖和服务器说明：依据固定 commit 获取五个源码仓库，不修改已存在的脏仓库、不删除文件；运行 LK 从 requirements 安装稳定版，不从主分支导入。

## 验证步骤

- 先写 `src/tests/baselines/` 中契约、Persistence、LK 平移、指标批次划分不变性、未来真值隔离、训练集成测试，运行确认新增模块不存在。
- 实现后运行同一组 pytest，真实调用 pySTEPS 1.21.5；补测小型原网络四步前向和一次 backward／optimizer update、checkpoint 保存和严格重载。
- CLI 解析真实 Hydra defaults；合成数据完成端到端报告；训练 smoke 运行真实原 trainer，合成 dataloader 仅用于测试，不作为真实数据训练结果。
- 检查既有源文件与上游 Git 状态，检查提交清单，独立 reviewer 审查后修复问题。

## 关键风险的具体检查

- 修改未来雷达／卫星／降雨后，模型条件与预测不变。
- rainfall 缺失的组合显式 skipped，不能调用 persistence 或从目标取最后一帧。
- 无雨、静止、平移和出界场保持有限；出界填 0，不能用真实未来填充。
- 最后不满 batch 仍统计；全干样本无事件指标保存 null，JSON 不包含 NaN。
- 原 checkpoint 的加载保持 strict；评测标明 online 或 EMA；原训练入口默认 YAML 名称不同，新的入口显式继承用户指定配置。

## 完成记录

2026-11-18：上述独立模块、配置、源码获取脚本和服务器说明均已完成。19 项 pytest 全部通过（26.58 秒），包括真实 LK 出界平移和原 trainer 的梯度累积、更新、完整评测、checkpoint 严格重载；独立 CLI 使用 3 个合成窗口生成报告。Ruff 检查通过。独立 reviewer 两轮只读审查后，checkpoint 命名及缺失历史/预测回填问题均已处理。五个上游 checkout 保持固定提交和干净状态；原 trainer、network、原配置没有修改。真实数据与完整尺寸 GPU smoke 交由服务器按运行说明执行。
