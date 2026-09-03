# 独立门控框架迁移验证

_RainPrediction · 2026-09-03 · 独立副本迁移的结构证据与服务器复测边界_

---

## 📋 迁移范围

依据[已批准的设计](../specs/2026-09-03-independent-gated-framework-design.md)与[实施计划](../plans/2026-09-03-independent-gated-framework.md)，模型和 trainer 主体迁入 `src/gated_modality_rain/`，既有门控 CLI 与配置切换到独立类。新旧框架只读共用基础网络积木、数据和 loss 工具；原 baseline 不接入我们的门控。

本次只做结构迁移与相关回归，不增加强降水监督、gate loss、空间 mask、VAE 或扩散模块，不运行真实训练，不推送。参数键名、初始化顺序、batch size（训练 8、验证 4）、梯度累积、loss 和固定起报设置保持第一版契约。

## 🔐 原文件保护

| 原文件 | 恢复后 Git blob |
| --- | --- |
| `src/networks/time_series/causal_patch_transformer_next_frame.py` | `7a60fbf3da6723eb220cb9c793dbd091318b35b9` |
| `src/trainer/rain_trainer_ts_next_frame.py` | `d81d4b13db01c98af7c1a00c881784797b2da485` |

两文件与 `9b02789` 的 Git 内容比较为空。新增测试按 LF 标准化换行后校验 blob，避免 Windows／Linux 换行误报。原模型和 trainer 的两份测试、原 YAML、原模型配置、fixed-origin 配置、受检加载工具及用户 `.gitignore` 另做内容指纹比较；它们不在本次功能编辑范围内。

```bash
git diff 9b02789 -- src/networks/time_series/causal_patch_transformer_next_frame.py src/trainer/rain_trainer_ts_next_frame.py
```

预期无内容差异。Git 可能输出本机 CRLF 提示；该提示不是源码差异。

## 🔍 副本审计

独立副本来自 `6c67b56`。控制端对完整模型／trainer 类和文件内辅助定义执行 AST 比较，忽略类型注解、docstring 与格式，不忽略实际计算语句。模型 12 个方法与 7 个辅助定义、trainer 45 个方法与 dropout 函数的核心实现一致。

模块组织差异限定为公共类重命名，移除旧演示和默认启动入口，以及新 trainer 去掉仅为旧入口服务的 `sys.path`、`colored_traceback` 等导入副作用。新 trainer 从 `accelerate.utils.deepspeed` 直接引用现有 `DummyOptim`／`DummyScheduler`，没有复制本地 fallback 类。此改动源于本机 Accelerate 顶层只在检测到 DeepSpeed 时导出这两个类；未安装或升级任何依赖，也未修改原 trainer。

训练方法只补必要返回类型注解。门控算法、融合公式、CUDA／BF16／activation-checkpoint 运算和严格数值断言均保留。此迁移不解决原 decoder 的时间因果性风险，相关历史诊断见[第一版验证记录](2026-09-03-gated-modality-rain-trainer.md)。

## ✅ 本地验证结果

本地使用 Python 3.13、CPU PyTorch 环境；以两个 CPU 线程执行指定五文件测试，不运行整个仓库或真实训练。

| 验证 | 结果 |
| --- | --- |
| 首次隔离测试 RED | 1 failed、1 passed；独立文件尚不存在，原文件保护通过 |
| 最终控制端完整相关回归（`45a72bb`） | 104 passed、6 skipped、21 warnings，114.46 秒 |
| 新旧模型／trainer 对照 | 权重加载、预测、loss、梯度和 SGD 参数更新零容差对照通过 |
| 两种门控 CLI `--cfg job` | 模块／脚本模式均通过，不启动训练 |
| 真实 logger 初始化回归 | 子进程日志／配置／TensorBoard 初始化通过，父进程日志 sink 保持 |
| Ruff 项目规则及额外 `--select F821` | 通过 |
| Python 语法编译、Git 差异检查 | 通过 |

六项跳过全部是 CUDA 等价性和反传用例，本机跳过不算 GPU 通过。21 条警告来自既有 Torch JIT 弃用提示，未通过全局屏蔽隐藏。

独立审查发现首轮测试没有覆盖真实日志初始化：清理导入时误移除了 `_configure_logger` 使用的 `sys`。新增回归先复现 `NameError`，随后仅在我们的 trainer 中补回 `import sys`。修复提交 `3d26509`，针对性覆盖集为 38 passed，之后执行上表中的完整相关回归；核心训练方法没有因此改变。

整体审查进一步发现进程内日志测试会清空 pytest 的全局 Loguru sinks。`45a72bb` 仅修改测试：真实初始化移入子进程，父进程增加 sentinel 验证既有日志输出仍可用，清理只移除测试自行新增的 sink。先复现 sentinel 消失，再得到隔离测试 6 passed；修复复审通过，最后重跑全部 110 个相关用例，得到上表结果。无遗留阻断审查项。

## 🔧 服务器复测命令

本地实现提交为 `11750ac`（独立副本）、`3d26509`（日志初始化修复）和 `45a72bb`（测试进程隔离）。

本次没有推送。以下命令只能在本地提交已另行发布到 GitHub 后，用服务器的 `yanjie` 环境执行。服务器的远端名是 `myorigin`；若拉取因本地修改或分叉拒绝，停止并检查，不覆盖、不强制重置、不清理数据。

```bash
cd ~/RainPrediction
git status --short
GIT_ASKPASS= git -c credential.helper= -c merge.autostash=false pull --ff-only --no-rebase myorigin project
```

确认拉取成功后先检查配置，不启动训练：

```bash
python -m src.trainer.gated_modality_rain_trainer --cfg job
```

模型目标应为 `src.gated_modality_rain.model.GatedModalityRainModel`。

GPU 专项复测保留原来的 6 个用例，比较侧是真实原 baseline：

```bash
python -m pytest src/tests/time_series/test_spatial_modality_gate.py -k cuda -v -rs -p no:cacheprovider
```

完整相关回归增加独立框架测试：

```bash
python -m pytest \
  src/tests/time_series/test_spatial_modality_gate.py \
  src/tests/time_series/test_causal_patch_transformer_next_frame.py \
  src/tests/trainer/test_gated_modality_rain_trainer.py \
  src/tests/trainer/test_rain_trainer_ts_next_frame.py \
  src/tests/gated_modality_rain/test_framework_isolation.py \
  -q -rs -p no:cacheprovider
```

## ⚠️ 证据边界

本地仅有 CPU，迁移后的 CUDA 路径需要服务器重新执行。迁移前用户日志中的 `104 passed` 与 `6 passed` 是历史证据，不转记为本次结果。真实 checkpoint、真实数据训练、448×448 的显存吞吐、多卡 DDP 和预测指标改善均未在本次迁移中验证。
