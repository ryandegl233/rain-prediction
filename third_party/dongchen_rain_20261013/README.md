# 同学提供的 RainPrediction 源码快照

这份目录用于保存和对照同学提供的代码包。现有仓库的 `src/` 是主框架；本目录中的 `src/` 是独立副本。本次导入没有把副本接入主框架的训练入口，也没有覆盖主框架的模型、trainer 或配置。

## 来源和完整性

- 来源文件：`rain.zip`，于 2026-10-13 导入。
- 压缩包 SHA-256：`b831c5658579279fc47dffd11e1973273b2ea5571cc0a4ad7278987775562c02`。
- 对照的主仓库提交：`c25ec88f6da94916f882170e7cdb5fa01aaa9048`。
- 路径映射：压缩包内 `rain/<路径>` 原样复制到本目录的 `src/<路径>`。
- 保存 367 个源码及附属文件，合计 4,808,800 字节；其中 322 个 Python 文件通过按 Python 3.12 语法规则进行的解析检查。
- 排除 `__MACOSX`、`._*`、`.DS_Store`、`__pycache__`、`.pyc`、`.pyo`、`.git` 和 `.idea`；不包含数据集、训练权重或新安装的依赖。
- 每个文件的来源路径和 SHA-256 记录在 `SOURCE_MANIFEST.json`。本目录的 `.gitattributes` 禁用副本源码的换行转换，使 Windows 和 Linux 拉取后仍能核验原始字节。

## 本次关心的文件

- `src/trainer/rain_trainer_ts_next_frame_transfer.py`：同学的 transfer trainer 和 rain-only decoder。
- `src/trainer/rain_trainer_ts_next_frame.py`：与 transfer trainer 配套的基类版本。
- `src/config/ts_rain_train/rain_trainer_ts_next_frame.yaml`：同学配套的实验配置。
- `src/tests/trainer/test_next_frame_transfer.py`：同学提供的传输和训练结果对照测试。

这份代码包相对于主仓库提交有 339 个相同文件、10 个新文件和 18 个不同文件，比较时统一了 CRLF/LF 换行。不同文件不只有 I/O 实现：模型副本没有主仓库已提交的 CCC 接入，trainer 的 loss 和 checkpoint 管理也有变化。

配套 YAML 将训练 batch size 从 8 改为 2，将数据筛选改为 rain-ratio，并调整了 rollout 权重及多个 loss 开关。它还设置了同学账号下的 `train.resume_path`，不能直接作为新服务器的正式训练配置。

## 同步到新服务器

在 num8 的已有仓库内运行：

```bash
cd /home/yanjieyu/RainPrediction
git status --short --branch
git -c merge.autostash=false pull --ff-only --no-rebase origin project
git log -1 --oneline
```

如 Git 因本地改动或分叉拒绝拉取，应保留现场并检查输出，不强制 reset、stash、覆盖或清理文件。

从主仓库根目录核验独立副本：

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path

snapshot = Path("third_party/dongchen_rain_20261013")
manifest = json.loads((snapshot / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
for record in manifest["files"]:
    path = snapshot / record["path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"], str(path)
print(f"Snapshot verified: {len(manifest['files'])} files")
PY
```

## 后续整合范围

保留主框架中的 `GatedModalityRainModel` 和已提交的创新点，将核验后的传输优化接入我们自己的独立 trainer 和配置。不要把这份副本整体覆盖到主仓库的 `src/`，也不要修改同学的副本来冒充原始来源。

这次只确认源码导入及字节完整性。没有执行真实数据训练、同学的训练脚本、checkpoint 清理逻辑或完整功能测试；语法检查不代表这些行为已经验证。
