# Independent Gated Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 将第一版门控模型与 trainer 迁入独立副本，保护原 baseline，保持已有算法与 CLI。

**Architecture:** 在 `src/gated_modality_rain/` 定义独立模型与 trainer，完整复制已验证第一版的核心实现。现有门控入口、配置、测试指向新包；原 baseline 文件恢复为 `9b02789` 并受测试保护。模型、trainer、配置构成一次原子迁移，因此本计划为一个完整交付任务。

**Tech Stack:** Python、PyTorch、Hydra、Accelerate、pytest、Ruff。

**Spec:** `doc/superpowers/specs/2026-09-03-independent-gated-framework-design.md`

## Global Constraints

- 原 baseline／他人代码必须保持不变；需要扩展或修复其中某个组件时，先新建独立文件，再让我们的框架引用新文件。
- 模型与 trainer 副本以 `6c67b56` 中已通过服务器相关回归的第一版内容为迁移来源。
- 新的公共类名分别为 `GatedModalityRainModel`、`GatedModalityRainTrainer`，类实现位于新包内，不继承、重导出或别名指向原 `RainCausalPatchTransformerNextFrame`、`RainTSNextFrameTrainer`。
- 保留所有模型参数的 state-dict 键名、形状、初始化顺序及门控开关语义；模块路径与公共类名变化不能引入参数名前缀。
- 保持 PyTorch CUDA、bf16 autocast、activation checkpoint 和正常反传路径，不新增 CPU 专用路径、不硬编码设备、不放宽数值比较容差。
- 本次不删除数据、不覆盖用户 `.gitignore` 或其他未提交修改、不推送、不启动训练。
- 在用户已批准的 `D:/rain-prediction` 工作目录内执行；保留此前已完成但未提交的原文件恢复与 `AGENTS.md` 保护规则。

---

### Task 1: 独立副本、入口迁移与回归

**Files:**

- Create: `src/gated_modality_rain/__init__.py`
- Create: `src/gated_modality_rain/model.py`
- Create: `src/gated_modality_rain/trainer.py`
- Create: `src/tests/gated_modality_rain/test_framework_isolation.py`
- Modify: `src/trainer/gated_modality_rain_trainer.py`
- Modify: `src/config/ts_rain_train/gated_modality_rain_trainer.yaml`
- Modify: `src/tests/time_series/test_spatial_modality_gate.py`
- Modify: `src/tests/trainer/test_gated_modality_rain_trainer.py`
- Include existing restoration, without further edits: `src/networks/time_series/causal_patch_transformer_next_frame.py`, `src/trainer/rain_trainer_ts_next_frame.py`
- Include existing policy update, without unrelated edits: `AGENTS.md`
- Controller documentation: update `doc/codex_changes/2026-09-03-gated_modality_rain_trainer第一版功能总结.md` and add `doc/superpowers/validation/2026-09-03-independent-gated-framework.md` after independent verification.

**Interfaces:**

- Produces: `src.gated_modality_rain.model.GatedModalityRainModel` with the exact constructor and forward interfaces from the v1 model, including gate disabled by default.
- Produces: `src.gated_modality_rain.trainer.GatedModalityRainTrainer(cfg: DictConfig)`, its full training methods, and its own `apply_context_modality_dropout` helper.
- Consumes unchanged: `src.utils.gated_checkpoint.load_gated_model_initialization`, shared network building blocks, data/loss/logging utilities.
- CLI unchanged: `python -m src.trainer.gated_modality_rain_trainer`; the script entry also remains valid.
- Test helper `build_model(**overrides)` becomes the independent model; add `build_baseline_model(**overrides)` to construct the real original baseline with the same tiny settings. All baseline comparison sides use the latter.

- [ ] **Step 1: Add failing isolation tests before production copies.**

Start with a missing-package assertion inside a test (not collection failure), then assert class/module ownership and baseline content protection. Use normalized Git blob hashing so Windows CRLF does not produce false positives:

```python
import hashlib
from pathlib import Path

def normalized_git_blob(path: Path) -> str:
    content = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()

def test_independent_framework_package_exists() -> None:
    root = Path(__file__).resolve().parents[3]
    assert (root / "src/gated_modality_rain/model.py").is_file()
    assert (root / "src/gated_modality_rain/trainer.py").is_file()
```

Protect original source blob IDs `7a60fbf3da6723eb220cb9c793dbd091318b35b9` and `d81d4b13db01c98af7c1a00c881784797b2da485`. Assert new classes are not old classes/subclasses, model helper classes live in the new module, and every trainer method is independently defined. Add real loss/gradient parity tests against the original trainer with nontrivial modality tensors and weights; compare actual loss/log tensors and gradients, not mock return values. Add a real optimizer-step parity test using original/new disabled models with identical states and identical small batches, and independently constructed trainer objects, without full dataset initialization. Mirror the original trainer test setup fields needed by `train_step`, but do not mutate old test files or their globals.

- [ ] **Step 2: Record the expected RED.**

Use `D:/rain-prediction/.superpowers/sdd/2026-09-03-gated-modality-rain-trainer/venv/Scripts/python.exe` with two torch CPU threads. Run only the new isolation test file first. Record the failing assertion that independent files are absent, and the passing original-content guards. Do not install packages or modify the existing environment.

- [ ] **Step 3: Copy the production bodies with controlled mechanical transformations.**

Read `git show 6c67b56:src/networks/time_series/causal_patch_transformer_next_frame.py` and `git show 6c67b56:src/trainer/rain_trainer_ts_next_frame.py`. Generate the new files with `apply_patch`; source strings can be captured from read-only Git commands using the functions JavaScript store. Preserve the core statements and parameter registration order.

```javascript
function modelCopy(source) {
  const normalized = source.replace(/\r\n/g, "\n");
  const anchor = '\nif __name__ == "__main__":';
  if (!normalized.includes(anchor)) throw new Error("missing demo boundary");
  return normalized.split(anchor)[0].trimEnd()
    .replace("class RainCausalPatchTransformerNextFrame(nn.Module):", "class GatedModalityRainModel(nn.Module):") + "\n";
}
function trainerCopy(source) {
  const normalized = source.replace(/\r\n/g, "\n");
  const anchor = "\n@hydra.main(";
  if (!normalized.includes(anchor)) throw new Error("missing CLI boundary");
  return normalized.split(anchor)[0].trimEnd()
    .replace("class RainTSNextFrameTrainer:", "class GatedModalityRainTrainer:") + "\n";
}
```

Read the copied code and check exact class replacements occurred. Remove entry-only imports and import-time CLI setup from the pure trainer module if no core method uses them; use direct existing dependencies instead of copying fallback compatibility classes. Preserve functional trainer method bodies. English comments and modern annotations apply to the new files; do not reformat original source. Keep `_SpatialModalityGate` and other private model helpers in the independent model file. Keep checkpoint loading in the independent trainer; do not duplicate the loader.

- [ ] **Step 4: Move our entry/config/tests to the independent classes.**

```python
from src.gated_modality_rain.trainer import GatedModalityRainTrainer

@hydra.main(config_path="../config/ts_rain_train", config_name="gated_modality_rain_trainer", version_base=None)
def main(cfg: DictConfig) -> None:
    trainer = GatedModalityRainTrainer(cfg)
    trainer.run()
```

```yaml
rain_prediction_model:
  _target_: src.gated_modality_rain.model.GatedModalityRainModel
  spatial_modality_gate_enabled: true
  spatial_modality_gate_hidden_channels: 32
```

In model tests, use real original baseline for neutral/disabled equivalence and all four CUDA precision/grad-mode comparisons. Keep strict `rtol=0, atol=0` under identical grad modes. Preserve two CUDA backward tests and all existing gate behavior cases. In trainer tests, checkpoint baseline sources remain original models; gated sources and targets use the new model. Adjust configuration target expectations and trainer monkeypatch paths. Fixed-origin baseline rollout tests instantiate original trainer/model; gated tests instantiate independent trainer/model. Preserve all eight checkpoint representations, incomplete/corrupt checkpoint rejection, and future-target perturbation tests.

- [ ] **Step 5: Verify the completed migration and self-review.**

Run the two updated test files, two unchanged original test files, and new isolation file in a single pytest invocation using two torch threads and `-q -rs -p no:cacheprovider`. Run Ruff and syntax compilation on new/changed Python files only. Also run Ruff `--select F821` on the independent production modules to check names used by copied methods after import cleanup. Verify module and script `--cfg job` paths via existing entry tests and exercise real logger setup in a temporary-directory subprocess; no real training. Review all source-copy AST differences against `6c67b56`, allowing public class rename, removed demo/main/setup and annotations, but no changed model/trainer core logic. Run:

```bash
git diff 9b02789 -- src/networks/time_series/causal_patch_transformer_next_frame.py src/trainer/rain_trainer_ts_next_frame.py
git diff -- src/tests/time_series/test_causal_patch_transformer_next_frame.py src/tests/trainer/test_rain_trainer_ts_next_frame.py
git diff --check
```

Both content comparisons must be empty. Record actual counts; the local environment is CPU-only, so CUDA skips are not GPU passes. Record existing Torch JIT warnings without suppressing them. No real data/weights, training, DDP or metric claims.

- [ ] **Step 6: Commit the scoped deliverable, then request controller review.**

Stage only the named production/test files plus the already-restored original source and `AGENTS.md`. Do not stage `.gitignore` or controller-owned docs. Commit with `refactor: isolate gated modality model and trainer`. Write the report with RED/GREEN commands/output and self-review findings in the task workspace; the controller will independently review, update documentation and leave everything unpushed.
