# Rain-priority Training Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Implement the approved rain-priority gate and bounded horizon-loss feedback, preserving the original 3D network and four-in/four-out protocol, then publish reproducible server commands.

**Architecture:** New files under `src/rain_priority/` inherit the independent gated model and I/O trainer. Explicit metadata accompanies inputs; explicit gradient routing assigns the rain objective to gate parameters. Existing modules remain unchanged.

**Tech Stack:** Python >=3.12, PyTorch, Accelerate, Hydra, pytest; single-device FP32/BF16 first release.

**Spec:** [Approved design](../specs/2026-10-19-rain-priority-gating-dynamic-loss-design.md), approved by the user on 2026-10-19.

## Global Constraints

- No deletion, no changes to baseline/colleague code, `.gitignore`, or existing diagnostic artifacts.
- Original 3D decoder, seven right-shift targets, four future rollout frames, detached generated history; no future weather injection.
- Keep the known original decoder causality limitation visible. Do not silently fix it.
- Independent entry/config; default batch1, BF16, activation checkpoint, save every5000; full validation in random order.
- Dynamic feedback uses training error only. Gate-specific gradients use rain loss only.
- Optional event/evolution losses remain disabled; no unimplemented configuration pretending to work.

## Review Focus

1. Duplicate context/seed timestamps and generated-input provenance: gate tests must cover both.
2. Accumulated gradients overwritten by auxiliary backward: compare two micro-batches against one combined batch.
3. Controller restored mid-statistics window: compare next update to uninterrupted execution.
4. Validation future labels leaking into rollout: mutate labels and compare all predictions.
5. Accelerator checkpointing/autocast and optimizer skip behavior: real small-model training/save/reload integration test; CUDA test runnable on num8.

### Task 1: Loss and controller

Create `src/rain_priority/loss.py`, `controller.py`, `optimization.py`, and `src/tests/rain_priority/test_objective.py`.

Interfaces: per-lead MSE `[4]`; `HorizonController.observe(error_sum, count, step)` updates detached CPU FP32 state after a successful optimizer step; `state_dict/load_state_dict` support Accelerate checkpointing. Explicit gradient helper preserves previous accumulated gate gradients.

- [x] Write tests with hand-derived sequence/rollout loss, biased-history correction, coefficient invariance of gate gradients, bounded weights and mid-window resume.
- [x] Run tests before implementation; verify missing feature failures.
- [x] Implement the exact spec formulas, fail on invalid/non-finite input, test FP32 and accumulation.
- [x] Run focused tests and record results.

### Task 2: Gate and independent model

Create `src/rain_priority/gate.py`, `model.py`, and `src/tests/rain_priority/test_model.py`.

Interfaces: metadata `source[B,3,T]` uses 0=missing,1=truth-fed,2=generated; `steps[B,T]` identifies input times relative to the forecast origin. `forward_ar` accepts these explicitly. Optional detached gate statistics are returned, not cached on the model.

- [x] Test identity initialization, prefix independence, repeated/gapped times, invalid metadata, same original decoder weights/outputs and checkpointed backward.
- [x] Watch tests fail, implement independent model overrides without modifying the inherited implementation.
- [x] Run new and existing gate/decoder tests; record the existing full-network causality limitation separately.

### Task 3: Trainer, evaluation and executable config

Create `src/rain_priority/trainer.py`, `evaluation.py`, `src/trainer/rain_priority_trainer.py`, `src/config/ts_rain_train/rain_priority.yaml`, and integration tests under `src/tests/rain_priority/`.

Interfaces: inherited data preparation produces unchanged sequence targets; trainer constructs provenance/steps, uses clean loss, and checkpoints the controller. Validation computes per-lead contingency counts and persistence baseline; original tensors drive raw fixed-scale figures.

- [x] Test config resolution, actual tiny training updates, label isolation, gradient accumulation, save/resume, controller isolation from validation and per-lead metrics.
- [x] Implement the new trainer, entry and config, allowing ablations through gate mode, rain-only routing and dynamic-controller switches.
- [x] Run focused/regression suite and repository pytest collection; report unavailable optional dependencies by name.
- [x] Supply CUDA smoke and short-run CLI; do not claim local CPU tests establish 24GB feasibility.

### Task 4: Review, publish and handoff

- [x] Review the complete change using one fresh reviewer as required by executing-plans; fix material findings with regression tests.
- [x] Write `doc/codex_changes/2026-10-19-降雨优先门控与动态损失.md` and server instructions.
- [ ] Check original files unchanged, stage exact task files, commit and push `project` as authorized for server synchronization.
- [ ] Verify remote commit and provide fetch/pull, smoke, short experiment and longer training commands.

## Execution Ledger

- Start: `ba45db8`; user approved implementation and server synchronization. Existing modified `.gitignore` and untracked artifacts/audit are excluded from this commit.
- Ruling: use the current `project` checkout and independent new files, preserving unrelated changes; no deletion-based workspace cleanup. This avoids copying or overwriting the user's untracked spec/audit.
- Ruling: first release explicitly supports a single device with FP32/BF16; unsupported distributed/FP16 modes fail before training instead of silently producing incorrect routed gradients.
- Pre-flight: model metadata feeds trainer sequence/rollout; controller state feeds checkpoint resume; evaluation never updates controller. Interfaces above agree.
- RED/GREEN: objective, model and trainer tests first failed for missing modules, then passed. Review regression cases first failed for missing metadata/time/protocol guards; added guards then passed.
- Final focused verification: 24 passed, 1 CUDA skip; Ruff passed. Expanded regression: 94 passed, 8 skipped, 1 old hard-coded baseline hash failure. Both original files match task-start HEAD exactly. Full repository collection: 59 errors from optional dependencies, missing historical modules and third-party duplicate test names; details in the implementation report.
- Fresh review: `/root/rain_priority_review`, read-only. Important findings: validate actual timestamps (including existing reversed augmentation), and require explicit temporal-gate metadata. Both fixed with tests. Minor diagnostic omissions were also implemented.
- Ruling: preserve original 3D temporal causality limitation as the user explicitly requested unchanged decoder/protocol; gate-prefix and fixed-origin label-isolation tests do not establish complete-network causality.
- Ruling: histogram quantiles use bounded memory with explicit tail counts; out-of-range quantiles are null. Gate-region groups describe input patches, never future targets. These diagnostics do not claim physical rainfall units.
- Ruling: no second long remote training or server process changes during local implementation. GPU feasibility/performance require the supplied smoke and controlled pilot. Controller/optimizer restore is tested on identical subsequent batches; exact dataset cursor replay is not implemented.
- Publication checklist is recorded before commit; completion and remote SHA are reported in the delivery response. Unrelated local files stay unstaged.
