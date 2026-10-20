# Rain-local fusion implementation plan

> **For agentic workers:** Use superpowers:executing-plans for inline execution and a fresh final review. Track checks below.

**Goal:** Implement controlled A0/R0/R1 experiments without modifying the baseline, decoder, data, or loss.

**Architecture:** Subclass RainPriorityModel in an independent package. Reuse its encoder, temporal blocks, rollout, and trainer; route a small spatial residual exclusively into the existing rain decoder.

**Tech Stack:** Python >=3.12, PyTorch, Hydra, Accelerate, pytest.

**Spec:** `doc/superpowers/specs/2026-10-20-rain-local-fusion-design.md`

## Global constraints

- Original 3D decoder; four past/four future frames at 30-minute intervals.
- A loss, fixed q=0.25, no new learned gate, no label replacement.
- Add independent files only; preserve all unrelated working changes and data.
- CPU implementation tests do not establish 24GB CUDA fit or forecast improvements.
- User authorized implementation after reviewing the minimal design and actual label sample statistics on 2026-10-20.

## Review focus

- Source masking must remove projection biases and handle missing auxiliary inputs.
- TF duplicated first seed and AR duplicated last seed must select the matching features.
- Added initialization must not perturb shared weights or the following RNG state.
- Each decoder runs once; auxiliary losses cannot update the fusion branch.
- Variants, optimizer state, and fixed horizon weights must survive checkpoint reload; incompatible variants must fail.

## Task 1: Residual operator and model integration

**Create:** `src/rain_local_fusion/fusion.py`, `model.py`, `__init__.py`.
**Tests:** `src/tests/rain_local_fusion/test_model.py`.
**Interfaces:** `RainLocalResidual(channels, auxiliary_channels, hidden_channels=32, kernel_size=3)` maps query/radar/satellite `[B,C,T,H,W]` and availability `[B,2,T]` to a rain-feature residual; `RainLocalFusionModel` implements RainPriorityModel's existing forward/forward_ar API.

- [x] Write and run failing tests for zero residual, masks, local spatial support, no temporal mixing, baseline/RNG equivalence, head isolation, gradients, source-index mapping, and checkpoint reload.
- [x] Implement operator with independent 1x1 projections, depthwise k=1/3 convolutions, replicate padding, SiLU mixing and zero output projection.
- [x] Implement model routing in a new subclass; reject unsupported decoder/adapter/gate settings instead of silently changing the experiment.
- [x] Run focused tests to green.

## Task 2: Controlled configurations and training integration

**Create:** `src/rain_local_fusion/trainer.py`, `src/trainer/rain_local_fusion_trainer.py`, `src/config/ts_rain_train/rain_local_fusion{,_a0,_r0}.yaml`.
**Tests:** `src/tests/rain_local_fusion/test_trainer.py`.
**Interfaces:** `RainLocalFusionTrainer(cfg)` reuses RainPriorityTrainer and validates the clean experiment settings.

- [x] Write failing configuration and real tiny-training tests.
- [x] Add 6000-step configs with batch1/accumulation2, scheduler horizon100000, fixed q, full validation every2000, checkpoints every2000.
- [x] Verify A0 loss/gradient/update equality against A; neutral R1 outputs and shared gradients within stated precision; four-frame rollout, fixed q, checkpoint resume, visualization/JSON output.
- [x] Add CUDA BF16 tests with explicit skips when CUDA is unavailable.

## Task 3: Review and delivery

**Create:** `doc/experiments/2026-10-20-rain-local-fusion-num8.md`.

- [x] Update audit/spec with actual ten-frame label evidence and remaining limits.
- [x] Run new tests plus rain_priority regressions, syntax/config/lint checks.
- [x] Fresh reviewer examines new code, interfaces, and experiment controls; fix material findings with regression tests.
- [x] Prepare separate server CLI blocks for pull, tmux, smoke, and sequential A0/R0/R1 experiments. No automatic remote training without credentials.
- Delivery: commit only task-owned files and push to the existing project remote; report the verified commit in the user handoff.

## Execution ledger

- Start: base `6b15f72`; existing `.gitignore`, artifacts, and prior untracked audit file are unrelated.
- Ruling: work in the user's current `project` checkout using independent new files, avoiding movement of unrelated uncommitted work. No baseline files or data will be removed.
- Ruling: continue implementation now under the user's explicit instruction; no further plan confirmation is needed.
- Ruling: actual grid and rain time semantics remain unresolved; controlled pilot uses the same existing labels for every group and does not claim physical alignment or causality has been repaired.
- Task 1: RED missing module observed; initial 15 operator/model tests GREEN. Task 2: RED missing trainer observed; expanded suite 20 passed/1 CUDA skip.
- Final review: fresh read-only reviewer approved current batch1 scope and identified BF16 batch2 decoder-layout drift. Regraded important for the documented batch2 option and neutral-control validity.
- Final fix: reproduced changed strides (2048→1792), then preserved the original strides with differentiable copy. CPU BF16 layout/output regression is GREEN; CUDA equality remains explicitly skipped locally.
- Ruling: neutral R1 shared backward gradients use FP32 tolerance rather than bitwise equality. Extra graph/layout operations alter summation rounding: measured max absolute error 2.61e-8, relative L2 1.11e-7. A0 still requires exact loss/gradient/update equality. Cost: no claim of bitwise-identical R1 training; intended learned trajectories and global clipping differ.
- Review boundaries retained: CUDA full-size memory/speed, actual grid/time semantics, physical offsets and forecast/multiseed gains require server evidence. No deferred minor findings remain after the layout fix.
- Final verification: `pytest src/tests/rain_local_fusion src/tests/rain_priority -q -p no:cacheprovider --import-mode=importlib` → 45 passed, 3 CUDA skips. Existing torch.jit deprecation warnings only. Ruff checks passed; resolved new CLI config retains 3D/4→4, fixed q, batch1/accum2, 6000 pilot steps and 100000-step main scheduler.
