# Six deep baselines implementation plan

**Goal:** train and evaluate ConvLSTM, PredRNN-V2, SimVP, Earthformer, AlphaPre, and NPM from this repository, then commit and push the scoped changes.

**Architecture:** keep the supplied 398-file project and all upstream checkouts unchanged. Independent model adapters and attributed import ports live in `src/baselines_stage3/deep/`. Reuse the existing datasets, common evaluator, and final Stage 3 experiment settings. Five multimodal methods use a seven-slot context (left-zero padding while fewer observations exist), one-step cores, and four-step rollout with causal auxiliary observations. NPM is a separate strictly satellite-only, fixed-origin 4-to-4 experiment using its forecaster and satellite-to-rain translator.

**Tech stack:** Python >=3.12, PyTorch, Hydra/OmegaConf, existing litdata loaders, pytest. CPU synthetic verification here; actual full-resolution GPU training on the server.

**Spec:** user requests all six remaining table baselines, parameter/layer alignment to the 44,162,136-parameter reference, original source preservation, current loss weights, detach_rain_anchor=false, GT auxiliary=true, 70,000 steps. Latest delivery instruction is commit and push, not ZIP.

## Constraints and decisions

- No deletions, original source rewrites, runtime monkeypatches, or unscoped Git staging. Existing `.gitignore` edits and unrelated artifacts remain untouched.
- Only historical rainfall is available at forecast origin. Observations at forecast step k enter the history after predicting k, never before. Future rain is never an inference input.
- Teacher-forced sequence supervision uses three historical transitions and four future transitions; modality weights 1/1/2, context/future weights 1/1. Rollout branch is rain MSE with weight 0.5. Input history is detached, residual rain anchor is not detached.
- Native architecture-specific regularizers remain explicit (PredRNN-V2 decoupling, AlphaPre spectral/branch supervision). Reports distinguish them from the shared prediction loss.
- All trainable baselines start from their own random initialization; reference Stage 2 weights cannot initialize another architecture. Default optimization follows the provided final config unless explicitly overridden.
- Target total parameters within 10% of reference, report actual totals and active/unused parameters where relevant. No artificial unused layers to inflate capacity.
- NPM uses ten satellite channels and precipitation labels; neither radar nor observed rainfall enters its input. Adapted end-to-end supervision is documented separately from the original two-stage GAN recipe.
- Single GPU/process; reproducible seeds, checkpoint/optimizer/scheduler/EMA/RNG restore, no automatic checkpoint deletion. Non-stateful data iterator resume is documented.

## Tasks

- [x] OpenSTL domain: real ConvLSTM, PredRNN-V2, original IncepU SimVP adapters; pinned provenance; CPU forward/backward tests and parameter profiles. Files `deep/openstl.py`, owned ports, `test_openstl.py`.
- [x] Earthformer/AlphaPre domain: cuboid attention and amplitude-phase implementations; stable FFT float32 path; supervised regularizers; tests and profiles. Files `deep/earth_alpha.py`, owned ports, `test_earth_alpha.py`.
- [x] NPM domain: official satellite predictor and translation generator, time features and ten-channel adaptation; satellite-only isolation; tests and parameter profiles. Files `deep/npm.py`, owned ports, `test_npm.py`.
- [x] Shared training domain: registry/configuration, padding/masking, sequence and rollout objective, train/eval entry points, six configs, checkpoint and resume. Files `deep/common.py`, `deep/training.py`, `train_deep.py`, `evaluate_deep.py`, config group and `test_deep_training.py`.
- [x] Integrate independent model tests with actual optimizer/EMA/resume and common evaluation; verify source hashes and parameter profiles; review causality, initialization, data coverage and resume.
- [x] Update server commands and tracked provenance; stage only explicit task paths and inspect staged diff. Git delivery uses the current branch; no code package.

## Review focus

1. Future rainfall poisoning and auxiliary-arrival ordering must not affect earlier predictions.
2. NPM must remain independent of radar, past/future rainfall, and future satellite observations.
3. Reduced-size tests must use the actual architectures; full-size parameter profiles must instantiate the final configs.
4. Model-specific non-scalar branch outputs must not accidentally enter generic scalar loss summation.
5. Resume must preserve EMA and step budget, reject architecture/config mismatches, and never delete checkpoints.

## Execution record

- Git verified at `D:/rain-prediction`, branch `ryan/baseline-evaluation`, origin `https://github.com/ryandegl233/rain-prediction.git`. Earlier no-Git observation is superseded by current evidence.
- Applied writing-plans, dispatching-parallel-agents, and requesting-code-review skills. Three independent model domains run in parallel; parent owns integration and delivery. User already authorized implementation and push, so no additional approval gate is needed.

- Final verification: 57 pytest tests passed; two CLI train/evaluate smoke cases passed. Four review findings addressed and covered by regression tests. No full real-data GPU training claimed.
- Ruling: preserve original scheduler behavior; its min_lr is a relative multiplier. Document the actual floor instead of silently changing the supplied method.
- Ruling: NPM uses a fixed-origin satellite-only protocol and absolute native rainfall output; it is kept separate from the GT-updated multimodal table.

- Portability verification: clean checkout made only from staged files, with no external upstream clones: 55 passed, 2 optional source-comparison tests skipped. Original 398 source blobs remain byte-identical in the Git index.
