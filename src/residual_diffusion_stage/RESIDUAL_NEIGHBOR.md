# Preserved original model with an independent correction

This experiment addresses the initial-output shift in the previous replacement experiment.
The original pointwise cross attention remains active. The model returns
`base_tokens + old_gate * old_pointwise_attention + new_gate * new_correction` before the original local branch.
All original parameters, including the pointwise adapter, local branch and decoders, are frozen.
Only the 888,961 parameters under `cross_modal_adapter.correction` train.
Frozen decoders still transmit gradients to the correction; decoding is not wrapped in `no_grad`.

The correction copies the trained history encoders, LayerNorm and Q/K/V projections from the original
pointwise module after checkpoint loading. Its output projection starts at zero, and its effective gate
starts at 0.1 (stored as atanh(0.1)). Initial correction is exactly zero. The output projection receives
gradient on the first update; upstream attention and modal encoders can receive gradient after it changes.
The trainer audits all correction gradients for the first two updates and requires nonzero output/QKV
and radar/satellite/rain adapter gradients. Frozen original weights are hashed at every validation.

Both arms start from `runs/rollout_dry_pro6000_0Q6klO/dry_guarded/checkpoints/checkpoint_1` with fresh optimizers.
`point_correction` adds a same-position correction; `neighbor_correction` adds a 3x3 correction.
They have identical parameters, initial predictions, loss, data order and training budget; only history
search radius differs. These controls separate extra adaptation capacity from the benefit of neighborhood
search. The immutable original model is represented by the shared step-zero results and images.

The existing guarded dry loss, teacher-forced loss, rollout weight 0.2, detached predicted history, and
alpha=0 are retained. No extra relative-position/time/provenance encoding is introduced in this round;
its effect should be tested separately. This design preserves the starting forecast, not a guarantee
that learned corrections cannot worsen it. Initial identity and frozen weights are auditable properties;
forecast skill and generalization require validation.

## Run on physical GPU 2

```bash
cd /home/rainpred/RainPrediction
tmux new-session -s residual_neighbor_pro6000 \
  'bash src/residual_diffusion_stage/ablate_residual_neighbor_pro6000.sh'
```

Default: two sequential arms, each 500 optimizer updates, batch 1, accumulation 8, validation every 100
updates on the same 60 development windows. This is a learning-curve experiment, not a claim that 500
updates are sufficient. It starts fresh and does not implement exact checkpoint resumption.
The PRO 6000 is selected by UUID; logical cuda:0 is physical GPU 2. The launcher checks >=18 GiB free
as a conservative guard and creates a new `runs/residual_neighbor_pro6000_*` directory.

Results: `comparison.md`, `comparison.json`, `comparison_index_*.png` (GT / point / neighbor),
`baseline_vs_neighbor_*.png` (GT / original / neighbor), per-arm `constraint_selection.jsonl`,
`gradient_audit.jsonl`, and all validation rain/change plots. Both initial reports and saved arrays must
match exactly or comparison export fails. Only candidates passing the original CSI/POD/intensity checks
and reducing long-lead FAR receive `best_eligible.json`.

The samples are rain-filtered, the seed is fixed, and the development set has been used repeatedly.
Independent weather events/months, including dry and mixed cases, are necessary before concluding robust gain.

Small pipeline check:

```bash
bash src/residual_diffusion_stage/ablate_residual_neighbor_pro6000.sh --steps 2 --val-every 2 --samples 2
```
