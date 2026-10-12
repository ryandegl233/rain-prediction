# 3x3 history cross-attention ablation

This experiment varies only the spatial search radius of cross attention. Standard multi-head attention,
LayerNorm, the local spatial MLP, the global residual gate and every checkpoint tensor are retained.
No GQA, RMSNorm, SwiGLU, additional positional embeddings or reliability gating are added in this round.
The baseline files are unchanged; the independent model subclass selects its conditioner at construction.

Each target patch queries all history frames and three modalities at its own position (`pointwise`, radius 0)
or in a 3x3 patch neighborhood (`neighborhood_3x3`, radius 1). With four history frames this is 12 versus
108 history tokens per target. Invalid boundary neighbors are masked, not wrapped or counted as zeros.
There is no explicit offset/time identity encoding added to history K/V; this is a search-support ablation,
not a complete motion-alignment or provenance-aware fusion model.

Q/K/V are projected once. Query positions are processed in chunks of 256. Neighbor gathering and attention
are checkpointed together during training, so expanded K/V are recomputed in backward rather than retained
for every chunk. Efficient SDPA is used; backend selection is controlled by PyTorch. This trades extra
backward work for lower activation storage. All three modalities and context frames still receive gradients.

Initialization: `runs/rollout_dry_pro6000_0Q6klO/dry_guarded/checkpoints/checkpoint_1` (step 250).
Both arms load all checkpoint tensors exactly, train cross/local parameters only, freeze backbone and
decoders, and start fresh identical optimizers. Both use the guarded dry-rollout loss, alpha=0 and rollout
weight=0.2. Batch=1, accumulation=8, seed=2025. The formal run uses 250 optimizer updates per arm,
validation at steps 0/125/250 on the same 60 development windows, and four free-rollout forecast frames.
History may contain previously generated predictions but no observed future modalities in free rollout.
Teacher-forced supervision remains as in the baseline and is distinct from free-rollout validation.

Unlike a loss-only ablation, expanding the neighborhood can change predictions immediately at step zero.
The report retains both initial evaluations. It checks the final 3x3 model against the common pointwise
initial reference as well as the matched-step trained control. Do not judge success solely from improvement
relative to the 3x3 model's own initial evaluation. The per-arm `best_eligible.json` uses each arm's initial
reference; the shared checks in `comparison.json` are needed for cross-arm interpretation.

## Launch

```bash
cd /home/rainpred/RainPrediction
tmux new-session -s cross3x3_pro6000 \
  'bash src/residual_diffusion_stage/ablate_neighborhood_cross_pro6000.sh'
```

GPU UUID locks to physical index 2 RTX PRO 6000, exposed as logical cuda:0.
The script checks the GPU identity and >=18 GiB free (a conservative guard), never stops other jobs, and
uses a new `runs/neighborhood_cross_pro6000_*` directory. Training progress is printed and full logs are
saved separately as `pointwise.log` and `neighborhood_3x3.log`. Ctrl-b then d detaches the tmux session;
`tmux attach -t cross3x3_pro6000` returns to it.

The final directory contains `comparison.md`, full metrics and shared checks in `comparison.json`, and
`comparison_index_*.png` with GT / pointwise / 3x3 on identical scales. Per-arm validation also saves rain
and temporal-change plots. Compare CSI/POD/FAR, core MAE/intensity, change amplitude and decay recall.
One seed and rain-filtered development samples cannot establish generalization.

Optional pipeline check (not an efficacy experiment):

```bash
bash src/residual_diffusion_stage/ablate_neighborhood_cross_pro6000.sh --steps 2 --val-every 2 --samples 2
```
