# Rollout dry-region constraint experiment

Entry: `src.trainer.rain_trainer_rollout_dry`; config: `rain_trainer_rollout_dry.yaml`.
The original trainer/model files are unchanged. Only cross attention and local-window parameters train;
the pretrained backbone/decoders remain frozen but decoder operations still propagate gradients.
CCC is disabled in both arms. Initialization uses the prior without-CCC step-250 checkpoint.
Both arms start fresh, identical optimizer/scheduler states; this is not exact optimizer resumption.

For physical rainfall in mm/min, the added loss contains:

- Dry-region error: masked SmoothL1 of `relu(pred - GT) / 0.05` where `GT < 0.01`.
  It penalizes dry-region false rainfall at all intensities, including the dominant weak background.
- Event coverage: masked `relu(threshold - pred) / threshold` where `GT >= threshold`,
  averaged over thresholds 0.01, 0.1 and 0.5. Coefficient 0.25.
- Strong-rain protection: masked `relu(GT - pred) / GT` where `GT >= 0.1`. Coefficient 0.1.

Each masked loss is normalized by active pixels per lead. Lead weights are `[1, 1, 1.25, 1.5]`.
The combined term has coefficient 0.01, ramps over 50 optimizer updates, and is capped at 20%
of the existing inner rollout loss value using a detached scaling factor. The complete rollout loss
still has weight 0.2. Teacher-forced losses and inference/display thresholds are unchanged.
These loss coefficients are initial hypotheses for the paired experiment, not validated optimal values.

Validation uses the same fixed manifest, all four free-rollout frames, and no future observations.
`constraint_selection.jsonl` checks every lead at 0.01/0.1/0.2/0.5: CSI and POD drops cannot
exceed 0.005 absolute relative to step zero. Core-region MAE at 0.1/0.5/1.0 cannot worsen over 2%,
and core intensity ratios cannot decline over 2%. Mean t+3/t+4 FAR@0.01 must decrease.
`best_eligible.json` points to the saved checkpoint only if all checks pass; no passing checkpoint
is a valid outcome. These checks do not guarantee improvement on unseen data.
Compare the matched-step control as well before adopting a candidate.

Run on the 4090 when at least 16 GiB is free (a launch guard, not a measured peak):

```bash
cd /home/rainpred/RainPrediction
tmux new-session -s rollout_dry 'bash src/residual_diffusion_stage/ablate_rollout_dry_4090.sh'
```

The two sequential arms each run 250 optimizer updates with batch 1, accumulation 8, validating
60 fixed examples at steps 0, 125, 250. Output goes to a fresh `runs/rollout_dry_4090_*` directory.
Use Ctrl-b then d to detach; `tmux attach -t rollout_dry` to return. There is no GPU waiting loop.
The launch script does not stop other GPU jobs.

Read `comparison.md`/`comparison.json`, per-arm `constraint_selection.jsonl`, and the rainfall/change
plots in `fixed_validation/step_*`. The validation windows are rain-filtered: independent dry,
mixed-weather and different-month cases are still necessary to assess generalization.
