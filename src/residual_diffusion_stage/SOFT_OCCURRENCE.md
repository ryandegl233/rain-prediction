# Soft occurrence experiment

This experiment leaves the original deterministic predictor frozen and trains a pixel-level rain-occurrence head.
The predictor still supplies rainfall intensity. The new head estimates `P(rain >= 0.01 mm/min)` and applies a soft
`p^gamma` attenuation after the original residual rainfall has been reconstructed.

The gate is exactly disabled through step 50, ramps to full strength by step 100, and is then evaluated at full
strength. Four matched arms isolate `gamma` and false-positive weighting. A candidate is selected only when its
long-lead weak-rain FAR decreases without exceeding the configured CSI, POD, or strong-core tolerances.

```bash
cd /home/rainpred/RainPrediction
tmux new-session -s soft_occurrence_pro6000 \
  'bash src/residual_diffusion_stage/ablate_soft_occurrence_pro6000.sh'
```

For a two-step smoke test:

```bash
bash src/residual_diffusion_stage/ablate_soft_occurrence_pro6000.sh --steps 2 --val-every 2 --samples 2
```
