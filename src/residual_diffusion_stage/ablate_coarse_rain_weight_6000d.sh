#!/usr/bin/env bash
set -euo pipefail

cd /home/rainpred/RainPrediction
export CUDA_VISIBLE_DEVICES=GPU-9951e0e7-f48d-5739-1b6e-bd1e139057e0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/rainpred-mpl
RAIN_PYTHON=/home/rainpred/miniconda3/envs/rainpred/bin/python
RAIN_CONFIG=/home/rainpred/RainPrediction/runs/coarse_evolution_20261005/config

"$RAIN_PYTHON" - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA unavailable"
name = torch.cuda.get_device_name(0)
assert "RTX 6000D" in name, f"Expected RTX 6000D, got {name}"
free, total = torch.cuda.mem_get_info()
print(f"GPU={name}; free={free / 2**30:.2f} GiB; total={total / 2**30:.2f} GiB", flush=True)
if free < 14.5 * 2**30:
    raise RuntimeError("Need at least 14.5 GiB free before starting this experiment")
PY

RAIN_OUTPUT=$(mktemp -d /home/rainpred/RainPrediction/runs/coarse_alpha_6000d_XXXXXX)
printf 'Output: %s\n' "$RAIN_OUTPUT"
for RAIN_ALPHA in 2 0.5 0; do
    RAIN_ARM="alpha_${RAIN_ALPHA//./p}"
    printf 'Starting %s on RTX 6000D\n' "$RAIN_ARM"
    "$RAIN_PYTHON" -m src.trainer.rain_trainer_coarse_evolution \
        --config-path "$RAIN_CONFIG" --config-name config_total \
        train.max_steps=250 \
        train.scheduler.num_training_steps=1000 \
        train.loss.rain_region_weight.alpha="$RAIN_ALPHA" \
        hydra.run.dir="$RAIN_OUTPUT/$RAIN_ARM" \
        2>&1 | tee "$RAIN_OUTPUT/$RAIN_ARM.launch.log"
done

"$RAIN_PYTHON" -m src.residual_diffusion_stage.ablate_coarse_rain_weight \
    --report-only --source "$RAIN_OUTPUT/alpha_2" --output "$RAIN_OUTPUT"
printf 'Finished. Comparison: %s/comparison.md\n' "$RAIN_OUTPUT"
