#!/usr/bin/env bash
set -euo pipefail
cd /home/rainpred/RainPrediction
export CUDA_VISIBLE_DEVICES=GPU-7ca52c3b-5267-21e3-3276-1e47df4af1c1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/rainpred-mpl
export PYTHONUNBUFFERED=1
RAIN_PYTHON=/home/rainpred/miniconda3/envs/rainpred/bin/python
"$RAIN_PYTHON" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
name = torch.cuda.get_device_name(0)
free, total = torch.cuda.mem_get_info()
print(f"GPU: {name}; free={free / 2**30:.2f}/{total / 2**30:.2f} GiB", flush=True)
assert "RTX PRO 6000" in name, f"Unexpected GPU: {name}"
assert free >= 16 * 2**30, "Need at least 16 GiB free; this is a conservative launch guard, not a measured peak"
PY
RAIN_OUTPUT=$(mktemp -d /home/rainpred/RainPrediction/runs/ccc_frozen_ablation_XXXXXX)
printf 'Experiment output: %s\n' "$RAIN_OUTPUT"
"$RAIN_PYTHON" -m src.residual_diffusion_stage.ablate_ccc_frozen \
    --output "$RAIN_OUTPUT" "$@" 2>&1 | tee "$RAIN_OUTPUT/launch.log"
