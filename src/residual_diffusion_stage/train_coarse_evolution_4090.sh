#!/usr/bin/env bash
set -euo pipefail

cd /home/rainpred/RainPrediction
export CUDA_VISIBLE_DEVICES=GPU-53d5627c-0a70-b40d-031b-51c20452acb6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/rainpred-mpl
RAIN_PYTHON=/home/rainpred/miniconda3/envs/rainpred/bin/python

"$RAIN_PYTHON" - <<'PY'
import torch

assert torch.cuda.is_available(), "CUDA unavailable"
name = torch.cuda.get_device_name(0)
assert "4090" in name, f"Expected RTX 4090, got {name}"
free, total = torch.cuda.mem_get_info()
print(f"GPU={name}; free={free / 2**30:.2f} GiB; total={total / 2**30:.2f} GiB", flush=True)
if free < 13.5 * 2**30:
    raise RuntimeError("Need at least 13.5 GiB free for the measured batch=1, accumulation=8 profile")
PY

exec "$RAIN_PYTHON" -m src.trainer.rain_trainer_coarse_evolution "$@"
