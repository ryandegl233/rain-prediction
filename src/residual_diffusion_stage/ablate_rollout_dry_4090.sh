#!/usr/bin/env bash
set -euo pipefail
cd /home/rainpred/RainPrediction
export CUDA_VISIBLE_DEVICES=GPU-53d5627c-0a70-b40d-031b-51c20452acb6
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
assert "RTX 4090" in name, f"Unexpected GPU: {name}"
assert free >= 16 * 2**30, "Need >=16 GiB free (launch guard, not measured peak); do not stop other jobs"
PY
RAIN_OUTPUT=$(mktemp -d /home/rainpred/RainPrediction/runs/rollout_dry_4090_XXXXXX)
printf 'Experiment output: %s\n' "$RAIN_OUTPUT"
"$RAIN_PYTHON" -m src.residual_diffusion_stage.ablate_rollout_dry --output "$RAIN_OUTPUT" "$@"
