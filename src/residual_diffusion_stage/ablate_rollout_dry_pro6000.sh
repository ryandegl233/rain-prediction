#!/usr/bin/env bash
set -euo pipefail
cd /home/rainpred/RainPrediction
export CUDA_VISIBLE_DEVICES=GPU-7ca52c3b-5267-21e3-3276-1e47df4af1c1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/rainpred-mpl
export PYTHONUNBUFFERED=1
RAIN_PYTHON=/home/rainpred/miniconda3/envs/rainpred/bin/python
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=index,name,uuid,memory.free --format=csv
"$RAIN_PYTHON" - <<'PY'
from pathlib import Path
import torch

assert torch.cuda.is_available(), "CUDA unavailable"
assert torch.cuda.device_count() == 1, "Expected one visible GPU"
name = torch.cuda.get_device_name(0)
free, total = torch.cuda.mem_get_info()
print(f"Logical cuda:0 = {name}; free={free / 2**30:.2f}/{total / 2**30:.2f} GiB", flush=True)
assert "RTX PRO 6000" in name, f"Unexpected GPU: {name}"
assert free >= 16 * 2**30, "Need >=16 GiB free (launch guard, not measured peak)"
checkpoint = Path("runs/ccc_frozen_ablation_4090_b112aM/without_ccc/checkpoints/checkpoint_0/model.safetensors")
assert checkpoint.is_file(), f"Missing initialization checkpoint: {checkpoint}"
assert Path("runs/ccc_fixed_comparison_20261005/comparison.json").is_file(), "Missing validation manifest"
PY
if [[ "${1:-}" == "--check-only" ]]; then
    exit 0
fi
RAIN_OUTPUT=$(mktemp -d /home/rainpred/RainPrediction/runs/rollout_dry_pro6000_XXXXXX)
printf 'Experiment output: %s\n' "$RAIN_OUTPUT"
printf 'Training logs: %s/control.log and %s/dry_guarded.log\n' "$RAIN_OUTPUT" "$RAIN_OUTPUT"
"$RAIN_PYTHON" -m src.residual_diffusion_stage.ablate_rollout_dry --output "$RAIN_OUTPUT" "$@"
