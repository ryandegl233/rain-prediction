#!/usr/bin/env bash
set -euo pipefail

cd /home/rainpred/RainPrediction

export CUDA_VISIBLE_DEVICES=GPU-53d5627c-0a70-b40d-031b-51c20452acb6
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/rainpred-mpl
export PYTHONUNBUFFERED=1

RAIN_PYTHON=/home/rainpred/miniconda3/envs/rainpred/bin/python

nvidia-smi -i "$CUDA_VISIBLE_DEVICES" \
  --query-gpu=index,name,memory.free \
  --format=csv

"$RAIN_PYTHON" - <<'PY'
from pathlib import Path
import torch

assert torch.cuda.is_available() and torch.cuda.device_count() == 1
name = torch.cuda.get_device_name(0)
free, total = torch.cuda.mem_get_info()
print(
    f"Logical cuda:0 = {name}; free={free / 2**30:.2f}/{total / 2**30:.2f} GiB",
    flush=True,
)
assert "RTX 4090" in name, name
assert free >= 10 * 2**30, "Need >=10 GiB free for the batch=2 training run"
assert Path(
    "runs/rollout_dry_pro6000_0Q6klO/dry_guarded/checkpoints/"
    "checkpoint_1/model.safetensors"
).is_file()
assert Path("runs/ccc_fixed_comparison_20261005/comparison.json").is_file()
PY

if [[ "${1:-}" == "--check-only" ]]; then
  exit 0
fi

RAIN_OUTPUT=$(mktemp -d \
  /home/rainpred/RainPrediction/runs/band_occurrence_4090_XXXXXX)
printf 'Experiment output: %s\n' "$RAIN_OUTPUT"

"$RAIN_PYTHON" -m src.residual_diffusion_stage.ablate_band_occurrence \
  --output "$RAIN_OUTPUT" "$@"
