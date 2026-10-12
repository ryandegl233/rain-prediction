#!/usr/bin/env bash
set -euo pipefail

cd /home/rainpred/RainPrediction
export CUDA_VISIBLE_DEVICES=GPU-7ca52c3b-5267-21e3-3276-1e47df4af1c1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/rainpred-mpl
export PYTHONUNBUFFERED=1
RAIN_PYTHON=/home/rainpred/miniconda3/envs/rainpred/bin/python
RAIN_ORIGINAL=runs/next_frame_final/2026-08-17/14-41-50_stage1_next_frame_block
RAIN_ENHANCED=runs/next_frame_cross_local_finetune/2026-08-29/22-03-48_cross_local_finetune

"$RAIN_PYTHON" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA unavailable"
name = torch.cuda.get_device_name(0)
free, total = torch.cuda.mem_get_info()
print(f"GPU: {name}; free={free / 2**30:.2f}/{total / 2**30:.2f} GiB", flush=True)
assert "RTX PRO 6000" in name, f"Unexpected GPU: {name}"
assert free >= 15 * 2**30, "Need at least 15 GiB free"
PY

RAIN_OUTPUT=$(mktemp -d /home/rainpred/RainPrediction/runs/original_cross_local_compare_XXXXXX)
printf 'Output: %s\n' "$RAIN_OUTPUT"
"$RAIN_PYTHON" -m src.residual_diffusion_stage.compare_original_cross_local \
    --original-config "$RAIN_ORIGINAL/config/config_total.yaml" \
    --original-checkpoint "$RAIN_ORIGINAL/checkpoints/checkpoint_19" \
    --cross-local-config "$RAIN_ENHANCED/config/config_total.yaml" \
    --cross-local-checkpoint "$RAIN_ENHANCED/checkpoints/checkpoint_12" \
    --manifest runs/ccc_fixed_comparison_20261005/comparison.json \
    --samples "${RAIN_COMPARE_SAMPLES:-60}" \
    --output-dir "$RAIN_OUTPUT/results" \
    2>&1 | tee "$RAIN_OUTPUT/launch.log"
