#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/rainpred/RainPrediction"
PYTHON_BIN="/home/rainpred/miniconda3/envs/rainpred/bin/python"
STAGE_CONFIG="${PROJECT_DIR}/src/residual_diffusion_stage/config_v4_robust_occurrence.yaml"
COARSE_CHECKPOINT="${COARSE_CHECKPOINT:-${PROJECT_DIR}/runs/next_frame_cross_local_finetune/2026-08-29/22-03-48_cross_local_finetune/checkpoints/checkpoint_12}"
INITIAL_CORRECTOR="${INITIAL_CORRECTOR:-${PROJECT_DIR}/runs/residual_diffusion_cross_local_v5_t1peak_balanced/corrector/best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/runs/residual_diffusion_cross_local_v6_robust_occurrence}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${OUTPUT_DIR}/logs"
cd "${PROJECT_DIR}"

"${PYTHON_BIN}" -m src.residual_diffusion_stage.train_robust_corrector \
  --stage-config "${STAGE_CONFIG}" \
  --coarse-checkpoint "${COARSE_CHECKPOINT}" \
  --initial-corrector-checkpoint "${INITIAL_CORRECTOR}" \
  --output-dir "${OUTPUT_DIR}" \
  2>&1 | tee "${OUTPUT_DIR}/logs/corrector.log"

echo "Robust corrector training complete: ${OUTPUT_DIR}/corrector/best.pt"
