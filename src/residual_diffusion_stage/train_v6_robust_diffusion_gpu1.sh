#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/rainpred/RainPrediction"
PYTHON_BIN="/home/rainpred/miniconda3/envs/rainpred/bin/python"
STAGE_CONFIG="${STAGE_CONFIG:-${PROJECT_DIR}/src/residual_diffusion_stage/config_v3_mu_gate_delta.yaml}"
COARSE_CHECKPOINT="${COARSE_CHECKPOINT:-${PROJECT_DIR}/runs/next_frame_cross_local_finetune/2026-08-29/22-03-48_cross_local_finetune/checkpoints/checkpoint_12}"
AE_CHECKPOINT="${AE_CHECKPOINT:-${PROJECT_DIR}/runs/residual_diffusion_cross_local_v2_t1peak/ae/final.pt}"
CORRECTOR_CHECKPOINT="${CORRECTOR_CHECKPOINT:-${PROJECT_DIR}/runs/residual_diffusion_cross_local_v6_robust_occurrence/corrector/best.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/runs/residual_diffusion_cross_local_v6_robust_occurrence}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${OUTPUT_DIR}/logs"
cd "${PROJECT_DIR}"

echo "Training diffusion with robust occurrence corrector on physical GPU 1"
"${PYTHON_BIN}" -m src.residual_diffusion_stage.train \
  --stage diffusion \
  --stage-config "${STAGE_CONFIG}" \
  --coarse-checkpoint "${COARSE_CHECKPOINT}" \
  --ae-checkpoint "${AE_CHECKPOINT}" \
  --corrector-checkpoint "${CORRECTOR_CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  2>&1 | tee "${OUTPUT_DIR}/logs/diffusion_robust.log"

echo "Robust diffusion training complete: ${OUTPUT_DIR}/diffusion/best.pt"
