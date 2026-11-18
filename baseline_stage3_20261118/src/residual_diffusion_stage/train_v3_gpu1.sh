#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/rainpred/RainPrediction"
PYTHON_BIN="/home/rainpred/miniconda3/envs/rainpred/bin/python"
STAGE_CONFIG="${PROJECT_DIR}/src/residual_diffusion_stage/config_v3_mu_gate_delta.yaml"
COARSE_CHECKPOINT="${COARSE_CHECKPOINT:-${PROJECT_DIR}/runs/next_frame_cross_local_finetune/2026-08-29/22-03-48_cross_local_finetune/checkpoints/checkpoint_12}"
AE_CHECKPOINT="${AE_CHECKPOINT:-${PROJECT_DIR}/runs/residual_diffusion_cross_local_v1/ae/final.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/runs/residual_diffusion_cross_local_v3}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "${OUTPUT_DIR}/logs"
cd "${PROJECT_DIR}"

echo "[1/2] Training deterministic mu + gate corrector on physical RTX 4090 (GPU 1)"
"${PYTHON_BIN}" -m src.residual_diffusion_stage.train \
  --stage corrector \
  --stage-config "${STAGE_CONFIG}" \
  --coarse-checkpoint "${COARSE_CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  2>&1 | tee "${OUTPUT_DIR}/logs/corrector.log"

if [[ ! -f "${OUTPUT_DIR}/corrector/best.pt" ]]; then
  echo "Corrector training finished without producing corrector/best.pt" >&2
  exit 1
fi

echo "[2/2] Training conditional diffusion for stochastic delta residual"
"${PYTHON_BIN}" -m src.residual_diffusion_stage.train \
  --stage diffusion \
  --stage-config "${STAGE_CONFIG}" \
  --coarse-checkpoint "${COARSE_CHECKPOINT}" \
  --ae-checkpoint "${AE_CHECKPOINT}" \
  --corrector-checkpoint "${OUTPUT_DIR}/corrector/best.pt" \
  --output-dir "${OUTPUT_DIR}" \
  2>&1 | tee "${OUTPUT_DIR}/logs/diffusion.log"

echo "Training complete"
echo "Corrector: ${OUTPUT_DIR}/corrector/best.pt"
echo "Diffusion: ${OUTPUT_DIR}/diffusion/best.pt"
