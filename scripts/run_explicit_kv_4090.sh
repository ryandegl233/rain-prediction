#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/rainpred/RainPrediction"
PYTHON_BIN="/home/rainpred/miniconda3/envs/rainpred/bin/python"
RUN_DIR="${1:-${PROJECT_ROOT}/runs/explicit_kv_4090_$(date +%Y%m%d_%H%M%S)}"

cd "${PROJECT_ROOT}"
exec env \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  MPLCONFIGDIR=/tmp/matplotlib-explicit-kv \
  "${PYTHON_BIN}" -m src.trainer.rain_trainer_explicit_kv \
  hydra.run.dir="${RUN_DIR}"
