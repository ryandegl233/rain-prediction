#!/usr/bin/env bash
set -euo pipefail

cd /home/rainpred/RainPrediction
RAIN_SOURCE=/home/rainpred/RainPrediction/runs/coarse_evolution_20261005
RAIN_CHECKPOINT="$RAIN_SOURCE/checkpoints/checkpoint_1"
for RAIN_FILE in model.safetensors optimizer.bin scheduler.bin random_states_0.pkl meta.json; do
    test -f "$RAIN_CHECKPOINT/$RAIN_FILE"
done
test -f "$RAIN_SOURCE/ema/ema.pt"
RAIN_RUN=$(mktemp -d /home/rainpred/RainPrediction/runs/coarse_evolution_resume_XXXXXX)
mkdir "$RAIN_RUN/ema"
cp "$RAIN_SOURCE/ema/ema.pt" "$RAIN_RUN/ema/ema.pt"
printf 'Resume step 500 -> 1000. Output: %s\n' "$RAIN_RUN"
exec bash src/residual_diffusion_stage/train_coarse_evolution_4090.sh \
    train.resume_path="$RAIN_CHECKPOINT" \
    train.init_model_path=null \
    train.max_steps=1000 \
    train.validate_before_training=false \
    hydra.run.dir="$RAIN_RUN"
