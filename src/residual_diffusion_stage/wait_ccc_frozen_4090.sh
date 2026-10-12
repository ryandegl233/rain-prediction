#!/usr/bin/env bash
set -euo pipefail
cd /home/rainpred/RainPrediction
RAIN_GPU_UUID=GPU-53d5627c-0a70-b40d-031b-51c20452acb6
exec 9>/home/rainpred/RainPrediction/runs/.ccc_frozen_4090_wait.lock
if ! flock -n 9; then
    echo 'A CCC 4090 waiter or its training process is already active.'
    exit 1
fi
RAIN_IDLE_CHECKS=0
while true; do
    if RAIN_FREE_MIB=$(nvidia-smi -i "$RAIN_GPU_UUID" --query-gpu=memory.free --format=csv,noheader,nounits) && \
       RAIN_COMPUTE=$(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits); then
        RAIN_GPU_PIDS=$(awk -F ', *' -v gpu="$RAIN_GPU_UUID" '$1 == gpu {print $2}' <<< "$RAIN_COMPUTE")
        RAIN_FREE_MIB=${RAIN_FREE_MIB//[[:space:]]/}
        if [[ "$RAIN_FREE_MIB" =~ ^[0-9]+$ ]] && [[ -z "$RAIN_GPU_PIDS" ]] && (( RAIN_FREE_MIB >= 22528 )); then
            RAIN_IDLE_CHECKS=$((RAIN_IDLE_CHECKS + 1))
        else
            RAIN_IDLE_CHECKS=0
        fi
        printf '%s | RTX 4090 free=%s MiB; compute PIDs=%s; idle checks=%s/3\n' \
            "$(date '+%F %T')" "$RAIN_FREE_MIB" "${RAIN_GPU_PIDS:-none}" "$RAIN_IDLE_CHECKS"
    else
        RAIN_IDLE_CHECKS=0
        echo 'GPU query failed; continuing to wait without launching.'
    fi
    if [[ ${1:-} == --check-only ]]; then
        exit 0
    fi
    if (( RAIN_IDLE_CHECKS >= 3 )); then
        echo 'RTX 4090 has been idle for 60 seconds; starting the paired CCC ablation.'
        exec bash src/residual_diffusion_stage/ablate_ccc_frozen_4090.sh "$@"
    fi
    sleep 30
done
