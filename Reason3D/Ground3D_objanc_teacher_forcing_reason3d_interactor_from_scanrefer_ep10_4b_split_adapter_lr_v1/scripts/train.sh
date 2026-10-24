#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/runtime.sh"
LOG_FILE="$PROJECT/logs/train_$(date +%Y%m%d_%H%M%S)_$$.log"
exec > >(tee -a "$LOG_FILE") 2>&1
# Probe the exact UUID mapping before loading 4B weights.
if (( ${#GPUS[@]} > 1 )); then
  PREFLIGHT_ENV="$PROJECT/logs/nccl_preflight_$(date +%Y%m%d_%H%M%S)_$$.env"
  "$PYTHON" -u -B "$PROJECT/distributed_preflight.py" \
    --world-size "${#GPUS[@]}" --env-file "$PREFLIGHT_ENV" --profile "${NCCL_PROFILE:-auto}"
  source "$PREFLIGHT_ENV"
fi
export NCCL_TRAIN_TIMEOUT_SECONDS="${NCCL_TRAIN_TIMEOUT_SECONDS:-600}"
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="${#GPUS[@]}" \
  "$PROJECT/train_reason3d.py" "$CONFIG" --output-dir "$OUTPUT_DIR" "$@"
