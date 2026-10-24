#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/runtime.sh"
PREFLIGHT_ENV="$PROJECT/logs/nccl_check_$(date +%Y%m%d_%H%M%S)_$$.env"
exec "$PYTHON" -u -B "$PROJECT/distributed_preflight.py" \
  --world-size "${#GPUS[@]}" --env-file "$PREFLIGHT_ENV" --profile "${NCCL_PROFILE:-auto}"
