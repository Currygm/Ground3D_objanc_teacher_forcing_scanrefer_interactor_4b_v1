#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/runtime.sh"
LOG_FILE="$PROJECT/logs/train_$(date +%Y%m%d_%H%M%S)_$$.log"
exec > >(tee -a "$LOG_FILE") 2>&1
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="${#GPUS[@]}" \
  "$PROJECT/train_reason3d.py" "$CONFIG" --output-dir "$OUTPUT_DIR" "$@"
