#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then echo "Usage: GPU_IDS=1,2,4,6 bash $0 CHECKPOINT OUTPUT_DIR [--modes ...]" >&2; exit 2; fi
CHECKPOINT="$1";EVALUATION_OUTPUT="$2";shift 2
source "$(dirname -- "${BASH_SOURCE[0]}")/runtime.sh"
LOG_FILE="$PROJECT/logs/evaluate_$(date +%Y%m%d_%H%M%S)_$$.log"
exec > >(tee -a "$LOG_FILE") 2>&1
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="${#GPUS[@]}" \
  "$PROJECT/evaluate_reason3d.py" "$CONFIG" "$CHECKPOINT" "$EVALUATION_OUTPUT" "$@"
