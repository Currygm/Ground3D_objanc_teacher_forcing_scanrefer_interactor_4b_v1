#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: GPU_IDS=1,2,4,6 bash $0 CHECKPOINT OUTPUT_DIR [--modes generated_no_fallback teacher] [--sources scannet scannetpp] [--levels object_qa part_qa]" >&2
  exit 2
fi
CHECKPOINT="$1"; RESULT_DIR="$2"; shift 2
source "$(dirname -- "${BASH_SOURCE[0]}")/runtime.sh"
ANNOTATIONS="${GROUND3D_ANNOTATIONS:-$PROJECT/data/ground3d_functional_val/annotations.jsonl}"
if [[ ! -s "$ANNOTATIONS" ]]; then
  echo "Missing prepared Ground3D annotations. Run scripts/prepare_ground3d_functional.sh first." >&2
  exit 2
fi
LOG_FILE="$PROJECT/logs/evaluate_ground3d_functional_$(date +%Y%m%d_%H%M%S)_$$.log"
exec > >(tee -a "$LOG_FILE") 2>&1
exec "$PYTHON" -B -m torch.distributed.run --standalone --nproc_per_node="${#GPUS[@]}" \
  "$PROJECT/evaluate_ground3d_functional.py" --config "$CONFIG" --checkpoint "$CHECKPOINT" \
  --annotations "$ANNOTATIONS" --output "$RESULT_DIR" "$@"
