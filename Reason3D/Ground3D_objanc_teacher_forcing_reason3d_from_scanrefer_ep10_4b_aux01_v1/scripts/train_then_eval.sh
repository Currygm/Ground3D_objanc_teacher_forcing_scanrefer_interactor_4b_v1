#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo "Usage: GPU_IDS=1,2,4,6 bash $0 [--resume NEW_REASON3D_CHECKPOINT]"
  echo "Train Answer3 + joint10 with CE3/OBJCE3, auxiliary SUM weight0.1 and baseline LR; save global3/8/13; evaluate final13 on the same GPU UUIDs."
  echo "Optional environment: OUTPUT_DIR, EVAL_OUTPUT_DIR, CONFIG, PYTHON."
  exit 0
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/runtime.sh"
bash "$PROJECT/scripts/train.sh" "$@"
CHECKPOINT="$("$PYTHON" -B "$PROJECT/tools/select_final_checkpoint.py" "$CONFIG" "$OUTPUT_DIR")"
echo "Verified completed final checkpoint: $CHECKPOINT"
bash "$PROJECT/scripts/evaluate_unconstrained.sh" "$CHECKPOINT" "$EVAL_OUTPUT_DIR" \
  --modes teacher generated
