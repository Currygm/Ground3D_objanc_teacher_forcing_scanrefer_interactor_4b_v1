#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo "Usage: GPU_IDS=1,2,4,6 bash $0 [--resume THIS_EXPERIMENT_CHECKPOINT]"
  echo "Answer3 + Joint15; Joint visual LR1e-5 and Qwen LoRA LR1e-4; CE3/OBJCE3/Aux0.5."
  echo "Save global3/8/13/18. After training, evaluate Joint10 and Joint15 (global13/18)."
  echo "Optional: OUTPUT_DIR, EVAL_OUTPUT_DIR (evaluation root), CONFIG, PYTHON."
  exit 0
fi
source "$(dirname -- "${BASH_SOURCE[0]}")/runtime.sh"
bash "$PROJECT/scripts/train.sh" "$@"
CHECKPOINT10="$("$PYTHON" -B "$PROJECT/tools/select_final_checkpoint.py" "$CONFIG" "$OUTPUT_DIR" 13)"
CHECKPOINT15="$("$PYTHON" -B "$PROJECT/tools/select_final_checkpoint.py" "$CONFIG" "$OUTPUT_DIR" 18)"
echo "Verified Joint10 checkpoint: $CHECKPOINT10"
echo "Verified Joint15 checkpoint: $CHECKPOINT15"
bash "$PROJECT/scripts/evaluate_unconstrained.sh" "$CHECKPOINT10" "$EVAL_OUTPUT_DIR/epoch013_unconstrained_by_dataset" --modes teacher generated
bash "$PROJECT/scripts/evaluate_unconstrained.sh" "$CHECKPOINT15" "$EVAL_OUTPUT_DIR/epoch018_unconstrained_by_dataset" --modes teacher generated
