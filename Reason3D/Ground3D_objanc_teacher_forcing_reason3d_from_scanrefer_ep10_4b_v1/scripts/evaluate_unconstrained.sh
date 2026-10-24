#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: GPU_IDS=1,2,4,6 bash scripts/evaluate_unconstrained.sh CHECKPOINT OUTPUT [ARGS...]"
  exit 2
fi
CHECKPOINT="$1"
EVALUATION_OUTPUT="$2"
shift 2
source "$(dirname -- "${BASH_SOURCE[0]}")/runtime.sh"
"$PYTHON" -B - "$PROJECT" "$CHECKPOINT" "$EVALUATION_OUTPUT" <<'PY'
from pathlib import Path
import sys
project,checkpoint,output=map(lambda p:Path(p).resolve(),sys.argv[1:])
checkpoint.relative_to(project);output.relative_to(project)
if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
if output.exists() and any(output.iterdir()):raise FileExistsError("Use a fresh empty evaluation output directory")
print("Unconstrained no-Interactor eval:",checkpoint,"->",output,flush=True)
PY
LOG_FILE="$PROJECT/logs/evaluate_unconstrained_$(date +%Y%m%d_%H%M%S)_$$.log"
exec > >(tee -a "$LOG_FILE") 2>&1
export PYTHONUNBUFFERED=1
echo "Pinned GPUs: $CUDA_VISIBLE_DEVICES"
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="${#GPUS[@]}" \
  "$PROJECT/evaluate_reason3d_unconstrained.py" "$CONFIG" "$CHECKPOINT" "$EVALUATION_OUTPUT" "$@"
