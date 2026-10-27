#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="/home/huangxiang/codes/hx_l/code/research/Ground3D_objanc_teacher_forcing_scanrefer_interactor_4b_v1/env/bin/python"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$PROJECT/logs"
LOG="$PROJECT/logs/${RUN:-baseline}_answer_$(date +%Y%m%d_%H%M%S)_$$.log"
"$PYTHON" "$PROJECT/launch_experiment.py" answer "$@" 2>&1 | tee "$LOG"
