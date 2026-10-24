#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
SOURCE=/home/huangxiang/codes/guiming/guiming_codes/research/Ground3D_objanc_teacher_forcing_scanrefer_4b_v1
PYTHON="${PYTHON:-$SOURCE/env/bin/python}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
export TMPDIR="$PROJECT/cache/ground3d_functional_tmp"
mkdir -p "$TMPDIR" "$PROJECT/logs"
LOG_FILE="$PROJECT/logs/prepare_ground3d_functional_$(date +%Y%m%d_%H%M%S)_$$.log"
exec > >(tee -a "$LOG_FILE") 2>&1
exec "$PYTHON" -B "$PROJECT/ground3d_functional.py" "$@"
