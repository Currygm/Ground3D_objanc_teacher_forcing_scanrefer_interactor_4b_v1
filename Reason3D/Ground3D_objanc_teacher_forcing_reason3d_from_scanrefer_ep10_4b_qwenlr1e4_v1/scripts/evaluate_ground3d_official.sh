#!/usr/bin/env bash
# Official model only. All writes/caches stay inside the authorized project.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
OFFICIAL_PYTHON="${OFFICIAL_PYTHON:-/home/yibo/ENTER/envs/ground3d/bin/python}"
SELECTION_PYTHON="${SELECTION_PYTHON:-$PROJECT/../Ground3D_objanc_teacher_forcing_scanrefer_4b_v1/env/bin/python}"
case "$PROJECT/" in /home/huangxiang/codes/guiming/*) ;; *) echo "Unauthorized project path" >&2; exit 2;; esac
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$PROJECT/cache/official_ground3d/hf" HF_MODULES_CACHE="$PROJECT/cache/official_ground3d/modules"
export TORCH_HOME="$PROJECT/cache/official_ground3d/torch" TORCH_EXTENSIONS_DIR="$PROJECT/cache/official_ground3d/extensions"
export CUDA_CACHE_PATH="$PROJECT/cache/official_ground3d/cuda" TRITON_CACHE_DIR="$PROJECT/cache/official_ground3d/triton"
export XDG_CACHE_HOME="$PROJECT/cache/official_ground3d/xdg" TMPDIR="$PROJECT/cache/official_ground3d/tmp"
mkdir -p "$HF_HOME" "$HF_MODULES_CACHE" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR" "$PROJECT/logs"
cd "$PROJECT"
for arg in "$@"; do
  if [[ "$arg" == --help || "$arg" == -h || "$arg" == --check-data || "$arg" == --check-model ]]; then
    export CUDA_VISIBLE_DEVICES=""
    exec "$OFFICIAL_PYTHON" -B "$PROJECT/evaluate_ground3d_official.py" "$@"
  fi
done
REQUESTED_GPU_IDS="${GPU_IDS:-1,2,4,6}"
GPU_UUIDS="$("$SELECTION_PYTHON" -B "$PROJECT/gpu_selection.py" "$REQUESTED_GPU_IDS")"
export CUDA_VISIBLE_DEVICES="$GPU_UUIDS"
IFS=',' read -ra GPUS <<< "$GPU_UUIDS"
echo "Requested physical GPUs: $REQUESTED_GPU_IDS"
echo "Frozen UUID order: $CUDA_VISIBLE_DEVICES"
LOG_FILE="$PROJECT/logs/evaluate_ground3d_official_$(date +%Y%m%d_%H%M%S)_$$.log"
exec > >(tee -a "$LOG_FILE") 2>&1
exec "$OFFICIAL_PYTHON" -B -m torch.distributed.run --standalone --nproc_per_node="${#GPUS[@]}" \
  "$PROJECT/evaluate_ground3d_official.py" "$@"
