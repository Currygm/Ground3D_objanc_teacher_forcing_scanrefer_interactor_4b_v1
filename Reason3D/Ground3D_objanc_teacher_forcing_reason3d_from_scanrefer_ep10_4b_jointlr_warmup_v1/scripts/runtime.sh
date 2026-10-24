#!/usr/bin/env bash
# Sourced by the launchers; all new artifacts remain in this project.
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
SOURCE=/home/huangxiang/codes/guiming/guiming_codes/research/Ground3D_objanc_teacher_forcing_scanrefer_4b_v1
export PYTHON="${PYTHON:-$SOURCE/env/bin/python}"
export CONFIG="${CONFIG:-$PROJECT/configs/reason3d_answer_lm3_objce3_warmup3_joint10.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT/outputs/reason3d_answer_lm3_objce3_warmup3_joint10}"
export EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$PROJECT/evaluations/reason3d_answer_lm3_objce3_warmup3_joint10/epoch013_unconstrained_by_dataset}"
REQUESTED_GPU_IDS="${GPU_IDS:-1,2,4,6}"
GPU_UUIDS="$("$PYTHON" -B "$PROJECT/gpu_selection.py" "$REQUESTED_GPU_IDS")"
export GPU_IDS="$GPU_UUIDS" CUDA_VISIBLE_DEVICES="$GPU_UUIDS"
IFS=',' read -ra GPUS <<< "$GPU_IDS"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HOME="$PROJECT/cache/huggingface" HF_MODULES_CACHE="$PROJECT/cache/huggingface/modules"
export TORCH_HOME="$PROJECT/cache/torch" TORCH_EXTENSIONS_DIR="$PROJECT/cache/torch_extensions"
export CUDA_CACHE_PATH="$PROJECT/cache/cuda_jit" TRITON_CACHE_DIR="$PROJECT/cache/triton" XDG_CACHE_HOME="$PROJECT/cache/xdg"
export TMPDIR=/home/huangxiang/codes/guiming/.g3lrwt
mkdir -p "$HF_HOME" "$HF_MODULES_CACHE" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR" "$PROJECT/logs"
chmod 700 "$TMPDIR"
echo "Requested physical GPUs: $REQUESTED_GPU_IDS"
echo "Frozen CUDA_VISIBLE_DEVICES UUID order: $GPU_IDS"
cd "$PROJECT"
