#!/usr/bin/env bash
# Sourced by the launchers; all new artifacts remain in this project.
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
SOURCE=/home/huangxiang/codes/guiming/guiming_codes/research/Ground3D_objanc_teacher_forcing_scanrefer_interactor_4b_v1
export PYTHON="${PYTHON:-$SOURCE/env/bin/python}"
# Variable-size scenes otherwise leave many unusable allocator fragments.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CONFIG="${CONFIG:-$PROJECT/configs/reason3d_answer_lm3_objce3_warmup3_joint10.yaml}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT/outputs/reason3d_answer_lm3_objce3_warmup3_joint10}"
export EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-$PROJECT/evaluations/reason3d_answer_lm3_objce3_warmup3_joint10/epoch013_unconstrained_by_dataset}"
REQUESTED_GPU_IDS="${GPU_IDS:-1,2,4,6}"
GPU_UUIDS="$("$PYTHON" -B "$PROJECT/gpu_selection.py" "$REQUESTED_GPU_IDS")"
export GPU_IDS="$GPU_UUIDS" CUDA_VISIBLE_DEVICES="$GPU_UUIDS"
IFS=',' read -ra GPUS <<< "$GPU_IDS"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$PROJECT/cache/huggingface" HF_MODULES_CACHE="$PROJECT/cache/huggingface/modules"
export TORCH_HOME="$PROJECT/cache/torch" TORCH_EXTENSIONS_DIR="$PROJECT/cache/torch_extensions"
export CUDA_CACHE_PATH="$PROJECT/cache/cuda_jit" TRITON_CACHE_DIR="$PROJECT/cache/triton" XDG_CACHE_HOME="$PROJECT/cache/xdg"
export TMPDIR=/home/huangxiang/codes/guiming/.g3ri3tmp
# Validate resolved write destinations before mkdir/tee.
"$PYTHON" -B - "$PROJECT" "$OUTPUT_DIR" "$EVAL_OUTPUT_DIR" "$TMPDIR" <<'PY'
import pathlib,sys
project=pathlib.Path(sys.argv[1]).resolve(strict=True)
root=pathlib.Path("/home/huangxiang/codes/guiming").resolve(strict=True)
assert project.is_relative_to(root)
if not pathlib.Path(sys.argv[4]).resolve().is_relative_to(root):raise ValueError("IPC temporary path escapes authorized root")
for value in sys.argv[2:4]:
    if not pathlib.Path(value).resolve().is_relative_to(project):raise ValueError("Artifact destination escapes project")
for name in ("logs","cache","cache/tmp","cache/huggingface","cache/huggingface/modules","cache/torch","cache/torch_extensions","cache/cuda_jit","cache/triton","cache/xdg"):
    if not (project/name).resolve().is_relative_to(project):raise ValueError("Cache/log symlink escapes project")
PY
mkdir -p "$HF_HOME" "$HF_MODULES_CACHE" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$XDG_CACHE_HOME" "$TMPDIR" "$PROJECT/logs"
chmod 700 "$TMPDIR"
echo "Requested physical GPUs: $REQUESTED_GPU_IDS"
echo "Frozen CUDA_VISIBLE_DEVICES UUID order: $GPU_IDS"
cd "$PROJECT"
