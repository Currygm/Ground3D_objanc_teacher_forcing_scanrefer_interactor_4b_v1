#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$PROJECT/env/bin/python}"
CONFIG="${CONFIG:-$PROJECT/configs/scanrefer_teacher_forcing_univlg9.yaml}"
IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-}}"
if [[ $# -lt 2 ]]; then
  echo "Usage: GPU_IDS=1,2 bash $0 CHECKPOINT NEW_OUTPUT_DIR [native evaluator options]" >&2
  exit 2
fi
if [[ -z "$IDS" ]]; then echo 'Explicit GPU_IDS or CUDA_VISIBLE_DEVICES required' >&2; exit 2; fi
IFS=',' read -ra GPUS <<< "$IDS"
for gpu in "${GPUS[@]}"; do
  if [[ ! "$gpu" =~ ^[0-9]+$ && ! "$gpu" =~ ^GPU-[[:xdigit:]-]+$ ]]; then
    echo "Invalid GPU ID or UUID: $gpu" >&2; exit 2
  fi
done
if [[ "$(printf '%s\n' "${GPUS[@]}" | sort -u | wc -l)" -ne "${#GPUS[@]}" ]]; then
  echo 'Duplicate GPU IDs' >&2; exit 2
fi
export CUDA_VISIBLE_DEVICES="$IDS" HF_HOME="$PROJECT/cache/huggingface"
export TMPDIR="${GROUND3D_TMPDIR:-/home/huangxiang/codes/guiming/.g3rf_tmp}"
TMPDIR="$(realpath -m -- "$TMPDIR")"
case "$TMPDIR" in
 /home/huangxiang/codes/guiming/*) ;;
 *) echo 'TMPDIR must resolve inside /home/huangxiang/codes/guiming' >&2; exit 2 ;;
esac
if [[ "${#TMPDIR}" -gt 70 ]]; then echo 'TMPDIR too long for multiprocessing Unix sockets' >&2; exit 2; fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
mkdir -p "$HF_HOME" "$TMPDIR"
cd "$PROJECT"
exec "$PYTHON" -B -m torch.distributed.run --standalone --nproc_per_node="${#GPUS[@]}" \
  "$PROJECT/evaluate_segmentation_unconstrained.py" "$CONFIG" "$@"
