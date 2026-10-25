#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$PROJECT/env/bin/python}"
CONFIG="${CONFIG:-$PROJECT/configs/feedback_a_visual.yaml}"
IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-}}"
if [[ -z "$IDS" ]]; then echo 'Explicit GPU_IDS or CUDA_VISIBLE_DEVICES required' >&2; exit 2; fi
if [[ ! "$IDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then echo 'Invalid GPU list' >&2; exit 2; fi
IFS=',' read -ra GPUS <<< "$IDS"
if [[ "$(printf '%s\n' "${GPUS[@]}" | sort -u | wc -l)" -ne "${#GPUS[@]}" ]]; then echo 'Duplicate GPU IDs' >&2; exit 2; fi
export CUDA_VISIBLE_DEVICES="$IDS" HF_HOME="$PROJECT/cache/huggingface"
# Unix socket paths include pymp/listener suffixes; keep TMPDIR short.
export TMPDIR="${GROUND3D_TMPDIR:-/home/huangxiang/codes/guiming/.g3rf_tmp}"
TMPDIR="$(realpath -m -- "$TMPDIR")"
case "$TMPDIR" in
  /home/huangxiang/codes/guiming/*) ;;
  *) echo 'TMPDIR must resolve inside /home/huangxiang/codes/guiming' >&2; exit 2 ;;
esac
if [[ "${#TMPDIR}" -gt 70 ]]; then
  echo 'TMPDIR is too long for multiprocessing Unix sockets (maximum 70 bytes)' >&2
  exit 2
fi
export PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
mkdir -p "$HF_HOME" "$TMPDIR"
cd "$PROJECT"
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="${#GPUS[@]}" tools/cache_first_stage.py "$CONFIG" "$@"
