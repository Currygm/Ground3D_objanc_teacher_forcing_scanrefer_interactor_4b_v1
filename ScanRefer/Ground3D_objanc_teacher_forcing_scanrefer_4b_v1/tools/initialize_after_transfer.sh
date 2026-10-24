#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
export HF_HOME="$PROJECT/cache/huggingface" XDG_CACHE_HOME="$PROJECT/cache"
"$PROJECT/env/bin/python" tools/verify_initialize.py > docs/full_weight_initialization.log 2>&1
"$PROJECT/env/bin/python" tools/verify_sources.py > docs/source_preservation.log 2>&1
"$PROJECT/env/bin/python" -m pytest -q > docs/full_cpu_post_transfer.log 2>&1
