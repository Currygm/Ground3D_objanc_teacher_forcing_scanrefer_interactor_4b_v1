#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
mkdir -p "$PROJECT/.verification/tmp"
export TMPDIR="$PROJECT/.verification/tmp"
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1
PYTHON="/home/huangxiang/codes/hx_l/code/research/Ground3D_objanc_teacher_forcing_scanrefer_interactor_4b_v1/env/bin/python"
for script in scripts/*.sh; do bash -n "$script"; done
"$PYTHON" -m pytest tests/test_anchor_relative_attention.py tests/test_anchor_integration.py tests/test_anchor_delivery.py tests/test_loss_and_checkpoint.py tests/test_unconstrained_evaluation.py tests/test_eval_shards.py tests/test_loader_rng.py tests/test_ground3d_resume_metrics.py tests/test_ground3d_forward_count.py tests/test_ground3d_language.py tests/test_zero_obj.py -q -p no:cacheprovider --junitxml=docs/cpu_checks.xml --basetemp="$PROJECT/.verification/pytest"
