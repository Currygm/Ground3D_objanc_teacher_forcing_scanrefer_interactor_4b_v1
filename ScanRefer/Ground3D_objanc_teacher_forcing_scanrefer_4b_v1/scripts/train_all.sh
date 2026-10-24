#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$PROJECT/scripts/train_answer.sh" "$@"
bash "$PROJECT/scripts/train_joint.sh" --answer-checkpoint "$PROJECT/outputs/open_answer/latest.pt"
