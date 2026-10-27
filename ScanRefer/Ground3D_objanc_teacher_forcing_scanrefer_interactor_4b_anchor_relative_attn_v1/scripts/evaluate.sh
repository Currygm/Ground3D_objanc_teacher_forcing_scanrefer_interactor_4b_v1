#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$PROJECT/scripts/evaluate_unconstrained.sh" "$@"
