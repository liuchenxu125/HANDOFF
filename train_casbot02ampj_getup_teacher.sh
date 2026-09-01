#!/usr/bin/env bash
# CASBOT02 27-DoF get-up-only AMP teacher.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

GPU_ID="0"
if [[ -n "${1:-}" && "${1:-}" != --* ]]; then
  GPU_ID="${1}"
  shift
fi

PYTHONPATH=. uv run python scripts/verify_casbot02ampj_getup_motion.py

CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run train Casbot02-AMPJ-GetUp-Teacher-Flat \
  --agent.experiment-name casbot02_ampj_getup_teacher_flat \
  --agent.run-name casbot02_ampj_getup_teacher_flat \
  --env.scene.num-envs 4096 \
  "$@"
