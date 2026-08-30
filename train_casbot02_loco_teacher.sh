#!/usr/bin/env bash
# Casbot02 12-leg loco teacher (velocity tracking, obs 180 / action 12).
# 训练完后, 把产生的 run 目录 model_*.pt 路径填进 train_casbot02_distill_student.sh。
set -euo pipefail

# Run from repo root so the top-level ``deploy/`` namespace package is on
# sys.path (g1_constants_custom.py imports it during wbc_mjlab init).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

GPU_ID="0"
if [[ -n "${1:-}" && "${1:-}" != --* ]]; then
  GPU_ID="${1}"
  shift
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run train Casbot02-Loco-Teacher-Flat \
  --agent.experiment-name casbot02_loco_teacher \
  --agent.run-name casbot02_loco_teacher \
  --env.scene.num-envs 4096 \
  "$@"
