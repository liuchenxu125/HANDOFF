#!/usr/bin/env bash
# Same CASBOT02 loco settings; independent rigid-guitar robot and log directory.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

GPU_ID="0"
if [[ -n "${1:-}" && "${1:-}" != --* ]]; then
  GPU_ID="${1}"
  shift
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run train Casbot02-Guitar-Loco-Teacher-Flat \
  --agent.experiment-name casbot02_guitar_loco_teacher \
  --agent.run-name casbot02_guitar_loco_teacher \
  --env.scene.num-envs 4096 \
  "$@"
