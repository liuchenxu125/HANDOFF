#!/usr/bin/env bash
# Play a CASBOT02 get-up-only AMP teacher checkpoint (assist force is zero).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHONPATH=. uv run python scripts/verify_casbot02ampj_getup_motion.py
uv run play Casbot02-AMPJ-GetUp-Teacher-Flat "$@"
