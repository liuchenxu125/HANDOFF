#!/usr/bin/env bash
# Play a CASBOT02 AMPJ teacher checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

PYTHONPATH=. uv run python scripts/verify_casbot02ampj_motion.py
uv run play Casbot02-AMPJ-Teacher-Flat "$@"
