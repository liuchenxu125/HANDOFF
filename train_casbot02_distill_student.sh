#!/usr/bin/env bash
# Casbot02 蒸馏学生: 显式 blend 双教师 KL (AMP 转弯 + loco 前进后退)。
# 前置: 先跑 train_casbot02_loco_teacher.sh 拿到 loco 教师 checkpoint, 再把
# LOCO_TEACHER_CKPT 改成那个 model_*.pt 的绝对路径。
set -euo pipefail

# Run from repo root so the top-level ``deploy/`` namespace package is on
# sys.path (g1_constants_custom.py imports it during wbc_mjlab init).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# AMP 教师 (已训好, 真机转弯验证过).
AMP_TEACHER_CKPT="/home/liuchenxu/Desktop/amp_mjlab/amp_mjlab/logs/rsl_rl/casbot02_leg_amp_locomotion/2026-08-25_11-28-46/model_8000.pt"
# Loco 教师 (由 train_casbot02_loco_teacher.sh 训出, 训练完填这里).
LOCO_TEACHER_CKPT="logs/rsl_rl/casbot02_loco_teacher/2026-08-26_23-27-11_casbot02_loco_teacher/model_20000.pt"

GPU_ID="0"
if [[ -n "${1:-}" && "${1:-}" != --* ]]; then
  GPU_ID="${1}"
  shift
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" uv run train Casbot02-Distill-Student-Flat \
  --agent.experiment-name casbot02_distill_student \
  --agent.run-name casbot02_distill_student \
  --agent.loco-teacher-checkpoint "${LOCO_TEACHER_CKPT}" \
  --agent.amp-teacher-checkpoint "${AMP_TEACHER_CKPT}" \
  --agent.dagger-coef 0.4 \
  --agent.dagger-coef-min 0.2 \
  --env.scene.num-envs 4096 \
  "$@"
