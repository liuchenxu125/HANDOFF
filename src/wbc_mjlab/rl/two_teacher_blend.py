"""Explicit-blend two-teacher KL distillation for the Casbot02 case.

This is the distilled essence of HANDOFF's ``DaggerPPO.update`` body-blend
block (``src/wbc_mjlab/rl/algorithms.py``), stripped to the user's exact
setup and made self-contained:

  * two frozen teachers — AMP (good at in-place turn) and loco (good at
    forward/back) — that share the SAME observation (180 = 45 x 4-frame
    history) and the SAME 12-dim leg action space;
  * a student pi_s trained by PPO in the same env;
  * an EXPLICIT gate (no learned MoE here) that is a pure function of the
    velocity command (vx, vy, wz): forward-ness routes to the loco teacher,
    turn-ness routes to the AMP teacher.

The per-step loss contribution is::

    loss_distill = gate_loco(cmd) * coef_loco * KL(pi_s || pi_loco)
                 + gate_amp (cmd) * coef_amp  * KL(pi_s || pi_amp)

where ``gate_loco + gate_amp == 1`` and the KL is a per-dimension Gaussian
KL (exactly ``_gaussian_kl_per_dim`` from HANDOFF). The two coefficients
are cosine-annealed over training (HANDOFF pattern: dagger_coef 0.4 -> 0.2).

Why this maps cleanly onto HANDOFF's machinery: in HANDOFF the WBC/loco
teachers are blended on the SAME body slice with
``(1-blend)*KL(wbc) + blend*KL(loco)`` where ``blend = sigmoid(||v_cmd||)``.
Here the two teachers are AMP/loco instead of WBC/loco, and the blend signal
is the turn-vs-forward ratio instead of a scalar speed — same algebra.
"""

from __future__ import annotations

import math

import torch


def gaussian_kl_per_dim(
  student_mu: torch.Tensor,
  student_std: torch.Tensor,
  teacher_mu: torch.Tensor,
  teacher_std: torch.Tensor,
) -> torch.Tensor:
  """Per-dimension KL(student || teacher) for diagonal Gaussians.

  Identical to HANDOFF's ``_gaussian_kl_per_dim`` (algorithms.py). All four
  inputs are ``[B, D]``. Returns ``[B, D]``.
  """
  return torch.log(teacher_std / student_std) + (
    student_std.pow(2) + (student_mu - teacher_mu).pow(2)
  ) / (2.0 * teacher_std.pow(2)) - 0.5


def explicit_blend_gate(
  cmd: torch.Tensor,
  eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Turn-vs-forward gate from the velocity command.

  ``cmd`` is ``[B, 3]`` = (vx, vy, wz). Returns ``(gate_loco, gate_amp)``,
  each ``[B]`` in [0, 1] with ``gate_loco + gate_amp == 1``.

    turn_ratio = |wz| / (||vx,vy|| + |wz| + eps)
    gate_loco  = 1 - turn_ratio    # forward/back -> trust loco
    gate_amp   = turn_ratio        # in-place turn -> trust AMP

  When the robot stands still (both ~0) turn_ratio -> 0, so it defaults to
  the loco teacher (which also supervises standing via its track-* rewards
  at cmd ~ 0). The ``eps`` keeps the ratio defined at exactly zero command.
  """
  speed = torch.norm(cmd[:, :2], dim=-1)       # [B]
  turn = cmd[:, 2].abs()                        # [B]
  turn_ratio = turn / (speed + turn + eps)      # [B] in [0, 1]
  gate_amp = turn_ratio
  gate_loco = 1.0 - turn_ratio
  return gate_loco, gate_amp


def two_teacher_kl_loss(
  student_mu: torch.Tensor,
  student_std: torch.Tensor,
  loco_mu: torch.Tensor,
  loco_std: torch.Tensor,
  amp_mu: torch.Tensor,
  amp_std: torch.Tensor,
  cmd: torch.Tensor,
  coef_loco: float,
  coef_amp: float,
  eps: float = 1e-4,
) -> tuple[torch.Tensor, dict[str, float]]:
  """Compute the explicit-blend distillation loss for one mini-batch.

  Args:
    student_*: ``[B, 12]`` student distribution params.
    loco_*:    ``[B, 12]`` frozen loco teacher params (``torch.no_grad()``).
    amp_*:     ``[B, 12]`` frozen AMP  teacher params (``torch.no_grad()``).
    cmd:       ``[B, 3]`` velocity command (vx, vy, wz) for this batch.
    coef_loco / coef_amp: current (annealed) KL weights.

  Returns ``(loss, logs)`` where ``loss`` is a scalar to add to the PPO loss
  and ``logs`` carries tensorboard keys (``loco_kl``, ``amp_kl``).
  """
  B = student_mu.shape[0]
  gate_loco, gate_amp = explicit_blend_gate(cmd, eps=eps)

  kl_loco = gaussian_kl_per_dim(student_mu, student_std, loco_mu, loco_std)  # [B,12]
  kl_amp = gaussian_kl_per_dim(student_mu, student_std, amp_mu, amp_std)     # [B,12]

  # Weighted mean over (env, dim). gate is [B] -> unsqueeze to [B,1] so it
  # weights every dim of an env equally (matching HANDOFF's body-blend mean).
  loco_loss = (kl_loco * gate_loco.unsqueeze(-1)).sum() / (B * kl_loco.shape[-1]) * coef_loco
  amp_loss = (kl_amp * gate_amp.unsqueeze(-1)).sum() / (B * kl_amp.shape[-1]) * coef_amp

  logs = {
    "loco_kl": loco_loss.item(),
    "amp_kl": amp_loss.item(),
    "mean_gate_amp": gate_amp.mean().item(),
  }
  return loco_loss + amp_loss, logs


def cosine_anneal_coef(
  coef_init: float,
  coef_min: float,
  update_counter: int,
  anneal_steps: int,
) -> float:
  """Cosine-anneal a KL coefficient (HANDOFF ``_update_dagger_coef`` pattern)."""
  if anneal_steps <= 0 or update_counter >= anneal_steps:
    return coef_min
  progress = update_counter / anneal_steps
  cosine = 0.5 * (1 + math.cos(math.pi * progress))
  return coef_min + (coef_init - coef_min) * cosine


__all__ = [
  "gaussian_kl_per_dim",
  "explicit_blend_gate",
  "two_teacher_kl_loss",
  "cosine_anneal_coef",
]
