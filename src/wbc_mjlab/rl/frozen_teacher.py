"""Reconstruct a frozen RSL-RL MLP actor (mean + std) from an amp_mjlab checkpoint.

The AMP teacher checkpoint (``model_*.pt``) is a plain RSL-RL ``ActorCritic``:
``model_state_dict`` carries ``actor.0/2/4/6.*`` (MLP), ``std`` (per-dim
Gaussian std), and ``obs_norm_state_dict`` carries the running
``_mean/_var/_std/count`` of an ``EmpiricalNormalization``.

This module rebuilds just the actor (the critic/discriminator are not needed
for distillation) as a drop-in frozen teacher exposing
``output_distribution_params = (mean, std)`` — the exact interface
HANDOFF's ``DaggerPPO`` / ``two_teacher_blend`` consume for the Gaussian KL.

Verified against ``0825-11Casbot02-Leg-AMP-Flat_model_8000.onnx`` in Step 0.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# The Casbot02 45-dim single-frame observation, term order:
#   base_ang_vel(3), projected_gravity(3), command(3),
#   joint_pos(12), joint_vel(12), actions(12)
OBS_TERM_DIMS: tuple[int, ...] = (3, 3, 3, 12, 12, 12)
OBS_HISTORY: int = 4


def build_term_to_time_permutation(
  term_dims: tuple[int, ...] = OBS_TERM_DIMS,
  history: int = OBS_HISTORY,
) -> torch.Tensor:
  """Permutation indices mapping term-major -> time-major.

  mjlab 1.4.0 (HANDOFF) flattens history TERM-major: each term's H frames are
  contiguous (``[term0_t0..term0_t3, term1_t0.., ...]``). mjlab 1.2.0 (the AMP
  teacher) used ``history_ordering="time"``: each frame's 45 dims are
  contiguous (``[frame0(45), frame1(45), ...]``). Both order frames
  oldest->newest. This returns ``perm`` such that::

      obs_time_major[p] == obs_term_major[perm[p]]
  """
  frame_dim = sum(term_dims)
  term_offsets = [0]
  for d in term_dims:
    term_offsets.append(term_offsets[-1] + d)

  perm = torch.zeros(frame_dim * history, dtype=torch.long)
  for t, d in enumerate(term_dims):
    for h in range(history):
      for dd in range(d):
        time_major_pos = h * frame_dim + term_offsets[t] + dd
        term_major_pos = term_offsets[t] * history + h * d + dd
        perm[time_major_pos] = term_major_pos
  return perm


def term_major_to_time_major(
  obs: torch.Tensor,
  perm: torch.Tensor | None = None,
) -> torch.Tensor:
  """Reorder a term-major obs (mjlab 1.4.0) to time-major (mjlab 1.2.0)."""
  if perm is None:
    perm = build_term_to_time_permutation()
  return obs[..., perm.to(obs.device)]


def _resolve_activation(name: str) -> nn.Module:
  if name.lower() == "elu":
    return nn.ELU()
  if name.lower() == "relu":
    return nn.ReLU()
  if name.lower() == "tanh":
    return nn.Tanh()
  raise ValueError(f"unsupported activation: {name}")


def _build_mlp(
  input_dim: int,
  hidden_dims: list[int] | tuple[int, ...],
  output_dim: int,
  activation: str,
) -> nn.Sequential:
  layers: list[nn.Module] = []
  in_d = input_dim
  for h in hidden_dims:
    layers.append(nn.Linear(in_d, h))
    layers.append(_resolve_activation(activation))
    in_d = h
  layers.append(nn.Linear(in_d, output_dim))
  return nn.Sequential(*layers)


class FrozenRslRlTeacher(nn.Module):
  """Frozen (mean, std) teacher reconstructed from an RSL-RL actor checkpoint.

  Args:
    obs_dim:      actor observation dim (e.g. 180 for Casbot02 leg AMP).
    hidden_dims:  actor MLP hidden sizes (e.g. (512, 256, 128)).
    output_dim:   action dim (e.g. 12).
    activation:   "elu" (matches amp_mjlab).
    min_std:      per-dim std floor (the checkpoint's ``min_normalized_std``).
  """

  def __init__(
    self,
    obs_dim: int,
    hidden_dims: list[int] | tuple[int, ...],
    output_dim: int,
    activation: str = "elu",
    min_std: float | None = None,
  ) -> None:
    super().__init__()
    self.obs_dim = obs_dim
    self.output_dim = output_dim

    # EmpiricalNormalization running stats, stored as non-trainable buffers so
    # they move with .to(device) and survive .eval(). Initialized to the
    # checkpoint values by ``load_checkpoint``.
    self.register_buffer("running_mean", torch.zeros(1, obs_dim))
    self.register_buffer("running_std", torch.ones(1, obs_dim))

    # Optional input permutation (term-major -> time-major). Identity by
    # default (no-op); ``use_time_major_input`` swaps in the real permutation
    # for teachers trained in mjlab 1.2.0 with ``history_ordering="time"``.
    self.register_buffer(
      "obs_permutation",
      torch.arange(obs_dim, dtype=torch.long),
      persistent=False,
    )

    self.mlp = _build_mlp(obs_dim, hidden_dims, output_dim, activation)

    # Per-dim Gaussian std (RSL-RL "scalar" std type -> one learned value per
    # action dim). Kept as a plain buffer: the teacher is frozen, we never
    # optimize it, and this avoids it being grabbed by an optimizer.
    self.register_buffer("std", torch.ones(output_dim))
    self.min_std = min_std

    # Interface mirroring HANDOFF's teacher models: after forward(), this holds
    # (mean, std) so ``_gaussian_kl_per_dim`` / ``two_teacher_blend`` can read it.
    self.output_distribution_params: tuple[torch.Tensor, torch.Tensor] | None = None

  def load_checkpoint(self, path: str, device: str | torch.device = "cpu") -> "FrozenRslRlTeacher":
    """Load an amp_mjlab (mjlab 1.2.0) checkpoint.

    ``model_state_dict`` carries ``actor.0/2/4/6.*`` + ``std``, and a separate
    ``obs_norm_state_dict`` carries the EmpiricalNormalization stats.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    msd = ckpt["model_state_dict"]
    obs_norm = ckpt["obs_norm_state_dict"]

    # actor.0/2/4/6.* -> mlp.0/2/4/6.*
    actor_state = {k[len("actor."):]: v for k, v in msd.items() if k.startswith("actor.")}
    self.mlp.load_state_dict(actor_state, strict=True)

    self.running_mean.copy_(obs_norm["_mean"].reshape(1, -1).to(device))
    self.running_std.copy_(obs_norm["_std"].reshape(1, -1).to(device))
    self.std.copy_(msd["std"].reshape(-1).to(device))

    self.to(device)
    self.eval()
    return self

  def load_handoff_checkpoint(
    self, path: str, device: str | torch.device = "cpu"
  ) -> "FrozenRslRlTeacher":
    """Load a HANDOFF (rsl-rl 5.x MLPModel) checkpoint.

    ``actor_state_dict`` carries ``mlp.*``, ``distribution.std_param`` and
    ``obs_normalizer.*`` — the format produced by training the loco teacher in
    this repo (mjlab 1.4.0).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    actor_sd = ckpt.get("actor_state_dict", ckpt.get("model_state_dict", {}))
    if not actor_sd:
      raise KeyError(f"No actor weights found in checkpoint: {path}")

    self.mlp.load_state_dict(
      {k[len("mlp."):]: v for k, v in actor_sd.items() if k.startswith("mlp.")},
      strict=True,
    )

    if "obs_normalizer._mean" in actor_sd:
      self.running_mean.copy_(actor_sd["obs_normalizer._mean"].reshape(1, -1).to(device))
      self.running_std.copy_(actor_sd["obs_normalizer._std"].reshape(1, -1).to(device))
    std_key = "distribution.std_param" if "distribution.std_param" in actor_sd else "std"
    if std_key in actor_sd:
      self.std.copy_(actor_sd[std_key].reshape(-1).to(device))

    self.to(device)
    self.eval()
    return self

  def _normalize(self, obs: torch.Tensor) -> torch.Tensor:
    obs = obs[..., self.obs_permutation.to(obs.device)]
    return (obs - self.running_mean) / self.running_std

  def forward(
    self,
    obs: torch.Tensor,
    stochastic_output: bool = False,  # noqa: ARG002  (kept for interface parity)
  ) -> torch.Tensor:
    """Return the deterministic mean and cache ``(mean, std)``.

    ``obs`` may be a flat ``[B, obs_dim]`` tensor (what the Casbot02 student
    env emits). ``stochastic_output`` is ignored — a frozen teacher only needs
    to expose its distribution params for the KL, never to sample.
    """
    mean = self.mlp(self._normalize(obs))
    std = self.std.clamp_min(self.min_std) if self.min_std is not None else self.std
    self.output_distribution_params = (mean, std.expand_as(mean))
    return mean

  @torch.no_grad()
  def mean_std(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convenience: return (mean, std) directly, detached."""
    mean = self.mlp(self._normalize(obs))
    std = self.std.clamp_min(self.min_std) if self.min_std is not None else self.std
    return mean, std.expand_as(mean)

  def use_time_major_input(
    self,
    term_dims: tuple[int, ...] = OBS_TERM_DIMS,
    history: int = OBS_HISTORY,
  ) -> "FrozenRslRlTeacher":
    """Enable the term-major -> time-major input permutation (mjlab 1.2 teacher)."""
    self.obs_permutation.copy_(build_term_to_time_permutation(term_dims, history))
    return self


__all__ = [
  "FrozenRslRlTeacher",
  "build_term_to_time_permutation",
  "term_major_to_time_major",
  "OBS_TERM_DIMS",
  "OBS_HISTORY",
]
