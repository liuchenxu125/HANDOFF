"""Casbot02-specific velocity command sampling for the loco teacher."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand
from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import quat_apply


class Casbot02VelocityCommand(UniformVelocityCommand):
  """Uniform velocity command with a mutually exclusive pure-turn cohort.

  The configured standing, pure-turn, forward and heading fractions are
  interpreted as absolute, mutually exclusive fractions of environments.
  Remaining environments retain independently sampled ``vx``/``wz`` commands.
  """

  cfg: Casbot02VelocityCommandCfg

  def __init__(self, cfg: Casbot02VelocityCommandCfg, env) -> None:
    super().__init__(cfg, env)
    self.is_turning_env = torch.zeros_like(self.is_standing_env)

  def _sample_turning_yaw_rate(self, count: int) -> torch.Tensor:
    if count == 0:
      return torch.empty(0, device=self.device)
    lo, hi = self.cfg.ranges.ang_vel_z
    if not (lo < 0.0 < hi):
      raise ValueError(
        "Pure-turn sampling requires ang_vel_z to span both signs, "
        f"got {(lo, hi)}"
      )
    sign_positive = torch.rand(count, device=self.device) >= 0.5
    max_magnitude = torch.where(
      sign_positive,
      torch.full((count,), hi, device=self.device),
      torch.full((count,), -lo, device=self.device),
    )
    min_magnitude = torch.clamp(
      torch.full_like(max_magnitude, self.cfg.min_turning_ang_vel),
      max=max_magnitude,
    )
    magnitude = min_magnitude + torch.rand(count, device=self.device) * (
      max_magnitude - min_magnitude
    )
    return torch.where(sign_positive, magnitude, -magnitude)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    count = len(env_ids)
    if count == 0:
      return

    random = torch.empty(count, device=self.device)
    self.vel_command_b[env_ids, 0] = random.uniform_(*self.cfg.ranges.lin_vel_x)
    self.vel_command_b[env_ids, 1] = random.uniform_(*self.cfg.ranges.lin_vel_y)
    self.vel_command_b[env_ids, 2] = random.uniform_(*self.cfg.ranges.ang_vel_z)

    if self.cfg.heading_command:
      assert self.cfg.ranges.heading is not None
      self.heading_target[env_ids] = random.uniform_(*self.cfg.ranges.heading)

    mode = torch.rand(count, device=self.device)
    standing_end = self.cfg.rel_standing_envs
    turning_end = standing_end + self.cfg.rel_turning_envs
    forward_end = turning_end + self.cfg.rel_forward_envs
    heading_end = forward_end + (
      self.cfg.rel_heading_envs if self.cfg.heading_command else 0.0
    )

    standing = mode < standing_end
    turning = (mode >= standing_end) & (mode < turning_end)
    forward = (mode >= turning_end) & (mode < forward_end)
    heading = (mode >= forward_end) & (mode < heading_end)

    self.is_standing_env[env_ids] = standing
    self.is_turning_env[env_ids] = turning
    self.is_forward_env[env_ids] = forward
    self.is_heading_env[env_ids] = heading

    # Pure in-place turn: vx=vy=0, non-trivial yaw rate of either sign.
    turning_ids = env_ids[turning]
    self.vel_command_b[turning_ids, :2] = 0.0
    self.vel_command_b[turning_ids, 2] = self._sample_turning_yaw_rate(
      len(turning_ids)
    )

    # Preserve the existing forward-only curriculum coverage.
    forward_ids = env_ids[forward]
    self.vel_command_b[forward_ids, 0] = (
      self.vel_command_b[forward_ids, 0].abs().clamp(min=0.3)
    )
    self.vel_command_b[forward_ids, 1:] = 0.0

    # World-frame commands are unused by Casbot02, but retain base-class
    # behavior for any future non-zero configuration.
    self.is_world_env[env_ids] = (
      torch.rand(count, device=self.device) <= self.cfg.rel_world_envs
    ) & ~(standing | turning | forward | heading)
    self.vel_command_w[env_ids] = self.vel_command_b[env_ids]

    init_velocity_mask = (
      torch.rand(count, device=self.device) < self.cfg.init_velocity_prob
    )
    init_velocity_env_ids = env_ids[init_velocity_mask]
    if len(init_velocity_env_ids) > 0:
      root_pos = self.robot.data.root_link_pos_w[init_velocity_env_ids]
      root_quat = self.robot.data.root_link_quat_w[init_velocity_env_ids]
      lin_vel_b = self.robot.data.root_link_lin_vel_b[init_velocity_env_ids].clone()
      lin_vel_b[:, :2] = self.vel_command_b[init_velocity_env_ids, :2]
      root_lin_vel_w = quat_apply(root_quat, lin_vel_b)
      root_ang_vel_b = self.robot.data.root_link_ang_vel_b[
        init_velocity_env_ids
      ].clone()
      root_ang_vel_b[:, 2] = self.vel_command_b[init_velocity_env_ids, 2]
      root_state = torch.cat(
        [root_pos, root_quat, root_lin_vel_w, root_ang_vel_b], dim=-1
      )
      self.robot.write_root_state_to_sim(root_state, init_velocity_env_ids)


@dataclass(kw_only=True)
class Casbot02VelocityCommandCfg(UniformVelocityCommandCfg):
  """Configuration for :class:`Casbot02VelocityCommand`."""

  rel_turning_envs: float = 0.2
  min_turning_ang_vel: float = 0.2

  def build(self, env) -> Casbot02VelocityCommand:
    return Casbot02VelocityCommand(self, env)

  def __post_init__(self) -> None:
    super().__post_init__()
    fractions = (
      self.rel_standing_envs,
      self.rel_turning_envs,
      self.rel_forward_envs,
      self.rel_heading_envs if self.heading_command else 0.0,
    )
    if any(value < 0.0 for value in fractions):
      raise ValueError(f"Command mode fractions must be non-negative: {fractions}")
    if sum(fractions) > 1.0 + 1.0e-8:
      raise ValueError(
        "Standing + turning + forward + heading command fractions must not "
        f"exceed 1.0, got {sum(fractions):.3f}"
      )
    if self.min_turning_ang_vel <= 0.0:
      raise ValueError("min_turning_ang_vel must be positive")


__all__ = ["Casbot02VelocityCommand", "Casbot02VelocityCommandCfg"]
