"""MDP terms for the dedicated CASBOT02AMPJ get-up teacher.

The reward and upward-assistance curriculum follow LeggedLab's G1 GET-UP
task, with heights and force levels scaled for CASBOT02AMPJ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class UpwardAssistCommand(CommandTerm):
  """Per-environment, episode-constant upward assistance in newtons."""

  cfg: "UpwardAssistCommandCfg"

  def __init__(self, cfg: "UpwardAssistCommandCfg", env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self._command = torch.full(
      (self.num_envs, 1), cfg.force, dtype=torch.float32, device=self.device
    )

  @property
  def command(self) -> torch.Tensor:
    return self._command

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # Curriculum state must survive command resets.  The force changes only in
    # ``assist_force_level`` after an environment completes an episode.
    del env_ids

  def _update_command(self) -> None:
    pass

  def _update_metrics(self) -> None:
    pass


@dataclass(kw_only=True)
class UpwardAssistCommandCfg(CommandTermCfg):
  force: float

  def build(self, env: ManagerBasedRlEnv) -> UpwardAssistCommand:
    return UpwardAssistCommand(self, env)


def apply_upward_assist(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  command_name: str,
  asset_cfg: SceneEntityCfg,
) -> None:
  """Apply the current assistance force to one body in the world +Z direction."""
  asset: Entity = env.scene[asset_cfg.name]
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device)
  if not isinstance(asset_cfg.body_ids, list) or len(asset_cfg.body_ids) != 1:
    raise ValueError("Upward assistance requires exactly one resolved body")

  command = env.command_manager.get_command(command_name)
  forces = torch.zeros((len(env_ids), 1, 3), device=env.device)
  torques = torch.zeros_like(forces)
  forces[:, 0, 2] = command[env_ids, 0]
  asset.write_external_wrench_to_sim(
    forces,
    torques,
    env_ids=env_ids,
    body_ids=asset_cfg.body_ids,
  )


def assist_force_level(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice,
  command_name: str,
  reward_term_name: str,
  success_fraction: float,
  force_step: float,
  minimum_force: float,
) -> dict[str, torch.Tensor]:
  """Reduce assistance for environments whose previous episode was successful.

  Rewards in mjlab are integrated over time.  Dividing the episode sum by
  ``max_episode_length_s`` recovers the mean reward rate, matching the
  LeggedLab threshold ``success_fraction * reward_weight``.
  """
  command_term = env.command_manager.get_term(command_name)
  if not isinstance(command_term, UpwardAssistCommand):
    raise TypeError(
      f"Command {command_name!r} must be UpwardAssistCommand, got "
      f"{type(command_term).__name__}"
    )
  episode_sum = env.reward_manager._episode_sums[reward_term_name]
  reward_cfg = env.reward_manager.get_term_cfg(reward_term_name)
  mean_reward_rate = episode_sum[env_ids] / env.max_episode_length_s
  successful = mean_reward_rate > success_fraction * reward_cfg.weight

  current_force = command_term._command[env_ids, 0]
  reduced_force = torch.clamp(current_force - force_step, min=minimum_force)
  command_term._command[env_ids, 0] = torch.where(
    successful, reduced_force, current_force
  )
  selected_force = command_term._command[env_ids, 0]
  return {
    "force_n": torch.mean(selected_force),
    "success_rate": torch.mean(successful.float()),
  }


def _phase3_mask(
  asset: Entity,
  target_base_height_phase3: float,
) -> torch.Tensor:
  return asset.data.root_link_pos_w[:, 2] > target_base_height_phase3


def ang_vel_xy(
  env: ManagerBasedRlEnv,
  target_base_height_phase3: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  active = _phase3_mask(asset, target_base_height_phase3)
  error = torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)
  return torch.exp(-2.0 * error) * active


def lin_vel_xy(
  env: ManagerBasedRlEnv,
  target_base_height_phase3: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  active = _phase3_mask(asset, target_base_height_phase3)
  error = torch.sum(torch.square(asset.data.root_link_lin_vel_b[:, :2]), dim=1)
  return torch.exp(-5.0 * error) * active


def target_orientation(
  env: ManagerBasedRlEnv,
  target_base_height_phase3: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  active = _phase3_mask(asset, target_base_height_phase3)
  error = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  return torch.exp(-5.0 * error) * active


def target_base_height(
  env: ManagerBasedRlEnv,
  base_height_target: float,
  target_base_height_phase3: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  base_height = asset.data.root_link_pos_w[:, 2]
  active = base_height > target_base_height_phase3
  return torch.exp(-20.0 * torch.abs(base_height - base_height_target)) * active


def target_joint_deviation_l2(
  env: ManagerBasedRlEnv,
  target_base_height_phase3: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  active = _phase3_mask(asset, target_base_height_phase3)
  joint_error = (
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  )
  return torch.sum(torch.square(joint_error), dim=1) * active


__all__ = [
  "UpwardAssistCommandCfg",
  "ang_vel_xy",
  "apply_upward_assist",
  "assist_force_level",
  "lin_vel_xy",
  "target_base_height",
  "target_joint_deviation_l2",
  "target_orientation",
]
