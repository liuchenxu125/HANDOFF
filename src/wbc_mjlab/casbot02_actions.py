"""Casbot02 locomotion actions shared by training and deployment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

from wbc_mjlab import casbot02_constants as C

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


ARM_SWING_GAIN = 0.5
LEFT_KNEE_NAME = "leg_l4_joint"
RIGHT_KNEE_NAME = "leg_r4_joint"
LEFT_SHOULDER_NAME = "upper_left_1_joint"
RIGHT_SHOULDER_NAME = "upper_right_1_joint"
ARM_JOINT_NAMES = tuple(
  name for name in C.CASBOT02_23DOF_JOINT_NAMES if name.startswith("upper_")
)


class Casbot02LegWithArmSwingAction(JointPositionAction):
  """12 leg actions with shoulder pitch targets derived from knee angles.

  The policy still emits only 12 leg actions.  On every physics substep, all
  arm joints are held at their default positions except the two first shoulder
  joints, which use the same cross-body swing formula as Casbot02 sim2sim.
  """

  def __init__(self, cfg: JointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    arm_ids, arm_names = self._entity.find_joints(
      ARM_JOINT_NAMES, preserve_order=True
    )
    knee_ids, _ = self._entity.find_joints(
      (LEFT_KNEE_NAME, RIGHT_KNEE_NAME), preserve_order=True
    )
    self._arm_ids = torch.tensor(arm_ids, device=self.device, dtype=torch.long)
    self._knee_ids = torch.tensor(knee_ids, device=self.device, dtype=torch.long)
    self._left_shoulder_col = arm_names.index(LEFT_SHOULDER_NAME)
    self._right_shoulder_col = arm_names.index(RIGHT_SHOULDER_NAME)
    self._arm_default = self._entity.data.default_joint_pos[:, self._arm_ids].clone()

  def apply_actions(self) -> None:
    super().apply_actions()

    left_knee = self._entity.data.joint_pos[:, self._knee_ids[0]]
    right_knee = self._entity.data.joint_pos[:, self._knee_ids[1]]
    knee_diff = left_knee - right_knee

    arm_target = self._arm_default.clone()
    arm_target[:, self._left_shoulder_col] += ARM_SWING_GAIN * knee_diff
    arm_target[:, self._right_shoulder_col] -= ARM_SWING_GAIN * knee_diff
    self._entity.set_joint_position_target(arm_target, joint_ids=self._arm_ids)


@dataclass(kw_only=True)
class Casbot02LegWithArmSwingActionCfg(JointPositionActionCfg):
  """Configuration for 12 policy actions plus computed shoulder swing."""

  def build(self, env: ManagerBasedRlEnv) -> Casbot02LegWithArmSwingAction:
    return Casbot02LegWithArmSwingAction(self, env)


__all__ = [
  "ARM_JOINT_NAMES",
  "ARM_SWING_GAIN",
  "Casbot02LegWithArmSwingAction",
  "Casbot02LegWithArmSwingActionCfg",
  "LEFT_KNEE_NAME",
  "LEFT_SHOULDER_NAME",
  "RIGHT_KNEE_NAME",
  "RIGHT_SHOULDER_NAME",
]
