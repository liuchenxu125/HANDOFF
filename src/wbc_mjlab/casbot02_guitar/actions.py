"""Twelve leg actions; both guitar robot arms hold the default Casbot02 pose."""

from dataclasses import dataclass

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from wbc_mjlab.casbot02_actions import ARM_JOINT_NAMES


class GuitarLegWithFixedArmsAction(JointPositionAction):
  def __init__(self, cfg: JointPositionActionCfg, env):
    super().__init__(cfg=cfg, env=env)
    arm_ids, _ = self._entity.find_joints(ARM_JOINT_NAMES, preserve_order=True)
    self._arm_ids = torch.tensor(arm_ids, device=self.device, dtype=torch.long)
    self._arm_default = self._entity.data.default_joint_pos[:, self._arm_ids].clone()

  def apply_actions(self) -> None:
    super().apply_actions()
    # Position-controlled holding, not XML welding; the motor limits/delays
    # still apply. Deliberately no dependency on measured knee positions.
    self._entity.set_joint_position_target(self._arm_default, joint_ids=self._arm_ids)


@dataclass(kw_only=True)
class GuitarLegWithFixedArmsActionCfg(JointPositionActionCfg):
  def build(self, env) -> GuitarLegWithFixedArmsAction:
    return GuitarLegWithFixedArmsAction(self, env)
