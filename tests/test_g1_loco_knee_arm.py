"""CPU checks for G1 loco knee-mapped arm swing (no physics)."""
import unittest

import torch

from wbc_mjlab.actions import (
  ARM_SWING_GAIN,
  G1LocoTeacherActionCfg,
  knee_diff_shoulder_offsets,
)
from wbc_mjlab.casbot02_actions import ARM_SWING_GAIN as CASBOT02_ARM_SWING_GAIN
from wbc_mjlab.config import unitree_g1_loco_teacher_flat_nobv_stable_env_cfg
from wbc_mjlab.observations import BODY_JOINT_NAMES


class G1LocoKneeArmTest(unittest.TestCase):
  def test_gain_matches_casbot02(self):
    self.assertEqual(ARM_SWING_GAIN, CASBOT02_ARM_SWING_GAIN)
    self.assertEqual(ARM_SWING_GAIN, 0.5)

  def test_knee_diff_formula(self):
    left = torch.tensor([0.8, 0.4])
    right = torch.tensor([0.3, 0.4])
    left_off, right_off = knee_diff_shoulder_offsets(left, right)
    self.assertTrue(torch.allclose(left_off, torch.tensor([0.25, 0.0])))
    self.assertTrue(torch.allclose(right_off, torch.tensor([-0.25, 0.0])))

  def test_stable_loco_cfg_drops_motion_curriculum(self):
    cfg = unitree_g1_loco_teacher_flat_nobv_stable_env_cfg()
    self.assertNotIn("motion", cfg.commands)
    self.assertNotIn("loco_arm_blend", cfg.curriculum)
    action = cfg.actions["joint_pos"]
    self.assertIsInstance(action, G1LocoTeacherActionCfg)
    self.assertEqual(action.arm_mode, "knee_swing")
    self.assertEqual(
      cfg.rewards["stand_pose"].params["asset_cfg"].joint_names,
      BODY_JOINT_NAMES,
    )


if __name__ == "__main__":
  unittest.main()
