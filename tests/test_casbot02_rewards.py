"""CPU checks for Casbot02 loco torque/energy rewards."""
import unittest
from types import SimpleNamespace

import torch

from wbc_mjlab.casbot02_config import (
  _ANKLE_PITCH_JOINT_WEIGHTS,
  casbot02_loco_teacher_env_cfg,
)
from wbc_mjlab.casbot02_constants import CASBOT02_LEG_ONLY_JOINT_NAMES
from wbc_mjlab.rewards import (
  applied_torque_limits_by_ratio,
  joint_energy,
  joint_torques_l2,
)


class Casbot02TorqueEnergyRewardTest(unittest.TestCase):
  def test_loco_cfg_weights_only_ankle_pitch(self):
    cfg = casbot02_loco_teacher_env_cfg()
    # Legacy normalized-over-limit term is not used; InstinctLab-style terms are.
    self.assertNotIn("joint_torques_l2", cfg.rewards)
    self.assertNotIn("dof_torque_limits", cfg.rewards)
    # joint_energy is currently disabled (commented out) in the loco config.
    self.assertNotIn("joint_energy", cfg.rewards)
    # dof_torques_l2: ankle-pitch only (weight is tuned by hand, just check sign).
    # Currently disabled (commented out) in the loco config; when re-enabled,
    # it must target only ankle pitch via _ANKLE_PITCH_JOINT_WEIGHTS.
    if "dof_torques_l2" in cfg.rewards:
      self.assertLess(cfg.rewards["dof_torques_l2"].weight, 0.0)
      self.assertEqual(
        cfg.rewards["dof_torques_l2"].params["joint_weights"],
        _ANKLE_PITCH_JOINT_WEIGHTS,
      )
    # torque_limits: ankle pitch only (leg_[lr]5), 0.8 ratio.
    self.assertEqual(cfg.rewards["torque_limits"].weight, -0.01)
    self.assertEqual(cfg.rewards["torque_limits"].params["limit_ratio"], 0.8)
    self.assertEqual(
      list(cfg.rewards["torque_limits"].params["asset_cfg"].actuator_names),
      ["leg_l5_joint", "leg_r5_joint"],
    )
    # feet_air_time: G1/HANDOFF landing-time reward gated by twist command.
    # Currently disabled (commented out) in the loco config; when re-enabled,
    # target=None means reward any lift proportional to air time (no cap).
    self.assertNotIn("air_time", cfg.rewards)
    if "feet_air_time" in cfg.rewards:
      self.assertGreater(cfg.rewards["feet_air_time"].weight, 0.0)
      self.assertIsNone(
        cfg.rewards["feet_air_time"].params["feet_air_time_target"]
      )
    pitches = {"leg_l5_joint", "leg_r5_joint"}
    for name, weight in zip(CASBOT02_LEG_ONLY_JOINT_NAMES, _ANKLE_PITCH_JOINT_WEIGHTS):
      self.assertEqual(weight, 1.0 if name in pitches else 0.0)

  def test_joint_energy_is_abs_vel_times_abs_torque(self):
    vel = torch.tensor([[1.0, -2.0], [0.5, 0.0]])
    tau = torch.tensor([[3.0, 4.0], [-8.0, 2.0]])
    asset = SimpleNamespace(
      data=SimpleNamespace(joint_vel=vel, actuator_force=tau)
    )
    env = SimpleNamespace(scene={"robot": asset})
    asset_cfg = SimpleNamespace(name="robot", joint_ids=[0, 1], actuator_ids=[0, 1])
    out = joint_energy(env, asset_cfg)
    self.assertTrue(torch.allclose(out, torch.tensor([11.0, 4.0])))

  def test_ankle_weights_zero_out_other_joints(self):
    tau = torch.tensor([[10.0, 2.0, 3.0]])
    vel = torch.tensor([[1.0, 4.0, 5.0]])
    asset = SimpleNamespace(
      data=SimpleNamespace(joint_vel=vel, actuator_force=tau)
    )
    env = SimpleNamespace(scene={"robot": asset})
    asset_cfg = SimpleNamespace(name="robot", joint_ids=[0, 1, 2], actuator_ids=[0, 1, 2])
    weights = (0.0, 1.0, 1.0)
    self.assertTrue(
      torch.allclose(joint_torques_l2(env, asset_cfg, weights), torch.tensor([13.0]))
    )
    self.assertTrue(
      torch.allclose(joint_energy(env, asset_cfg, weights), torch.tensor([23.0]))
    )

  def test_applied_torque_limits_by_ratio_penalizes_excess_only(self):
    # Two actuators: limit 100 each. limit_ratio 0.8 → threshold 80.
    # actuator_force = [70, 90] → excess = [0, 10] → sum of squares = 100.
    tau = torch.tensor([[70.0, 90.0]])
    force_range = torch.tensor([[-100.0, 100.0], [-100.0, 100.0]])
    asset = SimpleNamespace(data=SimpleNamespace(actuator_force=tau))
    model = SimpleNamespace(actuator_forcerange=force_range)
    env = SimpleNamespace(scene={"robot": asset}, sim=SimpleNamespace(model=model))
    asset_cfg = SimpleNamespace(name="robot", actuator_ids=[0, 1])
    out = applied_torque_limits_by_ratio(env, asset_cfg, limit_ratio=0.8)
    self.assertTrue(torch.allclose(out, torch.tensor([100.0])))
    # Below the threshold → no penalty.
    tau2 = torch.tensor([[10.0, 20.0]])
    env.scene["robot"].data.actuator_force = tau2
    self.assertTrue(torch.allclose(applied_torque_limits_by_ratio(env, asset_cfg), torch.tensor([0.0])))


if __name__ == "__main__":
  unittest.main()
