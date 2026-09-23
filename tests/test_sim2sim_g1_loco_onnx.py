"""CPU checks for G1 loco ONNX sim2sim observation layout."""
import importlib.util
import re
import unittest
from pathlib import Path

import numpy as np

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ACTION_SCALE
from wbc_mjlab.observations import ARM_JOINT_NAMES, BODY_JOINT_NAMES

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sim2sim_g1_loco_onnx.py"
_SPEC = importlib.util.spec_from_file_location("sim2sim_g1_loco_onnx", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)

ACTOR_OBS_DIM = _MOD.ACTOR_OBS_DIM
G1_JOINT_NAMES = _MOD.G1_JOINT_NAMES
NUM_FULL_JOINTS = _MOD.NUM_FULL_JOINTS
NUM_POLICY_ACTIONS = _MOD.NUM_POLICY_ACTIONS
gait_phase_features = _MOD.gait_phase_features
resolve_action_scale = _MOD.resolve_action_scale
resolve_default_joint_pos = _MOD.resolve_default_joint_pos


class G1LocoSim2SimTest(unittest.TestCase):
  def test_joint_order_is_xml_body_then_arm(self):
    self.assertEqual(G1_JOINT_NAMES, BODY_JOINT_NAMES + ARM_JOINT_NAMES)
    self.assertEqual(NUM_FULL_JOINTS, 29)
    self.assertEqual(NUM_POLICY_ACTIONS, 15)
    self.assertEqual(ACTOR_OBS_DIM, 86)

  def test_default_pose_is_knees_bent(self):
    q = resolve_default_joint_pos()
    self.assertEqual(q.shape, (29,))
    self.assertAlmostEqual(q[G1_JOINT_NAMES.index("left_hip_pitch_joint")], -0.312)
    self.assertAlmostEqual(q[G1_JOINT_NAMES.index("left_knee_joint")], 0.669)
    self.assertAlmostEqual(q[G1_JOINT_NAMES.index("left_shoulder_pitch_joint")], 0.2)
    self.assertAlmostEqual(q[G1_JOINT_NAMES.index("waist_yaw_joint")], 0.0)

  def test_action_scale_matches_g1_dict(self):
    scale = resolve_action_scale()
    self.assertEqual(scale.shape, (15,))
    for idx, name in enumerate(BODY_JOINT_NAMES):
      expected = None
      for pattern, value in G1_ACTION_SCALE.items():
        if re.fullmatch(pattern, name):
          expected = float(value)
          break
      self.assertIsNotNone(expected, name)
      self.assertAlmostEqual(scale[idx], expected)

  def test_standing_phase_is_zero_clock(self):
    phase = gait_phase_features(1.25, np.array([0.0, 0.0, 0.0]))
    np.testing.assert_allclose(phase, [0.0, 1.0, 0.0, 1.0])

  def test_walking_phase_has_half_cycle_offset(self):
    phase = gait_phase_features(0.0, np.array([0.5, 0.0, 0.0]))
    np.testing.assert_allclose(phase, [0.0, 1.0, 0.0, -1.0], atol=1e-6)
    phase_quarter = gait_phase_features(0.25, np.array([0.5, 0.0, 0.0]))
    np.testing.assert_allclose(phase_quarter, [1.0, 0.0, -1.0, 0.0], atol=1e-6)


if __name__ == "__main__":
  unittest.main()
