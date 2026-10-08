"""CPU regression checks for merged translation/turning air-time rewards."""

import unittest
from types import SimpleNamespace

import torch

from wbc_mjlab.casbot02_config import casbot02_loco_teacher_env_cfg
from wbc_mjlab.rewards import (
  command_conditioned_feet_air_time,
  feet_air_time,
  locomotion_feet_air_time,
)


class Casbot02AirTimeRewardTest(unittest.TestCase):
  def _make_env(self, commands, current_air_time, last_air_time, first_contact):
    sensor = SimpleNamespace(
      data=SimpleNamespace(
        current_air_time=torch.tensor(current_air_time),
        last_air_time=torch.tensor(last_air_time),
      ),
      compute_first_contact=lambda dt: torch.tensor(first_contact, dtype=torch.bool),
    )
    commands = torch.tensor(commands)
    return SimpleNamespace(
      scene={"feet_ground_contact": sensor},
      command_manager=SimpleNamespace(get_command=lambda name: commands),
      step_dt=0.02,
      extras={"log": {}},
    )

  def test_config_preserves_weight_and_thresholds_in_one_term(self):
    cfg = casbot02_loco_teacher_env_cfg()
    self.assertNotIn("air_time", cfg.rewards)
    self.assertNotIn("turning_air_time", cfg.rewards)
    term = cfg.rewards["feet_air_time"]
    self.assertIs(term.func, locomotion_feet_air_time)
    self.assertEqual(term.weight, 1.5)
    self.assertEqual(term.params, {
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "feet_air_time_target": None,
      "threshold_min": 0.05,
      "translation_threshold_max": 0.60,
      "turning_threshold_max": 0.42,
      "command_threshold": 0.2,
      "turning_linear_threshold": 0.2,
      "turning_angular_threshold": 0.2,
    })

  def test_matches_sum_of_old_terms_for_all_command_types(self):
    commands = [
      [0.5, 0.0, 0.0], [-0.5, 0.0, 0.0],
      [0.0, 0.0, 0.8], [0.0, 0.0, -0.8], [0.0, 0.0, 0.0],
      [0.1, 0.0, 0.8], [0.4, 0.0, 0.8],
      [0.05, 0.0, 0.0], [0.051, 0.0, 0.0],
      [0.0, 0.0, 0.2], [0.0, 0.0, 0.201],
    ]
    n = len(commands)
    for current in ([0.0, 0.0], [0.05, 0.42], [0.1, 0.4], [0.6, 0.8]):
      for contact in ([False, False], [True, False], [True, True]):
        with self.subTest(current=current, contact=contact):
          env = self._make_env(commands, [current] * n, [[0.8, 1.2]] * n, [contact] * n)
          cfg = SimpleNamespace(params={})
          translation = feet_air_time(cfg, env)(
            env, "feet_ground_contact", feet_air_time_target=None, uniform_cmd_name="twist"
          )
          turning = command_conditioned_feet_air_time(
            env, "feet_ground_contact", turning_only=True
          )
          actual = locomotion_feet_air_time(cfg, env)(env, "feet_ground_contact")
          self.assertTrue(torch.allclose(1.5 * actual, 1.5 * translation + 1.5 * turning))
          self.assertIn("Metrics/air_time_mean", env.extras["log"])
          self.assertIn("Metrics/air_time_max_target_mean", env.extras["log"])

  def test_translation_is_uncapped_and_only_rewarded_at_landing(self):
    env = self._make_env(
      [[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
      [[0.0, 0.0]] * 3, [[0.8, 1.2]] * 3,
      [[True, False], [False, True], [False, False]],
    )
    reward = locomotion_feet_air_time(SimpleNamespace(params={}), env)
    self.assertTrue(torch.allclose(reward(env, "feet_ground_contact"), torch.tensor([0.8, 1.2, 0.0])))

  def test_pure_turn_is_dense_and_keeps_strict_window_boundaries(self):
    env = self._make_env(
      [[0.0, 0.0, 0.8]] * 4,
      [[0.05, 0.42], [0.1, 0.0], [0.1, 0.4], [0.43, 0.8]],
      [[0.8, 1.2]] * 4, [[True, True]] * 4,
    )
    reward = locomotion_feet_air_time(SimpleNamespace(params={}), env)
    self.assertTrue(torch.equal(reward(env, "feet_ground_contact"), torch.tensor([0.0, 1.0, 2.0, 0.0])))


if __name__ == "__main__":
  unittest.main()
