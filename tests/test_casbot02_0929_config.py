"""2026-09-29 config regression checks, retaining current light-leg motors."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import unittest

import yaml

from wbc_mjlab import casbot02_constants as C
from wbc_mjlab.casbot02_config import casbot02_loco_teacher_env_cfg


REFERENCE = (
  Path(__file__).resolve().parents[1]
  / "logs/rsl_rl/casbot02_loco_teacher"
  / "2026-09-29_10-44-27_casbot02_loco_teacher/params/env.yaml"
)


class ConfigLoader(yaml.SafeLoader):
  """Read serialized Python tags as data without importing/executing them."""


def _tagged(loader, tag_suffix, node):
  if isinstance(node, yaml.ScalarNode):
    value = loader.construct_scalar(node)
  elif isinstance(node, yaml.SequenceNode):
    value = loader.construct_sequence(node, deep=True)
  else:
    value = loader.construct_mapping(node, deep=True)
  if tag_suffix == "python/tuple":
    return value
  return {"__yaml_tag__": tag_suffix, "value": value}


ConfigLoader.add_multi_constructor("tag:yaml.org,2002:", _tagged)


class Casbot02BaselineConfigTest(unittest.TestCase):
  @unittest.skipUnless(REFERENCE.is_file(), "Local training snapshot is unavailable")
  def test_training_config_matches_snapshot_except_current_light_leg_motors(self):
    reference = yaml.load(REFERENCE.read_text(), Loader=ConfigLoader)
    # The user explicitly keeps the current light-leg kp/armature and the
    # action scale derived from them, instead of restoring historical motors.
    light = C.CASBOT02_LEG_LIGHT_ACTUATOR
    baseline_light = reference["scene"]["entities"]["robot"]["articulation"]["actuators"][1]
    self.assertEqual(set(baseline_light["target_names_expr"]), set(light.target_names_expr))
    baseline_light["stiffness"] = light.stiffness
    baseline_light["armature"] = light.armature
    for name in light.target_names_expr:
      reference["actions"]["joint_pos"]["scale"][name] = C.CASBOT02_LEG_ONLY_ACTION_SCALE[name]
    cfg = casbot02_loco_teacher_env_cfg()
    # env.yaml is saved after environment initialization, when the step-zero
    # command curriculum has already narrowed the initial vx/wz ranges.
    curriculum = cfg.curriculum["command_vel"]
    env = SimpleNamespace(
      common_step_counter=0,
      command_manager=SimpleNamespace(
        get_term=lambda name: SimpleNamespace(cfg=cfg.commands[name])
      ),
    )
    curriculum.func(env, None, **curriculum.params)
    actual = yaml.load(yaml.dump(asdict(cfg), sort_keys=False), Loader=ConfigLoader)
    self.assertEqual(actual, reference)

  def test_motors_and_action_scale_follow_shared_constants_without_mutation(self):
    before = asdict(C.get_casbot02_23dof_robot_cfg())
    cfg = casbot02_loco_teacher_env_cfg()
    self.assertEqual(before, asdict(C.get_casbot02_23dof_robot_cfg()))
    self.assertEqual(asdict(cfg.scene.entities["robot"]), before)
    self.assertEqual(cfg.actions["joint_pos"].scale, C.CASBOT02_LEG_ONLY_ACTION_SCALE)

  def test_play_keeps_single_env_and_the_same_robot_action_setup(self):
    train = casbot02_loco_teacher_env_cfg()
    play = casbot02_loco_teacher_env_cfg(play=True)
    self.assertEqual(train.scene.num_envs, 4096)
    self.assertEqual(train.seed, 42)
    self.assertEqual(play.scene.num_envs, 1)
    self.assertEqual(play.actions["joint_pos"].scale, train.actions["joint_pos"].scale)
    self.assertEqual(
      asdict(play.scene.entities["robot"]), asdict(train.scene.entities["robot"])
    )
    self.assertFalse(play.observations["actor"].enable_corruption)
    self.assertNotIn("push_robot", play.events)
    self.assertEqual(play.curriculum, {})


if __name__ == "__main__":
  unittest.main()
