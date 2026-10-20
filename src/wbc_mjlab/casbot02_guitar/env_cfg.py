"""Independent guitar loco task, inheriting all current CASBOT02 settings."""

from dataclasses import asdict

from mjlab.envs import ManagerBasedRlEnvCfg

from wbc_mjlab.casbot02_config import casbot02_loco_teacher_env_cfg
from wbc_mjlab.casbot02_guitar.constants import get_robot_cfg
from wbc_mjlab.casbot02_guitar.actions import GuitarLegWithFixedArmsActionCfg
from wbc_mjlab.casbot02_guitar.motor_config import LEG_ACTION_SCALE


def casbot02_guitar_loco_teacher_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = casbot02_loco_teacher_env_cfg(play=play)
  cfg.scene.entities["robot"] = get_robot_cfg()
  # Preserve all leg action settings while replacing only the arm behavior.
  cfg.actions["joint_pos"] = GuitarLegWithFixedArmsActionCfg(**asdict(cfg.actions["joint_pos"]))
  cfg.actions["joint_pos"].scale = dict(LEG_ACTION_SCALE)
  return cfg
