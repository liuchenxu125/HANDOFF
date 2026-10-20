"""Same PPO and network settings, with separate guitar log names."""

from mjlab.rl import RslRlOnPolicyRunnerCfg
from wbc_mjlab.casbot02_rl_cfg import casbot02_loco_teacher_runner_cfg


def casbot02_guitar_loco_teacher_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = casbot02_loco_teacher_runner_cfg()
  cfg.experiment_name = "casbot02_guitar_loco_teacher"
  cfg.run_name = "casbot02_guitar_loco_teacher"
  return cfg
