"""Dedicated CASBOT02AMPJ get-up AMP teacher environment."""

from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

from wbc_mjlab import amp_mdp
from wbc_mjlab.casbot02ampj import constants as C
from wbc_mjlab.casbot02ampj import getup_mdp
from wbc_mjlab.casbot02ampj.env_cfg import (
  _apply_flat_terrain,
  _apply_robot,
  _build_amp_observations,
  _joint_asset,
)


TARGET_BASE_HEIGHT = 0.85
PHASE3_HEIGHT = 0.75
INITIAL_ASSIST_FORCE = 300.0
ASSIST_FORCE_STEP = 15.0
MINIMUM_ASSIST_FORCE = 0.0

_GETUP_DATA_DIR = (
  Path(__file__).resolve().parents[1]
  / "data"
  / "motions"
  / "casbot02ampj"
  / "amp"
  / "GetUp"
)


def _configure_commands(cfg: ManagerBasedRlEnvCfg, *, play: bool) -> None:
  """Keep the actor's twist input but make the get-up task command-free."""
  twist = cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.heading_command = False
  twist.rel_standing_envs = 1.0
  twist.rel_heading_envs = 0.0
  twist.rel_forward_envs = 0.0
  twist.ranges.lin_vel_x = (0.0, 0.0)
  twist.ranges.lin_vel_y = (0.0, 0.0)
  twist.ranges.ang_vel_z = (0.0, 0.0)
  twist.ranges.heading = None
  twist.debug_vis = False

  cfg.commands["assist_force"] = getup_mdp.UpwardAssistCommandCfg(
    force=0.0 if play else INITIAL_ASSIST_FORCE,
    resampling_time_range=(100.0, 100.0),
  )


def _configure_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
  """LeggedLab GET-UP rewards with CASBOT02 height thresholds."""
  robot_cfg = SceneEntityCfg("robot")
  joint_cfg = _joint_asset()
  phase3_params = {
    "target_base_height_phase3": PHASE3_HEIGHT,
    "asset_cfg": robot_cfg,
  }
  cfg.rewards = {
    "joint_acc_l2": RewardTermCfg(
      func=envs_mdp.joint_acc_l2,
      weight=-1.0e-7,
      params={"asset_cfg": joint_cfg},
    ),
    "action_rate_l2": RewardTermCfg(
      func=envs_mdp.action_rate_l2,
      weight=-0.005,
    ),
    "joint_torques_l2": RewardTermCfg(
      func=envs_mdp.joint_torques_l2,
      weight=-2.0e-6,
    ),
    "joint_pos_limits": RewardTermCfg(
      func=envs_mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": joint_cfg},
    ),
    "ang_vel_xy": RewardTermCfg(
      func=getup_mdp.ang_vel_xy,
      weight=2.0,
      params=phase3_params,
    ),
    "lin_vel_xy": RewardTermCfg(
      func=getup_mdp.lin_vel_xy,
      weight=2.0,
      params=phase3_params,
    ),
    "target_orientation": RewardTermCfg(
      func=getup_mdp.target_orientation,
      weight=2.0,
      params=phase3_params,
    ),
    "target_base_height": RewardTermCfg(
      func=getup_mdp.target_base_height,
      weight=5.0,
      params={
        "base_height_target": TARGET_BASE_HEIGHT,
        **phase3_params,
      },
    ),
    "target_joint_deviation_l2": RewardTermCfg(
      func=getup_mdp.target_joint_deviation_l2,
      weight=-0.1,
      params={
        "target_base_height_phase3": PHASE3_HEIGHT,
        "asset_cfg": joint_cfg,
      },
    ),
  }


def _configure_events(cfg: ManagerBasedRlEnvCfg, *, play: bool) -> None:
  # This teacher learns only from get-up reference frames.  There are no random
  # pushes, normal standing resets, or delayed-termination environments.
  for name in ("reset_base", "reset_robot_joints", "push_robot"):
    cfg.events.pop(name, None)

  cfg.events["init_getup_motion_loader"] = EventTermCfg(
    func=amp_mdp.init_motion_loader,
    mode="startup",
    params={
      "motion_dir": str(_GETUP_DATA_DIR),
      "recovery_dir": None,
      "delay_reset_env_ratio": 0.0,
      "max_delay_steps": 0,
    },
  )
  cfg.events["reset_from_getup_motion"] = EventTermCfg(
    func=amp_mdp.reset_from_motion_data,
    mode="reset",
    params={
      "motion_dir": str(_GETUP_DATA_DIR),
      "asset_cfg": _joint_asset(),
    },
  )
  cfg.events["apply_upward_assist"] = EventTermCfg(
    func=getup_mdp.apply_upward_assist,
    mode="reset",
    params={
      "command_name": "assist_force",
      "asset_cfg": SceneEntityCfg(
        "robot", body_names=(C.ANCHOR_BODY_NAME,)
      ),
    },
  )

  if play:
    # Deterministic evaluation: no assist force and no startup DR.
    for name in ("foot_friction", "encoder_bias", "base_com", "base_mass"):
      cfg.events.pop(name, None)
  else:
    cfg.events["base_mass"] = EventTermCfg(
      func=dr.body_mass,
      mode="startup",
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", body_names=(C.ANCHOR_BODY_NAME,)
        ),
        "ranges": (-1.0, 1.0),
        "operation": "add",
        "distribution": "uniform",
      },
    )


def casbot02_ampj_getup_teacher_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build the isolated 27-DoF CASBOT02 get-up teacher task."""
  cfg = make_velocity_env_cfg()
  _apply_robot(cfg)
  _apply_flat_terrain(cfg, play=False)
  _configure_commands(cfg, play=play)
  cfg.observations = _build_amp_observations(play=play)
  _configure_rewards(cfg)
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
  }
  _configure_events(cfg, play=play)

  if play:
    cfg.curriculum = {}
  else:
    cfg.curriculum = {
      "assist_force": CurriculumTermCfg(
        func=getup_mdp.assist_force_level,
        params={
          "command_name": "assist_force",
          "reward_term_name": "target_base_height",
          "success_fraction": 0.6,
          "force_step": ASSIST_FORCE_STEP,
          "minimum_force": MINIMUM_ASSIST_FORCE,
        },
      )
    }

  cfg.episode_length_s = 10.0
  return cfg


__all__ = [
  "ASSIST_FORCE_STEP",
  "INITIAL_ASSIST_FORCE",
  "MINIMUM_ASSIST_FORCE",
  "PHASE3_HEIGHT",
  "TARGET_BASE_HEIGHT",
  "casbot02_ampj_getup_teacher_flat_env_cfg",
]
