"""Independent CASBOT02 27-DoF AMP recovery-teacher environment."""

from __future__ import annotations

import math
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from wbc_mjlab import amp_mdp
from wbc_mjlab.casbot02ampj import constants as C


_AMP_DATA_DIR = (
  Path(__file__).resolve().parents[1]
  / "data"
  / "motions"
  / "casbot02ampj"
  / "amp"
)
_WALK_RUN_DIR = _AMP_DATA_DIR / "WalkandRun"
_RECOVERY_DIR = _AMP_DATA_DIR / "Recovery"


def _joint_asset() -> SceneEntityCfg:
  return SceneEntityCfg(
    "robot", joint_names=C.JOINT_NAMES, preserve_order=True
  )


def _foot_site_asset() -> SceneEntityCfg:
  return SceneEntityCfg(
    "robot", site_names=C.FEET_SITE_NAMES, preserve_order=True
  )


def _build_amp_observations(play: bool) -> dict[str, ObservationGroupCfg]:
  noise = not play
  joint_cfg = _joint_asset()
  actor_terms: dict[str, ObservationTermCfg] = {
    "base_ang_vel": ObservationTermCfg(
      func=envs_mdp.builtin_sensor,
      params={"sensor_name": "robot/angular-velocity"},
      noise=Unoise(n_min=-0.2, n_max=0.2) if noise else None,
    ),
    "projected_gravity": ObservationTermCfg(
      func=envs_mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05) if noise else None,
    ),
    "command": ObservationTermCfg(
      func=envs_mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      params={"asset_cfg": joint_cfg},
      noise=Unoise(n_min=-0.01, n_max=0.01) if noise else None,
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      params={"asset_cfg": joint_cfg},
      noise=Unoise(n_min=-0.5, n_max=0.5) if noise else None,
    ),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
  }

  anchor_cfg = SceneEntityCfg(
    "robot", body_names=(C.ANCHOR_BODY_NAME,)
  )
  body_cfg = SceneEntityCfg(
    "robot", body_names=C.AMP_BODY_NAMES, preserve_order=True
  )
  critic_terms: dict[str, ObservationTermCfg] = {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(
      func=envs_mdp.builtin_sensor,
      params={"sensor_name": "robot/linear-velocity"},
    ),
    "body_pos_b": ObservationTermCfg(
      func=amp_mdp.robot_body_pos_b,
      params={"anchor_cfg": anchor_cfg, "body_cfg": body_cfg},
    ),
    "body_ori_b": ObservationTermCfg(
      func=amp_mdp.robot_body_ori_b,
      params={"anchor_cfg": anchor_cfg, "body_cfg": body_cfg},
    ),
  }
  amp_terms: dict[str, ObservationTermCfg] = {
    "body_pos_b": ObservationTermCfg(
      func=amp_mdp.robot_body_pos_b,
      params={"anchor_cfg": anchor_cfg, "body_cfg": body_cfg},
    ),
    "body_ori_b": ObservationTermCfg(
      func=amp_mdp.robot_body_ori_b,
      params={"anchor_cfg": anchor_cfg, "body_cfg": body_cfg},
    ),
    "body_lin_vel_b": ObservationTermCfg(
      func=amp_mdp.robot_body_lin_vel_b,
      params={"anchor_cfg": anchor_cfg, "body_cfg": body_cfg},
    ),
    "body_ang_vel_b": ObservationTermCfg(
      func=amp_mdp.robot_body_ang_vel_b,
      params={"anchor_cfg": anchor_cfg, "body_cfg": body_cfg},
    ),
  }

  return {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=noise,
      history_length=4,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
      history_length=4,
    ),
    "amp": ObservationGroupCfg(
      terms=amp_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }


def _apply_robot(cfg: ManagerBasedRlEnvCfg) -> None:
  cfg.scene.entities = {"robot": C.get_robot_cfg()}

  # Recovery has many whole-body contacts; leave ample match capacity even
  # though the unused dexterous-hand subtrees have been removed.
  cfg.sim.njmax = 1500
  cfg.sim.mujoco.ccd_iterations = 128
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = None

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=name, entity="robot")
        for name in C.FEET_SITE_NAMES
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  feet_ground = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_leg_ankle_roll_link|right_leg_ankle_roll_link)$",
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern=C.ROOT_BODY_NAME, entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern=C.ROOT_BODY_NAME, entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (feet_ground, self_collision)

  cfg.actions["joint_pos"] = JointPositionActionCfg(
    entity_name="robot",
    actuator_names=C.JOINT_NAMES,
    scale=C.ACTION_SCALE,
    use_default_offset=True,
    preserve_order=True,
  )
  cfg.viewer.body_name = C.ANCHOR_BODY_NAME

  cfg.events["foot_friction"].params["asset_cfg"] = SceneEntityCfg(
    "robot", geom_names=C.FOOT_GEOM_NAMES, preserve_order=True
  )
  cfg.events["base_com"].params["asset_cfg"] = SceneEntityCfg(
    "robot", body_names=(C.ANCHOR_BODY_NAME,)
  )


def _apply_flat_terrain(cfg: ManagerBasedRlEnvCfg, *, play: bool) -> None:
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.scene.sensors = tuple(
    sensor for sensor in (cfg.scene.sensors or ())
    if sensor.name != "terrain_scan"
  )
  cfg.terminations.pop("out_of_terrain_bounds", None)
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}


def _apply_twist_ranges(cfg: ManagerBasedRlEnvCfg) -> None:
  """Match the loco teacher's vx/wz curriculum and keep vy disabled."""
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  vx, vy, wz = (-1.0, 1.0), (0.0, 0.0), (-1.0, 1.0)
  twist_cmd.ranges.lin_vel_x = vx
  twist_cmd.ranges.lin_vel_y = vy
  twist_cmd.ranges.ang_vel_z = wz

  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {
      "step": 0,
      "lin_vel_x": (vx[0] * 0.5, vx[1] * 0.5),
      "lin_vel_y": vy,
      "ang_vel_z": (wz[0] * 0.5, wz[1] * 0.5),
    },
    {
      "step": 5000 * 24,
      "lin_vel_x": vx,
      "lin_vel_y": vy,
      "ang_vel_z": wz,
    },
  ]


def _replace_with_amp_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
  foot_slip = cfg.rewards.get("foot_slip")
  if foot_slip is None:
    raise ValueError("Base velocity config no longer provides foot_slip")
  foot_slip.weight = -0.25
  foot_slip.params["asset_cfg"] = _foot_site_asset()

  cfg.rewards = {
    "track_anchor_linear_velocity": RewardTermCfg(
      func=amp_mdp.track_anchor_linear_velocity,
      weight=1.0,
      params={
        "command_name": "twist",
        "std": 1.0,
        "mask_delay": True,
        "delay_env_rew_ratio": 0.0,
        "anchor_cfg": SceneEntityCfg(
          "robot", body_names=(C.ANCHOR_BODY_NAME,)
        ),
      },
    ),
    "track_anchor_angular_velocity": RewardTermCfg(
      func=amp_mdp.track_anchor_angular_velocity,
      weight=1.0,
      params={
        "command_name": "twist",
        "std": 3.14,
        "mask_delay": True,
        "delay_env_rew_ratio": 0.0,
        "anchor_cfg": SceneEntityCfg(
          "robot", body_names=(C.ANCHOR_BODY_NAME,)
        ),
      },
    ),
    "track_root_height": RewardTermCfg(
      func=amp_mdp.track_root_height,
      weight=1.0,
      params={"std": 0.3, "mask_delay": True, "delay_env_rew_ratio": 3.5},
    ),
    "body_ang_vel_xy_l2": RewardTermCfg(
      func=amp_mdp.body_ang_vel_xy_l2,
      weight=0.5,
      params={
        "std": 3.14,
        "mask_delay": True,
        "delay_env_rew_ratio": 0.0,
        "body_cfg": SceneEntityCfg(
          "robot", body_names=(C.ROOT_BODY_NAME,)
        ),
      },
    ),
    "is_terminated": RewardTermCfg(
      func=envs_mdp.is_terminated,
      weight=-200.0,
    ),
    "joint_acc_l2": RewardTermCfg(func=envs_mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_pos_limits": RewardTermCfg(
      func=envs_mdp.joint_pos_limits,
      weight=-10.0,
    ),
    "action_rate_l2": RewardTermCfg(
      func=envs_mdp.action_rate_l2,
      weight=-0.01,
    ),
    "foot_slip": foot_slip,
    "self_collisions": RewardTermCfg(
      func=amp_mdp.self_collision_cost,
      weight=-0.1,
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
  }


def _replace_with_amp_terminations(cfg: ManagerBasedRlEnvCfg) -> None:
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "bad_orientation": TerminationTermCfg(
      func=envs_mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
    "bad_base_height": TerminationTermCfg(
      func=envs_mdp.root_height_below_minimum,
      params={"minimum_height": 0.6},
    ),
  }


def _add_amp_motion_events(cfg: ManagerBasedRlEnvCfg) -> None:
  cfg.events["init_motion_loader"] = EventTermCfg(
    func=amp_mdp.init_motion_loader,
    mode="startup",
    params={
      "motion_dir": str(_WALK_RUN_DIR),
      "recovery_dir": str(_RECOVERY_DIR),
      "delay_reset_env_ratio": 0.4,
      "max_delay_steps": 250,
    },
  )
  cfg.events["reset_from_motion"] = EventTermCfg(
    func=amp_mdp.reset_from_motion_data,
    mode="reset",
    params={
      "motion_dir": str(_WALK_RUN_DIR),
      "asset_cfg": _joint_asset(),
    },
  )
  for name in ("reset_base", "reset_joints", "reset_robot_joints"):
    cfg.events.pop(name, None)


def casbot02_ampj_teacher_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Build the independent 27-DoF CASBOT02 AMP recovery-teacher task."""
  cfg = make_velocity_env_cfg()
  _apply_robot(cfg)
  _apply_twist_ranges(cfg)
  _apply_flat_terrain(cfg, play=play)
  cfg.observations = _build_amp_observations(play=play)
  _replace_with_amp_rewards(cfg)
  _replace_with_amp_terminations(cfg)
  _add_amp_motion_events(cfg)
  cfg.metrics = {
    **(cfg.metrics or {}),
    "mean_delay_steps": MetricsTermCfg(func=amp_mdp.mean_delay_steps),
  }

  if play:
    cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 1.0
  cfg.episode_length_s = int(1e9) if play else 20.0
  return cfg


__all__ = [
  "casbot02_ampj_teacher_flat_env_cfg",
]
