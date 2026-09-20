"""Casbot02 env configs: 12-leg loco teacher + student.

The loco teacher is a plain velocity-tracking PPO on the Casbot02 (12 leg
joints), whose observation is aligned EXACTLY with the AMP teacher's actor obs:

    [base_ang_vel(3), projected_gravity(3), command(3),
     joint_pos(12), joint_vel(12), actions(12)]  x 4-frame history  == 180 dims

so the student env can emit one 180-dim vector that both frozen teachers
consume (see ``rl/two_teacher_blend.py`` for the distillation loss).
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as env_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import (
  ObservationGroupCfg,
  ObservationTermCfg,
)
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
  ContactMatch,
  ContactSensorCfg,
  ObjRef,
  RingPatternCfg,
  TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp as velocity_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from wbc_mjlab import casbot02_constants as C
from wbc_mjlab import rewards as wbc_rewards
from wbc_mjlab.casbot02_actions import Casbot02LegWithArmSwingActionCfg
from wbc_mjlab.casbot02_commands import Casbot02VelocityCommandCfg


# Forward-only gait targets fitted conservatively from the CASBOT02 sole-site
# motion data.  Backward motion retains the fixed 09-02 targets and pure turn
# remains independent and unchanged below.
_TRANSLATION_SPEED_KNOTS = (0.15, 0.30, 0.50, 0.80, 1.00)
_TRANSLATION_STEP_LENGTH_KNOTS = (0.11, 0.19, 0.31, 0.51, 0.55)
_TRANSLATION_PEAK_HEIGHT_KNOTS = (0.090, 0.100, 0.114, 0.150, 0.170)
# Keep forward timing fixed over speed: speed changes are expressed through
# step length and peak height instead of a second, competing cadence curve.
_TRANSLATION_SWING_TIME_KNOTS = (0.585, 0.585, 0.585, 0.585, 0.585)
_TRANSLATION_AIR_TIME_MAX_KNOTS = (0.60, 0.60, 0.60, 0.60, 0.60)
_TRANSLATION_ACTIVATION_START = 0.10
_TRANSLATION_ACTIVATION_END = 0.15
_BACKWARD_PEAK_HEIGHT = 0.114
_BACKWARD_SWING_TIME = 0.59
_BACKWARD_LANDING_HEIGHT = 0.12
_BACKWARD_AIR_TIME_MAX = 0.60


def _leg_asset() -> SceneEntityCfg:
  """SceneEntityCfg selecting only the 12 policy-controlled leg joints."""
  return SceneEntityCfg(
    "robot", joint_names=C.CASBOT02_LEG_ONLY_JOINT_NAMES, preserve_order=True
  )


def _leg_actuator_asset() -> SceneEntityCfg:
  """SceneEntityCfg selecting only the 12 policy-controlled leg actuators."""
  return SceneEntityCfg(
    "robot", actuator_names=C.CASBOT02_LEG_ONLY_JOINT_NAMES, preserve_order=True
  )


def _torso_asset() -> SceneEntityCfg:
  return SceneEntityCfg("robot", body_names=("torso",))


def _feet_body_asset() -> SceneEntityCfg:
  return SceneEntityCfg(
    "robot", body_names=("leg_l6_link", "leg_r6_link"), preserve_order=True
  )


def _feet_site_asset() -> SceneEntityCfg:
  return SceneEntityCfg(
    "robot", site_names=("left_foot", "right_foot"), preserve_order=True
  )


def _knee_body_asset() -> SceneEntityCfg:
  """Select knee and hip-yaw bodies in the order required by the reward."""
  return SceneEntityCfg(
    "robot",
    body_names=("leg_l4_link", "leg_l3_link", "leg_r4_link", "leg_r3_link"),
    preserve_order=True,
  )


def _roll_yaw_actuator_asset() -> SceneEntityCfg:
  """Select the lateral/yaw leg actuators that produced internal torque."""
  return SceneEntityCfg(
    "robot",
    actuator_names=(
      "leg_l2_joint",  # hip roll
      "leg_l3_joint",  # hip yaw
      "leg_l6_joint",  # ankle roll
      "leg_r2_joint",
      "leg_r3_joint",
      "leg_r6_joint",
    ),
    preserve_order=True,
  )


def _roll_yaw_joint_asset() -> SceneEntityCfg:
  """Select the six measured non-sagittal leg joint positions."""
  return SceneEntityCfg(
    "robot",
    joint_names=(
      "leg_l2_joint",  # hip roll
      "leg_l3_joint",  # hip yaw
      "leg_l6_joint",  # ankle roll
      "leg_r2_joint",
      "leg_r3_joint",
      "leg_r6_joint",
    ),
    preserve_order=True,
  )


def _loco_actor_terms(enable_noise: bool) -> dict[str, ObservationTermCfg]:
  """45-dim actor obs, matching the AMP teacher's layout + noise."""
  return {
    "base_ang_vel": ObservationTermCfg(
      func=env_mdp.builtin_sensor,
      params={"sensor_name": "robot/angular-velocity"},
      noise=Unoise(n_min=-0.2, n_max=0.2) if enable_noise else None,
    ),
    "projected_gravity": ObservationTermCfg(
      func=env_mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05) if enable_noise else None,
    ),
    "command": ObservationTermCfg(
      func=env_mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    "joint_pos": ObservationTermCfg(
      func=env_mdp.joint_pos_rel,
      params={"asset_cfg": _leg_asset()},
      noise=Unoise(n_min=-0.01, n_max=0.01) if enable_noise else None,
    ),
    "joint_vel": ObservationTermCfg(
      func=env_mdp.joint_vel_rel,
      params={"asset_cfg": _leg_asset()},
      noise=Unoise(n_min=-0.5, n_max=0.5) if enable_noise else None,
    ),
    "actions": ObservationTermCfg(func=env_mdp.last_action),
  }


def _feet_contact_sensor() -> ContactSensorCfg:
  return ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree", pattern=r"^(leg_l6_link|leg_r6_link)$", entity="robot"
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )


def _self_collision_sensor() -> ContactSensorCfg:
  return ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="torso", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="torso", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )


def _apply_casbot02_robot(cfg: ManagerBasedRlEnvCfg) -> None:
  """Swap in the Casbot02 robot, add its sensors, wire action + viewer."""
  cfg.scene.entities = {"robot": C.get_casbot02_23dof_robot_cfg()}

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    _feet_contact_sensor(),
    _self_collision_sensor(),
  )

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  cfg.actions["joint_pos"] = Casbot02LegWithArmSwingActionCfg(
    entity_name="robot",
    actuator_names=C.CASBOT02_LEG_ONLY_JOINT_NAMES,
    scale=C.CASBOT02_LEG_ONLY_ACTION_SCALE,
    use_default_offset=True,
  )

  cfg.viewer.body_name = "torso"


def _apply_flat_terrain(cfg: ManagerBasedRlEnvCfg) -> None:
  """Flat terrain: plane, no terrain generator / height sensors."""
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # 只移除 terrain_scan(flat 不需要高度扫描), 保留 foot_height_scan(供抬脚奖励用)。
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  # foot_height_scan 的 frame 设成脚 site(在 casbot02_constants.get_spec 里加的)。
  for sensor in cfg.scene.sensors:
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=s, entity="robot")
        for s in ("left_foot", "right_foot")
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.03, num_samples=6)

  # The velocity task's ``terrain_levels`` curriculum asserts a terrain
  # generator (rough terrain only); drop it on the plane.
  cfg.curriculum.pop("terrain_levels", None)


def _apply_loco_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
  """Casbot02 velocity-tracking reward stack with G1 foot shaping."""

  # Use air time as swing phase.  Positive vx follows the new speed curve;
  # negative vx uses the fixed targets saved by the 09-02 baseline.  The
  # already-tuned pure-turn curve stays fixed.
  cfg.rewards.pop("foot_clearance", None)
  cfg.rewards["swing_height_curve"] = RewardTermCfg(
    func=wbc_rewards.command_conditioned_swing_height_curve,
    weight=-3.0,
    params={
      "sensor_name": "feet_ground_contact",
      "height_sensor_name": "foot_height_scan",
      "translation_speed_knots": _TRANSLATION_SPEED_KNOTS,
      "translation_peak_height_knots": _TRANSLATION_PEAK_HEIGHT_KNOTS,
      "translation_swing_time_knots": _TRANSLATION_SWING_TIME_KNOTS,
      "backward_peak_height": _BACKWARD_PEAK_HEIGHT,
      "backward_swing_time": _BACKWARD_SWING_TIME,
      # Frozen pure-turn targets from 原地左转/右转.npz.
      "turning_peak_height": (0.065, 0.065),
      "turning_swing_time": (0.40, 0.40),
      "command_name": "twist",
      "command_threshold": 0.2,
      "translation_activation_start": _TRANSLATION_ACTIVATION_START,
      "translation_activation_end": _TRANSLATION_ACTIVATION_END,
      "turning_linear_threshold": 0.2,
      "turning_angular_threshold": 0.2,
    },
  )

  # 落地峰值必须使用同一套运动模式门控，否则原始 0.12 m 目标仍会
  # 在背后把原地转弯脚拉高。
  cfg.rewards["foot_swing_height"] = RewardTermCfg(
    func=wbc_rewards.command_conditioned_feet_swing_height,
    weight=-0.25,
    params={
      "sensor_name": "feet_ground_contact",
      "height_sensor_name": "foot_height_scan",
      "translation_speed_knots": _TRANSLATION_SPEED_KNOTS,
      "translation_target_height_knots": _TRANSLATION_PEAK_HEIGHT_KNOTS,
      "backward_target_height": _BACKWARD_LANDING_HEIGHT,
      "turning_target_height": 0.07,
      "command_name": "twist",
      "command_threshold": 0.05,
      "translation_activation_start": _TRANSLATION_ACTIVATION_START,
      "translation_activation_end": _TRANSLATION_ACTIVATION_END,
      "turning_linear_threshold": 0.2,
      "turning_angular_threshold": 0.2,
    },
  )
  cfg.rewards["foot_slip"].weight = -2
  cfg.rewards["foot_slip"].params["asset_cfg"] = _feet_site_asset()
  cfg.rewards["soft_landing"].weight = -6e-3

  # Keep the dense air-time signal, but shorten its translation window with
  # speed.  The pure-turn upper bound remains exactly 0.42 s.
  cfg.rewards["air_time"] = RewardTermCfg(
    func=wbc_rewards.command_conditioned_feet_air_time,
    weight=1.0,
    params={
      "sensor_name": "feet_ground_contact",
      "threshold_min": 0.05,
      "translation_speed_knots": _TRANSLATION_SPEED_KNOTS,
      "translation_threshold_max_knots": _TRANSLATION_AIR_TIME_MAX_KNOTS,
      "backward_threshold_max": _BACKWARD_AIR_TIME_MAX,
      "turning_threshold_max": 0.42,
      "command_name": "twist",
      "command_threshold": 0.2,
      "translation_activation_start": _TRANSLATION_ACTIVATION_START,
      "translation_activation_end": _TRANSLATION_ACTIVATION_END,
      "turning_linear_threshold": 0.2,
      "turning_angular_threshold": 0.2,
    },
  )
  # This reward did not exist in the 09-02 baseline, so it is forward-only.
  cfg.rewards["feet_step_length"] = RewardTermCfg(
    func=wbc_rewards.command_conditioned_feet_step_length,
    weight=-0.5,
    params={
      "sensor_name": "feet_ground_contact",
      "asset_cfg": _feet_site_asset(),
      "translation_speed_knots": _TRANSLATION_SPEED_KNOTS,
      "translation_step_length_knots": _TRANSLATION_STEP_LENGTH_KNOTS,
      "command_name": "twist",
      "translation_activation_start": _TRANSLATION_ACTIVATION_START,
      "translation_activation_end": _TRANSLATION_ACTIVATION_END,
      "turning_linear_threshold": 0.2,
      "turning_angular_threshold": 0.2,
    },
  )

  # Per-robot wiring: root/torso bodies + leg joints.
  cfg.rewards["track_linear_velocity"].params["asset_cfg"] = _torso_asset()
  cfg.rewards["track_linear_velocity"].params["std"] = 0.5
  cfg.rewards["track_angular_velocity"].params["asset_cfg"] = _torso_asset()
  cfg.rewards["track_angular_velocity"].params["std"] = 0.7071
  cfg.rewards["upright"].params["asset_cfg"] = _torso_asset()
  cfg.rewards["pose"].weight = 1.0
  cfg.rewards["pose"].params["asset_cfg"] = _leg_asset()
  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r"leg_[lr]1_joint": 0.3,   # hip pitch
    r"leg_[lr]2_joint": 0.15,  # hip roll
    r"leg_[lr]3_joint": 0.15,  # hip yaw
    r"leg_[lr]4_joint": 0.35,  # knee
    r"leg_[lr]5_joint": 0.25,  # ankle pitch
    r"leg_[lr]6_joint": 0.1,   # ankle roll
  }
  cfg.rewards["pose"].params["std_running"] = {
    r"leg_[lr]1_joint": 0.5,   # hip pitch
    r"leg_[lr]2_joint": 0.2,   # hip roll
    r"leg_[lr]3_joint": 0.2,   # hip yaw
    r"leg_[lr]4_joint": 0.6,   # knee
    r"leg_[lr]5_joint": 0.35,  # ankle pitch
    r"leg_[lr]6_joint": 0.15,  # ankle roll
  }
  # Match G1's dedicated zero-command posture penalty, but constrain only the
  # 12 policy-controlled leg joints (the arms follow the deterministic swing).
  cfg.rewards["stand_pose"] = RewardTermCfg(
    func=wbc_rewards.stand_pose,
    weight=-3.0,
    params={
      "command_name": "twist",
      "asset_cfg": _leg_asset(),
    },
  )
  # cfg.rewards["standing_foot_distance"] = RewardTermCfg(
  #   func=wbc_rewards.standing_foot_distance,
  #   weight=-5.0,
  #   params={
  #     "command_name": "twist",
  #     "command_threshold": 0.2,
  #     "target_lateral_distance": 0.285,
  #     "target_fore_distance": 0.0,
  #     "asset_cfg": _feet_site_asset(),
  #   },
  # )
  # Casbot02 nominal sole-center spacing is 0.285 m.  Allow 0.027 m inward
  # motion while leaving extra outward room for turning.  Positive weight is
  # intentional: the reward function returns a negative error outside this band.
  cfg.rewards["feet_distance_lateral"] = RewardTermCfg(
    func=wbc_rewards.feet_distance_lateral,
    weight=3,
    params={
      "asset_cfg": _feet_site_asset(),
      "min_distance": 0.266,
      "max_distance": 0.40,
    },
  )
  cfg.rewards["knee_distance_lateral"] = RewardTermCfg(
    func=wbc_rewards.knee_distance_lateral,
    weight=3,
    params={
      "asset_cfg": _knee_body_asset(),
      "min_distance": 0.279,
      "max_distance": 0.32,
    },
  )
  cfg.rewards["flat_foot"] = RewardTermCfg(
    func=wbc_rewards.flat_foot,
    weight=-1,
    params={
      "sensor_name": "feet_ground_contact",
      "asset_cfg": _feet_body_asset(),
    },
  )
  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["body_ang_vel"].params["asset_cfg"] = _torso_asset()
  cfg.rewards["angular_momentum"].weight = -0.02
  # Added after the 2026-09-02 reference run; keep disabled while reproducing
  # that policy's reward configuration.
  # cfg.rewards["straight_hip_yaw_pos_l2"] = RewardTermCfg(
  #   func=wbc_rewards.straight_hip_yaw_pos_l2,
  #   weight=-5.0,
  #   params={
  #     "command_name": "twist",
  #     "asset_cfg": _roll_yaw_joint_asset(),
  #     "joint_weights": (0.5, 1.0, 0.5, 0.5, 1.0, 0.5),
  #     "linear_command_threshold": 0.2,
  #     "yaw_relax_start": 0.15,
  #     "yaw_relax_end": 0.4,
  #   },
  # )
  # Penalize virtual PD-equilibrium offsets during straight translation.  Hip
  # yaw is weighted most strongly; hip/ankle roll retain authority for lateral
  # load transfer.  The term fades out between 0.15 and 0.40 rad/s yaw command.
  # cfg.rewards["straight_non_sagittal_target_deviation_l2"] = RewardTermCfg(
  #   func=wbc_rewards.straight_non_sagittal_target_deviation_l2,
  #   weight=-0.5,
  #   params={
  #     "command_name": "twist",
  #     "action_term_name": "joint_pos",
  #     "joint_names": (
  #       "leg_l2_joint",
  #       "leg_l3_joint",
  #       "leg_l6_joint",
  #       "leg_r2_joint",
  #       "leg_r3_joint",
  #       "leg_r6_joint",
  #     ),
  #     "joint_weights": (0.5, 1.0, 0.5, 0.5, 1.0, 0.5),
  #     "linear_command_threshold": 0.2,
  #     "yaw_relax_start": 0.15,
  #     "yaw_relax_end": 0.4,
  #   },
  # )
  # cfg.rewards["leg_torques_l2"] = RewardTermCfg(
  #   func=env_mdp.joint_torques_l2,
  #   weight=-5.0e-6,
  #   params={"asset_cfg": _leg_actuator_asset()},
  # )
  # Suppress persistent hip-roll closed-chain internal torques without
  # penalizing the other ten policy-controlled leg actuators.
  # cfg.rewards["hip_roll_torques_l2"] = RewardTermCfg(
  #   func=env_mdp.joint_torques_l2,
  #   weight=-5.0e-6,
  #   params={
  #     "asset_cfg": SceneEntityCfg(
  #       "robot",
  #       actuator_names=("leg_l2_joint", "leg_r2_joint"),
  #       preserve_order=True,
  #     )
  #   },
  # )
  # Penalize only the part of actuator effort above 80% of its physical limit.
  # At the measured 0.5 m/s gait this primarily targets ankle-pitch saturation.
  # cfg.rewards["dof_torque_limits"] = RewardTermCfg(
  #   func=wbc_rewards.dof_torque_limits,
  #   weight=-1,
  #   params={
  #     "asset_cfg": SceneEntityCfg("robot", actuator_names=(".*",)),
  #     "soft_torque_limit": 0.8,
  #   },
  # )
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=velocity_mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": "self_collision", "force_threshold": 10.0},
  )


def _apply_casbot02_dr(cfg: ManagerBasedRlEnvCfg) -> None:
  """Casbot02 域随机化, 对齐 amp_mjlab(补足 make_velocity_env_cfg 里空转的项)."""
  # 1. foot_friction: only randomize the two foot collision geoms.
  cfg.events["foot_friction"].params["asset_cfg"] = SceneEntityCfg(
    "robot", geom_names=C.CASBOT02_FOOT_GEOM_NAMES, preserve_order=True
  )

  # 2. base_mass: 躯干质量 add ±4.0 kg。
  cfg.events["base_mass"] = EventTermCfg(
    mode="startup",
    func=dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso",)),
      "ranges": (-4.0, 4.0),
      "operation": "add",
      "distribution": "uniform",
    },
  )

  # 3. actuator_gains: 腿部 actuator 组(LEG_HEAVY + LEG_LIGHT)的 kp/kd scale。
  cfg.events["actuator_gains"] = EventTermCfg(
    mode="startup",
    func=dr.pd_gains,
    params={
      "asset_cfg": SceneEntityCfg("robot", actuator_ids=[0, 1]),
      "kp_range": (0.85, 1.15),
      "kd_range": (0.85, 1.15),
      "operation": "scale",
      "distribution": "uniform",
    },
  )

  # 4. joint_friction: 腿部关节摩擦 scale 0.5~1.5。
  cfg.events["joint_friction"] = EventTermCfg(
    mode="startup",
    func=dr.joint_friction,
    params={
      "asset_cfg": _leg_asset(),
      "ranges": (0.5, 1.5),
      "operation": "scale",
      "distribution": "uniform",
    },
  )

  # 5. base_com: 躯干质心偏移。
  cfg.events["base_com"].params["asset_cfg"] = SceneEntityCfg(
    "robot", body_names=("torso",)
  )

  # 6. joint_default_pos: 零位偏移 ±0.02 rad。
  cfg.events["joint_default_pos"] = EventTermCfg(
    mode="startup",
    func=dr.joint_default_pos,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      "ranges": (-0.02, 0.02),
      "distribution": "uniform",
      "operation": "add",
    },
  )

  # 7. non_base_mass: 非躯干 body 质量 scale 0.9~1.1。
  cfg.events["non_base_mass"] = EventTermCfg(
    mode="startup",
    func=dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=(r"^(?!torso$).+$",)),
      "ranges": (0.80, 1.20),
      "operation": "scale",
      "distribution": "uniform",
      "shared_random": False,
    },
  )

  # 8. push_robot: match RoboParty AMP's lower-frequency planar velocity
  # disturbance.  Keep vertical/roll/pitch kicks disabled so most rollout
  # time is spent learning a clean nominal gait.
  # push_robot = cfg.events["push_robot"]
  # push_robot.interval_range_s = (5.0, 10.0)
  # push_robot.params["velocity_range"] = {
  #   "x": (-0.5, 0.5),
  #   "y": (-0.5, 0.5),
  #   "yaw": (-1.0, 1.0),
  # }


def _apply_twist_ranges(cfg: ManagerBasedRlEnvCfg) -> None:
  """G1-aligned vx/wz curriculum plus a dedicated pure-turn cohort."""
  base_twist_cmd = cfg.commands["twist"]
  assert isinstance(base_twist_cmd, UniformVelocityCommandCfg)
  vx, vy, wz = (-1.0, 1.0), (0.0, 0.0), (-1.0, 1.0)
  twist_cmd = Casbot02VelocityCommandCfg(
    resampling_time_range=base_twist_cmd.resampling_time_range,
    debug_vis=base_twist_cmd.debug_vis,
    entity_name=base_twist_cmd.entity_name,
    heading_command=base_twist_cmd.heading_command,
    heading_control_stiffness=base_twist_cmd.heading_control_stiffness,
    rel_standing_envs=base_twist_cmd.rel_standing_envs,
    rel_turning_envs=0.2,
    rel_startup_envs=0.1,
    startup_standing_time_range=(2.0, 3.0),
    startup_walking_time_range=(3.0, 4.0),
    startup_speed_range=(0.3, 0.6),
    rel_heading_envs=0.2,
    rel_world_envs=base_twist_cmd.rel_world_envs,
    rel_forward_envs=base_twist_cmd.rel_forward_envs,
    init_velocity_prob=base_twist_cmd.init_velocity_prob,
    min_turning_ang_vel=0.2,
    ranges=Casbot02VelocityCommandCfg.Ranges(
      lin_vel_x=vx,
      lin_vel_y=vy,
      ang_vel_z=wz,
      heading=base_twist_cmd.ranges.heading,
    ),
    viz=base_twist_cmd.viz,
  )
  cfg.commands["twist"] = twist_cmd

  # Expand only the translation range in four stages.  The existing wz
  # schedule (half range until 5000 iterations, then full range) is unchanged.
  cfg.curriculum["command_vel"].params["velocity_stages"] = [
    {
      "step": 0,
      "lin_vel_x": (-0.30, 0.30),
      "lin_vel_y": vy,
      "ang_vel_z": (wz[0] * 0.5, wz[1] * 0.5),
    },
    {
      "step": 2500 * 24,
      "lin_vel_x": (-0.50, 0.50),
      "lin_vel_y": vy,
      "ang_vel_z": (wz[0] * 0.5, wz[1] * 0.5),
    },
    {
      "step": 5000 * 24,
      "lin_vel_x": (-0.80, 0.80),
      "lin_vel_y": vy,
      "ang_vel_z": wz,
    },
    {
      "step": 8000 * 24,
      "lin_vel_x": vx,
      "lin_vel_y": vy,
      "ang_vel_z": wz,
    },
  ]


def _apply_leg_obs(cfg: ManagerBasedRlEnvCfg, *, play: bool) -> None:
  """Override actor/critic obs to 12-leg layout (history 4)."""
  enable_noise = not play
  cfg.observations["actor"] = ObservationGroupCfg(
    terms=_loco_actor_terms(enable_noise=enable_noise),
    concatenate_terms=True,
    enable_corruption=enable_noise,
    history_length=4,
  )
  # Critic = actor terms + privileged base_lin_vel (deployment-invisible).
  critic_terms = _loco_actor_terms(enable_noise=False)
  critic_terms["base_lin_vel"] = ObservationTermCfg(
    func=env_mdp.builtin_sensor,
    params={"sensor_name": "robot/linear-velocity"},
  )
  cfg.observations["critic"] = ObservationGroupCfg(
    terms=critic_terms,
    concatenate_terms=True,
    enable_corruption=False,
    history_length=4,
  )


def casbot02_loco_teacher_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """12-leg Casbot02 velocity-tracking loco teacher env (obs 180, action 12)."""
  cfg = make_velocity_env_cfg()

  _apply_casbot02_robot(cfg)
  _apply_flat_terrain(cfg)
  _apply_loco_rewards(cfg)
  _apply_twist_ranges(cfg)
  _apply_leg_obs(cfg, play=play)

  if play:
    # 禁用 DR + 扰动, 保证播放确定性(和 HANDOFF 的 _disable_play_randomization 一致)。
    for name in (
      "foot_friction", "encoder_bias", "base_com", "base_mass",
      "actuator_gains", "joint_friction", "joint_default_pos",
      "non_base_mass", "push_robot",
    ):
      cfg.events.pop(name, None)
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
  else:
    _apply_casbot02_dr(cfg)

  return cfg


def casbot02_student_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Student env = loco teacher env + a ``command`` obs group for the gate.

  The distillation runner reads ``obs["command"]`` (the current [B,3] velocity
  command) each step to compute the forward/turn blend gate (see
  ``rl/casbot02_distill.py``).
  """
  cfg = casbot02_loco_teacher_env_cfg(play=play)
  cfg.observations["command"] = ObservationGroupCfg(
    terms={
      "command": ObservationTermCfg(
        func=env_mdp.generated_commands,
        params={"command_name": "twist"},
      )
    },
    concatenate_terms=True,
    enable_corruption=False,
  )
  return cfg


__all__ = ["casbot02_loco_teacher_env_cfg", "casbot02_student_env_cfg"]
