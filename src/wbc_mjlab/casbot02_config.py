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


def _leg_asset() -> SceneEntityCfg:
  """SceneEntityCfg selecting only the 12 policy-controlled leg joints."""
  return SceneEntityCfg(
    "robot", joint_names=C.CASBOT02_LEG_ONLY_JOINT_NAMES, preserve_order=True
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

  # 用每只脚的当前腾空时间作为摆动进度，跟踪半正弦高度曲线。
  # AMP 前进数据：左/右峰值 11.6/11.4 cm，摆动时间 0.55/0.59 s。
  cfg.rewards.pop("foot_clearance", None)
  cfg.rewards["swing_height_curve"] = RewardTermCfg(
    func=wbc_rewards.swing_height_curve,
    weight=-3.0,
    params={
      "sensor_name": "feet_ground_contact",
      "height_sensor_name": "foot_height_scan",
      "peak_height": (0.114, 0.114),
      "swing_time": (0.59, 0.59),
      "command_name": "twist",
      "command_threshold": 0.2,
    },
  )

  # 落地时检查完整摆动周期的峰值高度，与上面的密集不足惩罚互补。
  cfg.rewards["foot_swing_height"].weight = -0.25
  cfg.rewards["foot_swing_height"].params["target_height"] = 0.12
  cfg.rewards["foot_slip"].weight = -0.1
  cfg.rewards["foot_slip"].params["asset_cfg"] = _feet_site_asset()
  cfg.rewards["soft_landing"].weight = -5e-3

  # 保留密集 air_time 信号：从离地 0.05 s 起给分，到 AMP 的约
  # 0.60 s 摆动时间停止给分；高度曲线同时在目标时刻回到零。
  cfg.rewards["air_time"].weight = 1.0
  cfg.rewards["air_time"].params["threshold_min"] = 0.05
  cfg.rewards["air_time"].params["threshold_max"] = 0.60
  cfg.rewards["air_time"].params["command_threshold"] = 0.2

  # Per-robot wiring: root/torso bodies + leg joints.
  cfg.rewards["track_linear_velocity"].params["asset_cfg"] = _torso_asset()
  cfg.rewards["track_linear_velocity"].params["std"] = 0.5
  cfg.rewards["track_angular_velocity"].params["asset_cfg"] = _torso_asset()
  cfg.rewards["track_angular_velocity"].params["std"] = 0.7071
  cfg.rewards["upright"].params["asset_cfg"] = _torso_asset()
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
    weight=-10.0,
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
    weight=2.0,
    params={
      "asset_cfg": _feet_site_asset(),
      "min_distance": 0.275,
      "max_distance": 0.35,
    },
  )
  cfg.rewards["knee_distance_lateral"] = RewardTermCfg(
    func=wbc_rewards.knee_distance_lateral,
    weight=3.0,
    params={
      "asset_cfg": _knee_body_asset(),
      "min_distance": 0.275,
      "max_distance": 0.35,
    },
  )
  cfg.rewards["flat_foot"] = RewardTermCfg(
    func=wbc_rewards.flat_foot,
    weight=-0.5,
    params={
      "sensor_name": "feet_ground_contact",
      "asset_cfg": _feet_body_asset(),
    },
  )
  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["body_ang_vel"].params["asset_cfg"] = _torso_asset()
  cfg.rewards["angular_momentum"].weight = -0.02
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

  # 2. base_mass: 躯干质量 add ±1.0 kg。
  cfg.events["base_mass"] = EventTermCfg(
    mode="startup",
    func=dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso",)),
      "ranges": (-1.0, 1.0),
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

  # 5. base_com: 躯干质心偏移(原配置 body_names 为空, 是 no-op)。
  cfg.events["base_com"].params["asset_cfg"] = SceneEntityCfg(
    "robot", body_names=("torso",)
  )

  # 6. joint_default_pos: 零位偏移 ±0.01 rad。
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
      "ranges": (0.85, 1.15),
      "operation": "scale",
      "distribution": "uniform",
      "shared_random": False,
    },
  )


def _apply_twist_ranges(cfg: ManagerBasedRlEnvCfg) -> None:
  """G1-aligned vx/wz curriculum with lateral velocity fixed at zero."""
  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  vx, vy, wz = (-1.0, 1.0), (0.0, 0.0), (-1.0, 1.0)
  twist_cmd.ranges.lin_vel_x = vx
  twist_cmd.ranges.lin_vel_y = vy
  twist_cmd.ranges.ang_vel_z = wz

  # Mirror G1's half-speed warmup, while keeping vy disabled in both stages.
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
