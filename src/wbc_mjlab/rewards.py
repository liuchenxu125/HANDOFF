"""Reward helpers for HANDOFF tracking and locomotion-teacher tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.envs import mdp as env_mdp
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.sensor.terrain_height_sensor import TerrainHeightSensor
from mjlab.utils.lab_api.math import (
  euler_xyz_from_quat,
  quat_apply_inverse,
  quat_error_magnitude,
  wrap_to_pi,
  yaw_quat,
)

from wbc_mjlab.observations import (
  FEET_BODY_NAMES,
  HAND_BODY_NAMES,
  KEY_BODY_NAMES,
  LOCO_STANCE_RATIO,
  get_motion_command,
  loco_leg_phase,
  motion_hand_pos_b_tensor,
  motion_loco_command,
  motion_loco_leg_phase,
  robot_hand_pos_b_tensor,
  tracked_body_indices,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
_ANKLE_JOINT_NAMES = (
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
)
_HIP_PITCH_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "right_hip_pitch_joint",
)
_HIP_ROLL_JOINT_NAMES = (
  "left_hip_roll_joint",
  "right_hip_roll_joint",
)
_G = 9.81  # m/s²


def _get_robot(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> Entity:
  return env.scene[asset_cfg.name]


# ---------------------------------------------------------------------------
# Stability metric helpers (whole-body CoM, capture point, momentum).
# Ported from handoff_mjlab_stable; original adapted from IHMC IsaacLab.
# ---------------------------------------------------------------------------


def _body_masses(asset: Entity) -> torch.Tensor:
  """Per-entity-body masses. Shape [num_envs, n_bodies]."""
  return asset.data.model.body_mass[:, asset.data.indexing.body_ids]


def _whole_body_com_pos(asset: Entity) -> torch.Tensor:
  """Whole-body CoM position in world frame. Shape [num_envs, 3]."""
  masses = _body_masses(asset)
  total_mass = masses.sum(dim=1, keepdim=True).clamp_min(1e-8)
  return (masses.unsqueeze(-1) * asset.data.body_com_pos_w).sum(dim=1) / total_mass


def _whole_body_com_vel(asset: Entity) -> torch.Tensor:
  """Whole-body CoM velocity in world frame. Shape [num_envs, 3]."""
  masses = _body_masses(asset)
  total_mass = masses.sum(dim=1, keepdim=True).clamp_min(1e-8)
  return (masses.unsqueeze(-1) * asset.data.body_com_lin_vel_w).sum(dim=1) / total_mass


def _capture_point(asset: Entity) -> torch.Tensor:
  """LIPM capture point in world XY. Shape [num_envs, 2].

  CP = CoM_xy + CoM_vel_xy / sqrt(g / h) where h = CoM height.
  Valid for flat ground; use a ray-caster for uneven terrain.
  """
  com_pos = _whole_body_com_pos(asset)
  com_vel = _whole_body_com_vel(asset)
  h = com_pos[:, 2].clamp(min=1e-3)
  omega = torch.sqrt(torch.tensor(_G, device=h.device, dtype=h.dtype) / h)
  return com_pos[:, :2] + com_vel[:, :2] / omega.unsqueeze(-1)


def _linear_momentum(asset: Entity) -> torch.Tensor:
  """Total linear momentum p = Σ m_i * v_i. Shape [num_envs, 3]."""
  masses = _body_masses(asset)
  return (masses.unsqueeze(-1) * asset.data.body_com_lin_vel_w).sum(dim=1)


def _angular_momentum(asset: Entity) -> torch.Tensor:
  """Orbital angular momentum L = Σ (r_i − r_CoM) × (m_i v_i).

  Spin term (I_i ω_i) is omitted for GPU performance (~10–20 % error vs full
  CAM). Shape [num_envs, 3].
  """
  masses = _body_masses(asset)
  total_mass = masses.sum(dim=1, keepdim=True).clamp_min(1e-8)
  body_pos = asset.data.body_com_pos_w
  body_vel = asset.data.body_com_lin_vel_w
  com_pos = (masses.unsqueeze(-1) * body_pos).sum(dim=1) / total_mass
  rel_pos = body_pos - com_pos.unsqueeze(1)
  return torch.linalg.cross(
    rel_pos, masses.unsqueeze(-1) * body_vel, dim=-1
  ).sum(dim=1)


def _support_polygon_dist(
  query_xy: torch.Tensor,        # [N, 2]
  contact_pos_xy: torch.Tensor,  # [N, M, 2]
  contact_mask: torch.Tensor,    # [N, M] bool
  tolerance: float,
) -> torch.Tensor:
  """Distance from query_xy to tolerance-expanded bounding box of active contacts.

  Returns 0 when query_xy is inside the bounding box + tolerance margin.
  """
  _large = 1e6
  x = contact_pos_xy[..., 0]
  x_min = torch.where(contact_mask, x, torch.full_like(x,  _large)).min(dim=1).values
  x_max = torch.where(contact_mask, x, torch.full_like(x, -_large)).max(dim=1).values
  y = contact_pos_xy[..., 1]
  y_min = torch.where(contact_mask, y, torch.full_like(y,  _large)).min(dim=1).values
  y_max = torch.where(contact_mask, y, torch.full_like(y, -_large)).max(dim=1).values
  dx = torch.clamp_min(
    torch.maximum(x_min - query_xy[:, 0] - tolerance, query_xy[:, 0] - x_max - tolerance), 0.0
  )
  dy = torch.clamp_min(
    torch.maximum(y_min - query_xy[:, 1] - tolerance, query_xy[:, 1] - y_max - tolerance), 0.0
  )
  return torch.sqrt(dx**2 + dy**2)


def _cop_penalty(sat: torch.Tensor, alpha: float) -> torch.Tensor:
  """Reciprocal CoP-style penalty: peaks at saturation=1 (friction cone limit)."""
  return alpha / ((sat - 1.0) ** 2 + alpha)


# ---------------------------------------------------------------------------
# Stability metric rewards.
# ---------------------------------------------------------------------------


def simplified_com_in_support_polygon(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  feet_sensor_cfg: SceneEntityCfg = SceneEntityCfg("feet_ground_contact"),
  hands_sensor_cfg: SceneEntityCfg | None = None,
  tolerance: float = 0.05,
  std: float = 0.1,
) -> torch.Tensor:
  """Reward whole-body CoM staying inside the convex hull of active contacts.

  Support polygon is approximated by the axis-aligned bounding box of active
  contact points (fast, GPU-friendly, negligible error for 2-foot stance).
  ``hands_sensor_cfg`` extends the polygon during hand-on-ground contact
  (fall recovery).
  """
  asset = _get_robot(env, asset_cfg)
  feet_sensor: ContactSensor = env.scene[feet_sensor_cfg.name]

  force = feet_sensor.data.force
  assert force is not None
  foot_mask = force.norm(dim=-1) > 20.0

  foot_ids, _ = asset.find_bodies(FEET_BODY_NAMES, preserve_order=True)
  foot_pos_xy = asset.data.body_link_pos_w[:, foot_ids, :2]

  contact_pos = foot_pos_xy
  contact_mask = foot_mask

  if hands_sensor_cfg is not None:
    hand_sensor: ContactSensor = env.scene[hands_sensor_cfg.name]
    hand_force = hand_sensor.data.force
    if hand_force is not None:
      _HAND_BODY_NAMES = ("left_wrist_yaw_link", "right_wrist_yaw_link")
      hand_ids, _ = asset.find_bodies(_HAND_BODY_NAMES, preserve_order=True)
      hand_pos_xy = asset.data.body_link_pos_w[:, hand_ids, :2]
      hand_mask = hand_force.norm(dim=-1) > 20.0
      contact_pos = torch.cat([contact_pos, hand_pos_xy], dim=1)
      contact_mask = torch.cat([contact_mask, hand_mask], dim=1)

  com_xy = _whole_body_com_pos(asset)[:, :2]
  has_contact = contact_mask.any(dim=1)
  dist = _support_polygon_dist(com_xy, contact_pos, contact_mask, tolerance)
  return torch.exp(-(dist**2) / (std**2)) * has_contact.float()


def simplified_capture_point_in_support_polygon(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  feet_sensor_cfg: SceneEntityCfg = SceneEntityCfg("feet_ground_contact"),
  wall_sensor_cfg: SceneEntityCfg | None = None,
  table_sensor_cfg: SceneEntityCfg | None = None,
  tolerance: float = 0.05,
  std: float = 0.1,
) -> torch.Tensor:
  """Reward LIPM capture point staying inside the active foot support polygon.

  When the robot has wall/table contact, bypasses the foot-polygon check and
  returns max reward (legitimately stable through multi-surface support).
  """
  asset = _get_robot(env, asset_cfg)
  feet_sensor: ContactSensor = env.scene[feet_sensor_cfg.name]

  force = feet_sensor.data.force
  assert force is not None
  foot_mask = force.norm(dim=-1) > 50.0

  foot_ids, _ = asset.find_bodies(FEET_BODY_NAMES, preserve_order=True)
  foot_pos_xy = asset.data.body_link_pos_w[:, foot_ids, :2]

  cp_xy = _capture_point(asset)
  has_contact = foot_mask.any(dim=1)
  dist = _support_polygon_dist(cp_xy, foot_pos_xy, foot_mask, tolerance)
  reward = torch.exp(-(dist**2) / (std**2)) * has_contact.float()

  if wall_sensor_cfg is not None:
    wall_sensor: ContactSensor = env.scene[wall_sensor_cfg.name]
    wf = wall_sensor.data.force
    if wf is not None:
      wall_active = wf.norm(dim=-1).sum(dim=-1) > 10.0
      reward = torch.where(wall_active, torch.ones_like(reward), reward)

  if table_sensor_cfg is not None:
    table_sensor: ContactSensor = env.scene[table_sensor_cfg.name]
    tf = table_sensor.data.force
    if tf is not None:
      table_active = tf.norm(dim=-1).sum(dim=-1) > 10.0
      reward = torch.where(table_active, torch.ones_like(reward), reward)

  return reward


class AnkleHipStepReward:
  """Ankle → Hip → Step strategy reward hierarchy for the G1.

  Joint IDs and actuator limits are cached at ``__init__`` to avoid per-step
  string matching. Reward levels (additive):
    • **Ankle** – torque opposes CoM drift, within friction cone.
    • **Hip**   – activated when ankle torque saturates.
    • **Step**  – swing foot velocity toward capture point when ankle + hip
      strategies are saturated.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    self._asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
    asset = _get_robot(env, self._asset_cfg)

    self._ap_ids = asset.find_joints(".*ankle_pitch.*")[0]
    self._ar_ids = asset.find_joints(".*ankle_roll.*")[0]
    self._hp_ids = asset.find_joints(".*hip_pitch.*")[0]
    self._hr_ids = asset.find_joints(".*hip_roll.*")[0]

    self._ankle_act_ids = asset.find_actuators(".*ankle.*")[0]
    self._hip_p_act_ids = asset.find_actuators(".*hip_pitch.*")[0]
    self._hip_r_act_ids = asset.find_actuators(".*hip_roll.*")[0]

    self._foot_ids, _ = asset.find_bodies(FEET_BODY_NAMES, preserve_order=True)

    feet_sensor_cfg: SceneEntityCfg = cfg.params["feet_sensor_cfg"]
    self._sensor_name: str = feet_sensor_cfg.name
    self._friction_coeff: float = cfg.params.get("friction_coeff", 0.7)
    self._cop_alpha: float = cfg.params.get("cop_alpha", 1e-3)
    self._contact_threshold: float = cfg.params.get("contact_force_threshold", 20.0)

  def _effort_limit(self, env: ManagerBasedRlEnv, act_ids: list[int]) -> torch.Tensor:
    fr = env.sim.model.actuator_forcerange
    if fr.ndim == 2:
      return fr[act_ids, 1].abs().max()
    return fr[:, act_ids, 1].abs().max(dim=-1).values

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
    feet_sensor_cfg: SceneEntityCfg = SceneEntityCfg("feet_ground_contact"),
    friction_coeff: float = 0.7,
    cop_alpha: float = 1e-3,
    contact_force_threshold: float = 20.0,
  ) -> torch.Tensor:
    asset = _get_robot(env, self._asset_cfg)
    feet_sensor: ContactSensor = env.scene[self._sensor_name]

    ankle_limit = self._effort_limit(env, self._ankle_act_ids)
    hip_p_limit = self._effort_limit(env, self._hip_p_act_ids)
    hip_r_limit = self._effort_limit(env, self._hip_r_act_ids)

    tau = asset.data.qfrc_actuator
    ap_tau = tau[:, self._ap_ids].sum(dim=1)
    ar_tau = tau[:, self._ar_ids].sum(dim=1)
    hp_tau = tau[:, self._hp_ids].sum(dim=1)
    hr_tau = tau[:, self._hr_ids].sum(dim=1)

    force = feet_sensor.data.force
    assert force is not None
    fx, fy = force[..., 0], force[..., 1]
    fz = force[..., 2].clamp(min=0.0)
    tangential = torch.sqrt(fx**2 + fy**2)
    contact_mask = fz > self._contact_threshold

    margin = self._friction_coeff * fz - tangential
    cone_sat = (
      1.0 - (margin / (self._friction_coeff * fz + 1e-6)).clamp(0.0, 1.0)
    ).max(dim=1).values

    ankle_ps = torch.clamp(ap_tau.abs() / ankle_limit, 0, 1)
    ankle_rs = torch.clamp(ar_tau.abs() / ankle_limit, 0, 1)
    hip_ps   = torch.clamp(hp_tau.abs() / hip_p_limit, 0, 1)
    hip_rs   = torch.clamp(hr_tau.abs() / hip_r_limit, 0, 1)
    ankle_ps = torch.maximum(ankle_ps, cone_sat)
    ankle_rs = torch.maximum(ankle_rs, cone_sat)
    hip_ps   = torch.maximum(hip_ps,   cone_sat)
    hip_rs   = torch.maximum(hip_rs,   cone_sat)

    com_vel_xy = asset.data.root_com_lin_vel_w[:, :2]
    speed = com_vel_xy.norm(dim=1).clamp(min=1e-6)
    com_dir = com_vel_xy / speed.unsqueeze(1)

    r_ankle = (
      torch.tanh(-com_dir[:, 0] * ap_tau / ankle_limit).clamp(min=0)
      * (1 - _cop_penalty(ankle_ps, self._cop_alpha))
      + torch.tanh(-com_dir[:, 1] * ar_tau / ankle_limit).clamp(min=0)
      * (1 - _cop_penalty(ankle_rs, self._cop_alpha))
    )

    r_hip = (
      torch.tanh(-com_dir[:, 0] * hp_tau / hip_p_limit).clamp(min=0)
      * ankle_ps**2
      * (1 - _cop_penalty(hip_ps, self._cop_alpha))
      + torch.tanh(-com_dir[:, 1] * hr_tau / hip_r_limit).clamp(min=0)
      * ankle_rs**2
      * (1 - _cop_penalty(hip_rs, self._cop_alpha))
    )

    cp_xy = _capture_point(asset)
    foot_pos_xy = asset.data.body_link_pos_w[:, self._foot_ids, :2]
    foot_vel_xy = asset.data.body_link_lin_vel_w[:, self._foot_ids, :2]

    to_cp = cp_xy.unsqueeze(1) - foot_pos_xy
    to_cp_dir = to_cp / to_cp.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    vel_toward = (foot_vel_xy * to_cp_dir).sum(dim=-1).clamp(min=0.0)

    swing_mask = (~contact_mask).float()
    step_quality = torch.tanh(
      (vel_toward * swing_mask).max(dim=1).values / 0.5
    )
    step_urgency = torch.minimum(ankle_ps * hip_ps, ankle_rs * hip_rs)
    r_step = step_quality * step_urgency

    return r_ankle + r_hip + r_step

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    pass


class LinearMomentumChangePenalty:
  """Penalize rate of change of whole-body linear momentum (F = dp/dt).

  Returns a *negative* squared penalty; pair with a positive weight.
  Spin angular momentum is excluded (orbital term only).
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    self._asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
    self._env = env
    asset = _get_robot(env, self._asset_cfg)
    self._prev_lin_mom: torch.Tensor = _linear_momentum(asset).clone()

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    asset = _get_robot(env, self._asset_cfg)
    cur = _linear_momentum(asset)
    net_force = (cur - self._prev_lin_mom) / env.step_dt
    self._prev_lin_mom = cur.clone()
    return -torch.sum(net_force**2, dim=-1)

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    asset = _get_robot(self._env, self._asset_cfg)
    full = _linear_momentum(asset)
    if env_ids is None:
      self._prev_lin_mom = full.clone()
    else:
      self._prev_lin_mom[env_ids] = full[env_ids]


class AngularMomentumChangePenalty:
  """Penalize rate of change of whole-body angular momentum (τ = dL/dt).

  Uses orbital angular momentum only (no body spin term).
  Returns a *negative* squared penalty; pair with a positive weight.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv) -> None:
    self._asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
    self._env = env
    asset = _get_robot(env, self._asset_cfg)
    self._prev_ang_mom: torch.Tensor = _angular_momentum(asset).clone()

  def __call__(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    asset = _get_robot(env, self._asset_cfg)
    cur = _angular_momentum(asset)
    net_torque = (cur - self._prev_ang_mom) / env.step_dt
    self._prev_ang_mom = cur.clone()
    return -torch.sum(net_torque**2, dim=-1)

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    asset = _get_robot(self._env, self._asset_cfg)
    full = _angular_momentum(asset)
    if env_ids is None:
      self._prev_ang_mom = full.clone()
    else:
      self._prev_ang_mom[env_ids] = full[env_ids]


def _expand_gravity_for_bodies(
  gravity_vec_w: torch.Tensor, *, num_envs: int, num_bodies: int
) -> torch.Tensor:
  """Broadcast gravity to `[num_envs, num_bodies, 3]` for body-wise projections."""
  if gravity_vec_w.ndim == 1:
    gravity_vec_w = gravity_vec_w.unsqueeze(0)

  if gravity_vec_w.shape[0] == 1 and num_envs != 1:
    gravity_vec_w = gravity_vec_w.expand(num_envs, -1)
  elif gravity_vec_w.shape[0] != num_envs:
    raise ValueError(
      f"Expected gravity_vec_w to have {num_envs} rows, got {gravity_vec_w.shape[0]}"
    )

  return gravity_vec_w.unsqueeze(1).expand(-1, num_bodies, -1)


def _motion_command_is_active(
  env: ManagerBasedRlEnv,
  *,
  command_name: str,
  command_threshold: float,
  cmd_limits: tuple[float, float, float] = (1.0, 1.0, 1.0),
  uniform_cmd_name: str | None = None,
) -> torch.Tensor:
  """Return activity mask based on loco command magnitude.

  When ``uniform_cmd_name`` is given, gate by the uniform command instead of
  the motion anchor so the reward fires consistently with what the student
  sees in its observation slot.
  """
  command = motion_loco_command(
    env,
    command_name=command_name,
    cmd_limits=cmd_limits,
    uniform_cmd_name=uniform_cmd_name,
  )
  return (torch.norm(command, dim=-1) > command_threshold).float()






def tracking_joint_dof(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  dof_err_w: tuple[float, ...] | None = None,
) -> torch.Tensor:
  command = get_motion_command(env, command_name)
  dof_diff = command.joint_pos - command.robot_joint_pos
  if dof_err_w is None:
    weights = torch.ones(dof_diff.shape[-1], device=env.device, dtype=dof_diff.dtype)
  else:
    weights = torch.tensor(dof_err_w, device=env.device, dtype=dof_diff.dtype)
  dof_err = torch.sum(weights * torch.square(dof_diff), dim=-1)
  return torch.exp(-0.15 * dof_err)


def tracking_joint_vel(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  dof_err_w: tuple[float, ...] | None = None,
) -> torch.Tensor:
  command = get_motion_command(env, command_name)
  vel_diff = command.joint_vel - command.robot_joint_vel
  if dof_err_w is None:
    weights = torch.ones(vel_diff.shape[-1], device=env.device, dtype=vel_diff.dtype)
  else:
    weights = torch.tensor(dof_err_w, device=env.device, dtype=vel_diff.dtype)
  vel_err = torch.sum(weights * torch.square(vel_diff), dim=-1)
  return torch.exp(-0.01 * vel_err)


def tracking_root_translation_z(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
) -> torch.Tensor:
  command = get_motion_command(env, command_name)
  z_err_sq = torch.square(command.body_pos_w[:, 0, 2] - command.robot_body_pos_w[:, 0, 2])
  return torch.exp(-5.0 * z_err_sq)








def tracking_root_rotation(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  include_yaw: bool = True,
) -> torch.Tensor:
  """Reward closeness of robot root rotation to motion root rotation.

  ``include_yaw=False`` drops the yaw component and scores only roll and
  pitch against the motion. Use this when yaw is being controlled via a
  separate yaw-rate target (e.g. uniform cmd's wz via
  ``tracking_root_angular_vel``).
  """
  command = get_motion_command(env, command_name)
  robot_quat = command.robot_body_quat_w[:, 0]
  motion_quat = command.body_quat_w[:, 0]
  if include_yaw:
    quat_err_sq = torch.square(
      quat_error_magnitude(robot_quat, motion_quat)
    )
    return torch.exp(-5.0 * quat_err_sq)
  r_robot, p_robot, _ = euler_xyz_from_quat(robot_quat)
  r_motion, p_motion, _ = euler_xyz_from_quat(motion_quat)
  roll_err = wrap_to_pi(r_robot - r_motion)
  pitch_err = wrap_to_pi(p_robot - p_motion)
  err_sq = torch.square(roll_err) + torch.square(pitch_err)
  return torch.exp(-5.0 * err_sq)


def tracking_root_linear_vel(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  uniform_cmd_name: str | None = None,
  std: float = 1.0,
) -> torch.Tensor:
  """Reward robot lin vel matching the target lin vel.

  Default target: motion anchor body-frame linear velocity (full 3D).
  If ``uniform_cmd_name`` is set, target is the uniform cmd (vx, vy) and
  only xy is scored (uniform cmd does not specify vz).
  Reward is ``exp(-err_sq / std**2)``.
  """
  del asset_cfg
  command = get_motion_command(env, command_name)
  robot_lin_vel_b = command.robot.data.root_link_lin_vel_b
  if uniform_cmd_name is not None:
    uni_xy = env_mdp.generated_commands(env, command_name=uniform_cmd_name)[:, :2]
    vel_err_sq = torch.sum(
      torch.square(uni_xy - robot_lin_vel_b[:, :2]), dim=-1
    )
  else:
    ref_lin_vel_b = quat_apply_inverse(
      command.body_quat_w[:, 0], command.body_lin_vel_w[:, 0]
    )
    vel_err_sq = torch.sum(
      torch.square(ref_lin_vel_b - robot_lin_vel_b), dim=-1
    )
  return torch.exp(-vel_err_sq / (std ** 2))






def tracking_root_angular_vel(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  uniform_cmd_name: str | None = None,
  std: float = 1.0,
) -> torch.Tensor:
  """Reward robot ang vel matching the target ang vel.

  Default target: motion anchor body-frame angular velocity (full 3D).
  If ``uniform_cmd_name`` is set, target is the uniform cmd's wz and only
  the z component is scored; roll/pitch rates (wx, wy) are regularized
  separately by ``ang_vel_xy``.
  Reward is ``exp(-err_sq / std**2)``.
  """
  del asset_cfg
  command = get_motion_command(env, command_name)
  robot_ang_vel_b = command.robot.data.root_link_ang_vel_b
  if uniform_cmd_name is not None:
    uni_wz = env_mdp.generated_commands(env, command_name=uniform_cmd_name)[:, 2]
    vel_err_sq = torch.square(uni_wz - robot_ang_vel_b[:, 2])
  else:
    ref_ang_vel_b = quat_apply_inverse(
      command.body_quat_w[:, 0], command.body_ang_vel_w[:, 0]
    )
    vel_err_sq = torch.sum(
      torch.square(ref_ang_vel_b - robot_ang_vel_b), dim=-1
    )
  return torch.exp(-vel_err_sq / (std ** 2))


def tracking_keybody_pos(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
) -> torch.Tensor:
  command = get_motion_command(env, command_name)
  key_body_indices = tracked_body_indices(command)

  robot_root_pos_w = command.robot_body_pos_w[:, 0]
  robot_root_quat_w = command.robot_body_quat_w[:, 0]
  ref_root_pos_w = command.body_pos_w[:, 0]
  ref_root_quat_w = command.body_quat_w[:, 0]

  robot_delta_w = command.robot_body_pos_w[:, key_body_indices] - robot_root_pos_w[:, None, :]
  ref_delta_w = command.body_pos_w[:, key_body_indices] - ref_root_pos_w[:, None, :]

  robot_yaw_quat = yaw_quat(robot_root_quat_w)[:, None, :].expand(-1, len(KEY_BODY_NAMES), -1)
  ref_yaw_quat = yaw_quat(ref_root_quat_w)[:, None, :].expand(-1, len(KEY_BODY_NAMES), -1)

  robot_delta_b = quat_apply_inverse(
    robot_yaw_quat.reshape(-1, 4), robot_delta_w.reshape(-1, 3)
  ).reshape(env.num_envs, len(KEY_BODY_NAMES), 3)
  ref_delta_b = quat_apply_inverse(
    ref_yaw_quat.reshape(-1, 4), ref_delta_w.reshape(-1, 3)
  ).reshape(env.num_envs, len(KEY_BODY_NAMES), 3)

  key_err_sq = torch.sum(torch.square(robot_delta_b - ref_delta_b), dim=-1).sum(dim=-1)
  return torch.exp(-10.0 * key_err_sq)


def tracking_keybody_pos_global(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
) -> torch.Tensor:
  command = get_motion_command(env, command_name)
  key_body_indices = tracked_body_indices(command)
  key_err_sq = torch.sum(
    torch.square(
      command.robot_body_pos_w[:, key_body_indices] - command.body_pos_w[:, key_body_indices]
    ),
    dim=-1,
  ).sum(dim=-1)
  return torch.exp(-10.0 * key_err_sq)










def tracking_hand_pos(
  env: ManagerBasedRlEnv,
  command_name: str = "motion",
  body_names: tuple[str, ...] = HAND_BODY_NAMES,
) -> torch.Tensor:
  robot_hand_pos = robot_hand_pos_b_tensor(
    env,
    command_name=command_name,
    body_names=body_names,
  )
  ref_hand_pos = motion_hand_pos_b_tensor(
    env,
    command_name=command_name,
    body_names=body_names,
  )
  hand_err_sq = torch.sum(torch.square(robot_hand_pos - ref_hand_pos), dim=-1).sum(dim=-1)
  return torch.exp(-10.0 * hand_err_sq)


def feet_contact_forces(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  max_contact_force: float = 500.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  assert force is not None
  vertical_force = torch.abs(force[..., 2])
  return torch.clamp(vertical_force - max_contact_force, min=0.0).sum(dim=1)


def feet_stumble(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  assert force is not None
  horizontal = torch.norm(force[..., :2], dim=-1)
  vertical = torch.abs(force[..., 2])
  return torch.any(horizontal > 4.0 * vertical, dim=1).float()


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  contact_force_threshold: float = 5.0,
) -> torch.Tensor:
  asset = _get_robot(env, asset_cfg)
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  assert force is not None
  contact = torch.abs(force[..., 2]) > contact_force_threshold
  foot_ids, _ = asset.find_bodies(FEET_BODY_NAMES, preserve_order=True)
  foot_vel_xy = asset.data.body_link_lin_vel_w[:, foot_ids, :2]
  foot_speed_norm = torch.norm(foot_vel_xy, dim=-1)
  slip = torch.sqrt(torch.clamp(foot_speed_norm, min=0.0))
  return torch.sum(slip * contact.float(), dim=1)




def ang_vel_xy(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset = _get_robot(env, asset_cfg)
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def dense_swing_height(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  height_sensor_name: str,
  target_height: float,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize insufficient clearance only while a commanded foot is airborne."""
  if target_height <= 0.0:
    raise ValueError(f"target_height must be positive, got {target_height}")

  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.found is not None
  height_sensor = env.scene[height_sensor_name]
  assert isinstance(height_sensor, TerrainHeightSensor), (
    "dense_swing_height requires a TerrainHeightSensor, "
    f"got {type(height_sensor).__name__}"
  )

  foot_heights = height_sensor.data.heights
  in_air = contact_sensor.data.found == 0
  assert foot_heights.shape == in_air.shape, (
    "dense_swing_height requires matching height/contact frames, "
    f"got heights {tuple(foot_heights.shape)} and contacts {tuple(in_air.shape)}"
  )

  active = torch.ones(env.num_envs, device=env.device)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      active = (
        linear_norm + angular_norm > command_threshold
      ).float()

  swing_mask = in_air.float() * active.unsqueeze(1)
  normalized_deficit = torch.relu(target_height - foot_heights) / target_height
  cost = torch.sum(torch.square(normalized_deficit) * swing_mask, dim=1)

  num_swing_feet = torch.clamp(torch.sum(swing_mask), min=1.0)
  env.extras["log"]["Metrics/swing_foot_height_mean"] = (
    torch.sum(foot_heights * swing_mask) / num_swing_feet
  )
  env.extras["log"]["Metrics/swing_height_deficit_mean"] = (
    torch.sum(normalized_deficit * swing_mask) / num_swing_feet
  )
  return cost


class swing_height_curve:
  """Track a half-sine foot-height curve over each contact-defined swing.

  Each foot uses its contact sensor's ``current_air_time`` as swing progress,
  so this term does not require an externally supplied gait phase.  The target
  starts at zero at lift-off, reaches ``peak_height`` halfway through the
  desired ``swing_time``, and returns to zero at the desired touchdown time.
  Scalar targets are shared by all feet; tuples provide per-foot targets in
  the same order as the contact and height sensor frames.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    height_sensor = env.scene[cfg.params["height_sensor_name"]]
    assert isinstance(height_sensor, TerrainHeightSensor), (
      "swing_height_curve requires a TerrainHeightSensor, "
      f"got {type(height_sensor).__name__}"
    )
    num_feet = height_sensor.num_frames

    def _per_foot_target(
      value: float | tuple[float, ...], name: str
    ) -> torch.Tensor:
      target = torch.as_tensor(value, device=env.device, dtype=torch.float32)
      if target.ndim == 0:
        target = target.repeat(num_feet)
      else:
        target = target.flatten()
      if target.numel() != num_feet:
        raise ValueError(
          f"{name} must be a scalar or contain one value per foot "
          f"({num_feet}), got {target.numel()}"
        )
      if bool(torch.any(target <= 0.0)):
        raise ValueError(f"{name} values must all be positive, got {value}")
      return target.unsqueeze(0)

    self.peak_height = _per_foot_target(cfg.params["peak_height"], "peak_height")
    self.swing_time = _per_foot_target(cfg.params["swing_time"], "swing_time")

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    height_sensor_name: str,
    peak_height: float | tuple[float, ...],
    swing_time: float | tuple[float, ...],
    command_name: str | None = None,
    command_threshold: float = 0.05,
  ) -> torch.Tensor:
    del peak_height, swing_time  # Precomputed once in ``__init__``.
    contact_sensor: ContactSensor = env.scene[sensor_name]
    assert contact_sensor.data.found is not None
    assert contact_sensor.data.current_air_time is not None
    height_sensor = env.scene[height_sensor_name]
    assert isinstance(height_sensor, TerrainHeightSensor)

    foot_heights = height_sensor.data.heights
    current_air_time = contact_sensor.data.current_air_time
    in_air = contact_sensor.data.found == 0
    assert foot_heights.shape == current_air_time.shape == in_air.shape, (
      "swing_height_curve requires matching height/contact frames, "
      f"got heights {tuple(foot_heights.shape)}, air times "
      f"{tuple(current_air_time.shape)}, and contacts {tuple(in_air.shape)}"
    )

    swing_progress = torch.clamp(
      current_air_time / self.swing_time, min=0.0, max=1.0
    )
    desired_height = self.peak_height * torch.sin(torch.pi * swing_progress)

    active = torch.ones(env.num_envs, device=env.device)
    if command_name is not None:
      command = env.command_manager.get_command(command_name)
      if command is not None:
        linear_norm = torch.norm(command[:, :2], dim=1)
        angular_norm = torch.abs(command[:, 2])
        active = (linear_norm + angular_norm > command_threshold).float()

    swing_mask = in_air.float() * active.unsqueeze(1)
    normalized_error = (foot_heights - desired_height) / self.peak_height
    squared_error = torch.square(normalized_error) * swing_mask
    cost = torch.sum(squared_error, dim=1)

    num_swing_feet = torch.clamp(torch.sum(swing_mask), min=1.0)
    env.extras["log"]["Metrics/swing_foot_height_mean"] = (
      torch.sum(foot_heights * swing_mask) / num_swing_feet
    )
    env.extras["log"]["Metrics/swing_height_target_mean"] = (
      torch.sum(desired_height * swing_mask) / num_swing_feet
    )
    env.extras["log"]["Metrics/swing_height_curve_rmse"] = torch.sqrt(
      torch.sum(squared_error) / num_swing_feet
    )
    return cost


def _positive_per_foot_target(
  value: float | tuple[float, ...],
  *,
  num_feet: int,
  device: torch.device | str,
  name: str,
) -> torch.Tensor:
  """Convert a scalar/per-foot target to a positive ``[1, num_feet]`` tensor."""
  target = torch.as_tensor(value, device=device, dtype=torch.float32)
  if target.ndim == 0:
    target = target.repeat(num_feet)
  else:
    target = target.flatten()
  if target.numel() != num_feet:
    raise ValueError(
      f"{name} must be scalar or contain {num_feet} values, got {target.numel()}"
    )
  if bool(torch.any(target <= 0.0)):
    raise ValueError(f"{name} values must all be positive, got {value}")
  return target.unsqueeze(0)


def command_turning_blend(
  command: torch.Tensor,
  *,
  turning_linear_threshold: float,
  turning_angular_threshold: float,
) -> torch.Tensor:
  """Return a smooth 0=translation, 1=in-place-turn command blend.

  A pure turn with ``|wz| >= turning_angular_threshold`` receives 1.  The
  blend fades to zero as linear speed reaches ``turning_linear_threshold``;
  normal curved walking therefore keeps the translational gait target.
  """
  if turning_linear_threshold <= 0.0 or turning_angular_threshold <= 0.0:
    raise ValueError("Turning blend thresholds must be positive")
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  low_translation = torch.clamp(
    1.0 - linear_norm / turning_linear_threshold, min=0.0, max=1.0
  )
  active_turn = torch.clamp(
    angular_norm / turning_angular_threshold, min=0.0, max=1.0
  )
  return low_translation * active_turn


class command_conditioned_swing_height_curve:
  """Track separate translational and in-place-turn swing-height curves."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    height_sensor = env.scene[cfg.params["height_sensor_name"]]
    assert isinstance(height_sensor, TerrainHeightSensor), (
      "command_conditioned_swing_height_curve requires a TerrainHeightSensor, "
      f"got {type(height_sensor).__name__}"
    )
    num_feet = height_sensor.num_frames
    self.translation_peak_height = _positive_per_foot_target(
      cfg.params["translation_peak_height"],
      num_feet=num_feet,
      device=env.device,
      name="translation_peak_height",
    )
    self.turning_peak_height = _positive_per_foot_target(
      cfg.params["turning_peak_height"],
      num_feet=num_feet,
      device=env.device,
      name="turning_peak_height",
    )
    self.translation_swing_time = _positive_per_foot_target(
      cfg.params["translation_swing_time"],
      num_feet=num_feet,
      device=env.device,
      name="translation_swing_time",
    )
    self.turning_swing_time = _positive_per_foot_target(
      cfg.params["turning_swing_time"],
      num_feet=num_feet,
      device=env.device,
      name="turning_swing_time",
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    height_sensor_name: str,
    translation_peak_height: float | tuple[float, ...],
    turning_peak_height: float | tuple[float, ...],
    translation_swing_time: float | tuple[float, ...],
    turning_swing_time: float | tuple[float, ...],
    command_name: str = "twist",
    command_threshold: float = 0.05,
    turning_linear_threshold: float = 0.2,
    turning_angular_threshold: float = 0.2,
  ) -> torch.Tensor:
    del (
      translation_peak_height,
      turning_peak_height,
      translation_swing_time,
      turning_swing_time,
    )
    contact_sensor: ContactSensor = env.scene[sensor_name]
    assert contact_sensor.data.found is not None
    assert contact_sensor.data.current_air_time is not None
    height_sensor = env.scene[height_sensor_name]
    assert isinstance(height_sensor, TerrainHeightSensor)

    command = env.command_manager.get_command(command_name)
    assert command is not None
    turn_blend = command_turning_blend(
      command,
      turning_linear_threshold=turning_linear_threshold,
      turning_angular_threshold=turning_angular_threshold,
    ).unsqueeze(1)
    peak_height = torch.lerp(
      self.translation_peak_height, self.turning_peak_height, turn_blend
    )
    swing_time = torch.lerp(
      self.translation_swing_time, self.turning_swing_time, turn_blend
    )

    foot_heights = height_sensor.data.heights
    current_air_time = contact_sensor.data.current_air_time
    in_air = contact_sensor.data.found == 0
    assert foot_heights.shape == current_air_time.shape == in_air.shape, (
      "command-conditioned swing reward requires matching height/contact "
      f"frames, got {tuple(foot_heights.shape)}, "
      f"{tuple(current_air_time.shape)}, and {tuple(in_air.shape)}"
    )

    swing_progress = torch.clamp(current_air_time / swing_time, 0.0, 1.0)
    desired_height = peak_height * torch.sin(torch.pi * swing_progress)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    active = (linear_norm + angular_norm > command_threshold).float()
    swing_mask = in_air.float() * active.unsqueeze(1)
    normalized_error = (foot_heights - desired_height) / peak_height
    squared_error = torch.square(normalized_error) * swing_mask
    cost = torch.sum(squared_error, dim=1)

    num_swing_feet = torch.clamp(torch.sum(swing_mask), min=1.0)
    env.extras["log"]["Metrics/swing_foot_height_mean"] = (
      torch.sum(foot_heights * swing_mask) / num_swing_feet
    )
    env.extras["log"]["Metrics/swing_height_target_mean"] = (
      torch.sum(desired_height * swing_mask) / num_swing_feet
    )
    env.extras["log"]["Metrics/swing_height_curve_rmse"] = torch.sqrt(
      torch.sum(squared_error) / num_swing_feet
    )
    env.extras["log"]["Metrics/swing_height_turn_blend_mean"] = (
      turn_blend.mean()
    )
    return cost


class command_conditioned_feet_swing_height:
  """Penalize landing peak-height error with a lower pure-turn target."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    height_sensor = env.scene[cfg.params["height_sensor_name"]]
    assert isinstance(height_sensor, TerrainHeightSensor), (
      "command_conditioned_feet_swing_height requires a TerrainHeightSensor, "
      f"got {type(height_sensor).__name__}"
    )
    num_feet = height_sensor.num_frames
    self.peak_heights = torch.zeros(
      (env.num_envs, num_feet), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    height_sensor_name: str,
    translation_target_height: float,
    turning_target_height: float,
    command_name: str = "twist",
    command_threshold: float = 0.05,
    turning_linear_threshold: float = 0.2,
    turning_angular_threshold: float = 0.2,
  ) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[sensor_name]
    assert contact_sensor.data.found is not None
    height_sensor = env.scene[height_sensor_name]
    assert isinstance(height_sensor, TerrainHeightSensor)
    foot_heights = height_sensor.data.heights
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air, torch.maximum(self.peak_heights, foot_heights), self.peak_heights
    )

    command = env.command_manager.get_command(command_name)
    assert command is not None
    turn_blend = command_turning_blend(
      command,
      turning_linear_threshold=turning_linear_threshold,
      turning_angular_threshold=turning_angular_threshold,
    )
    target_height = torch.lerp(
      torch.full_like(turn_blend, translation_target_height),
      torch.full_like(turn_blend, turning_target_height),
      turn_blend,
    ).unsqueeze(1)
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    active = (linear_norm + angular_norm > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active

    landing_mask = first_contact.float() * active.unsqueeze(1)
    num_landings = torch.clamp(torch.sum(landing_mask), min=1.0)
    env.extras["log"]["Metrics/peak_height_mean"] = (
      torch.sum(self.peak_heights * landing_mask) / num_landings
    )
    env.extras["log"]["Metrics/peak_height_target_mean"] = (
      torch.sum(target_height * landing_mask) / num_landings
    )
    self.peak_heights = torch.where(
      first_contact, torch.zeros_like(self.peak_heights), self.peak_heights
    )
    return cost

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      self.peak_heights.zero_()
    else:
      self.peak_heights[env_ids] = 0.0


def command_conditioned_feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold_min: float = 0.05,
  translation_threshold_max: float = 0.60,
  turning_threshold_max: float = 0.42,
  command_name: str = "twist",
  command_threshold: float = 0.2,
  turning_linear_threshold: float = 0.2,
  turning_angular_threshold: float = 0.2,
) -> torch.Tensor:
  """Dense air-time reward with a shorter window for in-place turns."""
  sensor: ContactSensor = env.scene[sensor_name]
  current_air_time = sensor.data.current_air_time
  assert current_air_time is not None
  command = env.command_manager.get_command(command_name)
  assert command is not None
  turn_blend = command_turning_blend(
    command,
    turning_linear_threshold=turning_linear_threshold,
    turning_angular_threshold=turning_angular_threshold,
  )
  threshold_max = torch.lerp(
    torch.full_like(turn_blend, translation_threshold_max),
    torch.full_like(turn_blend, turning_threshold_max),
    turn_blend,
  ).unsqueeze(1)
  in_range = (current_air_time > threshold_min) & (
    current_air_time < threshold_max
  )
  reward = torch.sum(in_range.float(), dim=1)
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  active = (linear_norm + angular_norm > command_threshold).float()

  in_air = current_air_time > 0.0
  num_in_air = torch.clamp(torch.sum(in_air.float()), min=1.0)
  env.extras["log"]["Metrics/air_time_mean"] = (
    torch.sum(current_air_time * in_air.float()) / num_in_air
  )
  env.extras["log"]["Metrics/air_time_max_target_mean"] = threshold_max.mean()
  return reward * active


def standing_foot_distance(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float,
  target_lateral_distance: float,
  target_fore_distance: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize standing foot spacing measured in the robot root frame."""
  asset = _get_robot(env, asset_cfg)
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  standing = (total_command < command_threshold).float()

  foot_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
  left_right_delta_w = foot_pos_w[:, 0] - foot_pos_w[:, 1]
  left_right_delta_b = quat_apply_inverse(
    asset.data.root_link_quat_w, left_right_delta_w
  )
  fore_distance = torch.abs(left_right_delta_b[:, 0])
  lateral_distance = torch.abs(left_right_delta_b[:, 1])
  fore_error = torch.square(fore_distance - target_fore_distance)
  lateral_error = torch.square(lateral_distance - target_lateral_distance)
  cost = (fore_error + lateral_error) * standing

  denom = torch.clamp(torch.sum(standing), min=1)
  env.extras["log"]["Metrics/standing_foot_fore_error"] = (
    torch.sum(torch.sqrt(fore_error) * standing) / denom
  )
  env.extras["log"]["Metrics/standing_foot_lateral_error"] = (
    torch.sum(torch.sqrt(lateral_error) * standing) / denom
  )
  return cost


def stand_pose(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset = _get_robot(env, asset_cfg)
  default_joint_pos = asset.data.default_joint_pos
  joint_ids = asset_cfg.joint_ids
  error = torch.sum(
    torch.square(asset.data.joint_pos[:, joint_ids] - default_joint_pos[:, joint_ids]),
    dim=-1,
  )
  twist_cmd = env.command_manager.get_term(command_name)
  return error * twist_cmd.is_standing_env.float()


def flat_foot(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  contact_force_threshold: float = 1.0,
) -> torch.Tensor:
  asset = _get_robot(env, asset_cfg)
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  assert force is not None
  contact = (torch.norm(force[..., :3], dim=-1) > contact_force_threshold).float()
  foot_quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
  gravity = _expand_gravity_for_bodies(
    asset.data.gravity_vec_w,
    num_envs=env.num_envs,
    num_bodies=foot_quat.shape[1],
  )
  projected = quat_apply_inverse(
    foot_quat.reshape(-1, 4), gravity.reshape(-1, 3)
  ).reshape(env.num_envs, len(asset_cfg.body_ids), 3)
  tilt_error = torch.sum(torch.square(projected[..., :2]), dim=-1)
  return torch.sum(tilt_error * contact, dim=-1)


def gait_phase_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str = "twist",
  gait_period: float = 1.0,
  gait_offset: float = 0.5,
  stance_ratio: float = LOCO_STANCE_RATIO,
  contact_force_threshold: float = 1.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  assert force is not None
  contact = torch.norm(force[..., :3], dim=-1) > contact_force_threshold
  expected_stance = loco_leg_phase(
    env,
    command_name=command_name,
    gait_period=gait_period,
    gait_offset=gait_offset,
  ) < stance_ratio
  return (~(contact ^ expected_stance)).float().sum(dim=-1)


def motion_gait_phase_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str = "motion",
  gait_period: float = 1.0,
  gait_offset: float = 0.5,
  stance_ratio: float = LOCO_STANCE_RATIO,
  contact_force_threshold: float = 1.0,
  stand_vel_threshold: float = 0.1,
  cmd_limits: tuple[float, float, float] = (1.0, 1.0, 1.0),
  uniform_cmd_name: str | None = None,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  assert force is not None
  contact = torch.norm(force[..., :3], dim=-1) > contact_force_threshold
  expected_stance = motion_loco_leg_phase(
    env,
    command_name=command_name,
    gait_period=gait_period,
    gait_offset=gait_offset,
    stand_vel_threshold=stand_vel_threshold,
    cmd_limits=cmd_limits,
    uniform_cmd_name=uniform_cmd_name,
  ) < stance_ratio
  return (~(contact ^ expected_stance)).float().sum(dim=-1)


def feet_distance_lateral(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  min_distance: float,
  max_distance: float,
) -> torch.Tensor:
  """Reward zero inside a lateral foot-spacing band and negative outside it.

  The two feet may be selected by either body names (the original G1 usage)
  or site names.  Site positions are preferable for Casbot02 because its
  ``left_foot``/``right_foot`` sites lie at the collision-sole centers.
  """
  asset = _get_robot(env, asset_cfg)
  root_quat = asset.data.root_link_quat_w
  if asset_cfg.site_names is not None:
    foot_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
  else:
    foot_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
  if foot_pos_w.shape[1] != 2:
    raise ValueError(
      "feet_distance_lateral requires exactly two foot bodies or sites, "
      f"got {foot_pos_w.shape[1]}"
    )

  left_right_delta_w = foot_pos_w[:, 0] - foot_pos_w[:, 1]
  left_right_delta_b = quat_apply_inverse(root_quat, left_right_delta_w)
  lateral = torch.abs(left_right_delta_b[:, 1])
  too_close = torch.clamp(lateral - min_distance, max=0.0)
  too_far = torch.clamp(-lateral + max_distance, max=0.0)
  return too_close + too_far


def knee_distance_lateral(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  min_distance: float,
  max_distance: float,
) -> torch.Tensor:
  asset = _get_robot(env, asset_cfg)
  root_pos = asset.data.root_link_pos_w
  root_quat = asset.data.root_link_quat_w
  body_pos = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
  delta = body_pos - root_pos.unsqueeze(1)
  body_pos_b = quat_apply_inverse(
    root_quat.unsqueeze(1).expand(-1, len(asset_cfg.body_ids), -1).reshape(-1, 4),
    delta.reshape(-1, 3),
  ).reshape(env.num_envs, len(asset_cfg.body_ids), 3)
  lateral = torch.abs(body_pos_b[:, 0, 1] - body_pos_b[:, 2, 1]) + torch.abs(
    body_pos_b[:, 1, 1] - body_pos_b[:, 3, 1]
  )
  too_close = torch.clamp(lateral - 2.0 * min_distance, max=0.0)
  too_far = torch.clamp(-lateral + 2.0 * max_distance, max=0.0)
  return too_close + too_far


class straight_non_sagittal_target_deviation_l2:
  """Penalize straight-walking roll/yaw position-target offsets.

  A position-policy action is a virtual PD equilibrium, not the measured joint
  position.  Consequently the posture reward can remain high while a large,
  nearly constant target offset produces substantial internal torque against a
  planted foot.  This term operates on ``action * scale`` (radians), so it
  directly penalizes the target displacement from the action term's default
  offset.  It is active for translational commands and smoothly fades out as
  yaw-rate demand rises, preserving the policy's in-place turning authority.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._action_term_name = cfg.params.get("action_term_name", "joint_pos")
    action_term = env.action_manager.get_term(self._action_term_name)
    joint_names = tuple(cfg.params["joint_names"])
    missing = [name for name in joint_names if name not in action_term.target_names]
    if missing:
      raise ValueError(
        "Target-deviation joints are not controlled by action term "
        f"{self._action_term_name!r}: {missing}"
      )
    self._action_ids = torch.tensor(
      [action_term.target_names.index(name) for name in joint_names],
      device=env.device,
      dtype=torch.long,
    )
    joint_weights = tuple(float(v) for v in cfg.params["joint_weights"])
    if len(joint_weights) != len(joint_names):
      raise ValueError(
        "joint_weights must have one entry per joint: "
        f"got {len(joint_weights)} weights for {len(joint_names)} joints"
      )
    self._joint_weights = torch.tensor(
      joint_weights, device=env.device, dtype=torch.float32
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    joint_names: tuple[str, ...],
    joint_weights: tuple[float, ...],
    command_name: str = "twist",
    action_term_name: str = "joint_pos",
    linear_command_threshold: float = 0.2,
    yaw_relax_start: float = 0.15,
    yaw_relax_end: float = 0.4,
  ) -> torch.Tensor:
    del joint_names, joint_weights, action_term_name
    if yaw_relax_end <= yaw_relax_start:
      raise ValueError(
        f"yaw_relax_end ({yaw_relax_end}) must exceed "
        f"yaw_relax_start ({yaw_relax_start})"
      )

    action_term = env.action_manager.get_term(self._action_term_name)
    raw_action = action_term.raw_action[:, self._action_ids]
    scale = action_term.scale
    if isinstance(scale, torch.Tensor):
      if scale.ndim == 2:
        selected_scale = scale[:, self._action_ids]
      else:
        selected_scale = scale[self._action_ids].unsqueeze(0)
      target_offset = raw_action * selected_scale
    else:
      target_offset = raw_action * float(scale)

    cost = torch.sum(
      torch.square(target_offset) * self._joint_weights.unsqueeze(0), dim=1
    )
    command = env.command_manager.get_command(command_name)
    assert command is not None
    translation_active = (
      torch.linalg.vector_norm(command[:, :2], dim=1) > linear_command_threshold
    ).float()
    yaw_rate = torch.abs(command[:, 2])
    turn_blend = torch.clamp(
      (yaw_rate - yaw_relax_start) / (yaw_relax_end - yaw_relax_start),
      min=0.0,
      max=1.0,
    )
    straight_weight = translation_active * (1.0 - turn_blend)

    active_count = torch.clamp(torch.sum(straight_weight), min=1.0)
    env.extras["log"]["Metrics/non_sagittal_target_offset_rms"] = torch.sqrt(
      torch.sum(
        torch.mean(torch.square(target_offset), dim=1) * straight_weight
      )
      / active_count
    )
    return cost * straight_weight


class dof_torque_limits:
  """HANDOFF normalized actuator-force-over-limit penalty."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
    asset = _get_robot(env, self._asset_cfg)
    actuator_ids = asset.find_actuators((".*",), preserve_order=True)[0]
    self._actuator_ids = torch.tensor(
      actuator_ids, device=env.device, dtype=torch.long
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    soft_torque_limit: float = 0.95,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del asset_cfg
    asset = _get_robot(env, self._asset_cfg)
    actuator_force = torch.abs(asset.data.actuator_force[:, self._actuator_ids])
    force_range = env.sim.model.actuator_forcerange
    if force_range.ndim == 2:
      max_force = force_range[self._actuator_ids, 1].unsqueeze(0)
    else:
      max_force = force_range[:, self._actuator_ids, 1]
    max_force = torch.clamp(max_force, min=1.0e-6)
    over_limit = torch.clamp(actuator_force / max_force - soft_torque_limit, min=0.0)
    return torch.sum(over_limit, dim=1)


class ankle_dof_acc:
  """HANDOFF ankle-only acceleration penalty."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
    asset = _get_robot(env, self._asset_cfg)
    joint_ids = asset.find_joints(_ANKLE_JOINT_NAMES, preserve_order=True)[0]
    self._joint_ids = torch.tensor(joint_ids, device=env.device, dtype=torch.long)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del asset_cfg
    asset = _get_robot(env, self._asset_cfg)
    return torch.sum(torch.square(asset.data.joint_acc[:, self._joint_ids]), dim=1)


class ankle_dof_vel:
  """HANDOFF ankle-only velocity penalty."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self._asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", _DEFAULT_ASSET_CFG)
    asset = _get_robot(env, self._asset_cfg)
    joint_ids = asset.find_joints(_ANKLE_JOINT_NAMES, preserve_order=True)[0]
    self._joint_ids = torch.tensor(joint_ids, device=env.device, dtype=torch.long)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    del asset_cfg
    asset = _get_robot(env, self._asset_cfg)
    return torch.sum(torch.square(asset.data.joint_vel[:, self._joint_ids]), dim=1)


class feet_air_time:
  """HANDOFF landing-time reward gated by reference motion speed."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    command_name: str = "motion",
    feet_air_time_target: float = 0.5,
    uniform_cmd_name: str | None = None,
  ) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    last_air_time = sensor.data.last_air_time
    assert last_air_time is not None
    first_contact = sensor.compute_first_contact(dt=self.step_dt).float()
    air_time = torch.clamp(last_air_time - feet_air_time_target, max=0.0)
    reward = torch.sum(air_time * first_contact, dim=1)
    if uniform_cmd_name is not None:
      uni_xy = env_mdp.generated_commands(env, command_name=uniform_cmd_name)[:, :2]
      active = torch.norm(uni_xy, dim=1) > 0.05
    else:
      command = get_motion_command(env, command_name)
      active = torch.norm(command.body_lin_vel_w[:, 0, :2], dim=1) > 0.05
    return reward * active.float()

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    del env_ids


def motion_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold_min: float = 0.05,
  threshold_max: float = 0.5,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  cmd_limits: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
  """Reward feet air time with gating from motion command magnitude."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
  reward = torch.sum(in_range.float(), dim=1)

  in_air = current_air_time > 0
  num_in_air = torch.sum(in_air.float())
  mean_air_time = torch.sum(current_air_time * in_air.float()) / torch.clamp(
    num_in_air, min=1
  )
  env.extras["log"]["Metrics/air_time_mean"] = mean_air_time

  if command_name is not None:
    reward = reward * _motion_command_is_active(
      env,
      command_name=command_name,
      command_threshold=command_threshold,
      cmd_limits=cmd_limits,
    )
  return reward


def motion_feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  height_sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  cmd_limits: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
  """Penalize swing-foot clearance error, gated by motion command magnitude."""
  asset: Entity = env.scene[asset_cfg.name]
  height_sensor = env.scene[height_sensor_name]
  assert isinstance(height_sensor, TerrainHeightSensor), (
    "motion_feet_clearance requires a TerrainHeightSensor, "
    f"got {type(height_sensor).__name__}"
  )
  foot_height = height_sensor.data.heights
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)
  delta = torch.abs(foot_height - target_height)
  cost = torch.sum(delta * vel_norm, dim=1)

  if command_name is not None:
    cost = cost * _motion_command_is_active(
      env,
      command_name=command_name,
      command_threshold=command_threshold,
      cmd_limits=cmd_limits,
    )
  return cost


class motion_feet_swing_height:
  """Penalize landing swing-height error gated by motion command magnitude."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    height_sensor = env.scene[cfg.params["height_sensor_name"]]
    assert isinstance(height_sensor, TerrainHeightSensor), (
      "motion_feet_swing_height requires a TerrainHeightSensor, "
      f"got {type(height_sensor).__name__}"
    )
    num_feet = height_sensor.num_frames
    self.peak_heights = torch.zeros(
      (env.num_envs, num_feet), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    height_sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
    cmd_limits: tuple[float, float, float] = (1.0, 1.0, 1.0),
  ) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[sensor_name]
    assert contact_sensor.data.found is not None
    height_sensor: TerrainHeightSensor = env.scene[height_sensor_name]
    foot_heights = height_sensor.data.heights

    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)

    active = _motion_command_is_active(
      env,
      command_name=command_name,
      command_threshold=command_threshold,
      cmd_limits=cmd_limits,
    )
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active

    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height

    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    del env_ids


def motion_soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  cmd_limits: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
  """Penalize landing impacts, gated by motion command magnitude."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force
  force_magnitude = torch.norm(forces, dim=-1)
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)
  landing_impact = force_magnitude * first_contact.float()
  cost = torch.sum(landing_impact, dim=1)

  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force

  if command_name is not None:
    cost = cost * _motion_command_is_active(
      env,
      command_name=command_name,
      command_threshold=command_threshold,
      cmd_limits=cmd_limits,
    )
  return cost


# ---------------------------------------------------------------------------
# CBF reward (additive — attached only in the CBF task variant)
# ---------------------------------------------------------------------------














