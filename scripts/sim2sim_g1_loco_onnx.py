"""G1 15-DoF loco teacher sim2sim (ONNX policy, keyboard command).

NoBV actor obs is a single 86-vector (no history):

  ang_vel(3)*0.25 + gravity(3) + twist(3) + qpos_rel(29) + qvel(29)*0.05
  + last_action(15) + gait phase(4)

Actions are the 15 body joints. Unused arm joints stay at the knees-bent
default; shoulder pitch is remapped each physics step from the measured
knee difference (same formula as training).

Keyboard: Up/Down=vx  Left/Right=yaw  A/D=vy  Space=stop  R=reset.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import time
from pathlib import Path

import mujoco
import numpy as np

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  G1_ACTION_SCALE,
  KNEES_BENT_KEYFRAME,
  get_g1_robot_cfg,
)
from mjlab.entity import Entity
from mjlab.utils.string import resolve_expr

from wbc_mjlab.actions import (
  ARM_SWING_GAIN,
  G1_LEFT_KNEE_NAME,
  G1_LEFT_SHOULDER_PITCH_NAME,
  G1_RIGHT_KNEE_NAME,
  G1_RIGHT_SHOULDER_PITCH_NAME,
)
from wbc_mjlab.observations import (
  ARM_JOINT_NAMES,
  BODY_JOINT_NAMES,
  LOCO_BASE_ANG_VEL_SCALE,
  LOCO_GAIT_OFFSET,
  LOCO_GAIT_PERIOD,
  LOCO_JOINT_POS_SCALE,
  LOCO_JOINT_VEL_SCALE,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

SIM_TIMESTEP = 0.005
POLICY_DECIMATION = 4
POLICY_DT = SIM_TIMESTEP * POLICY_DECIMATION
NUM_POLICY_ACTIONS = len(BODY_JOINT_NAMES)  # 15
G1_JOINT_NAMES: tuple[str, ...] = BODY_JOINT_NAMES + ARM_JOINT_NAMES  # 29
NUM_FULL_JOINTS = len(G1_JOINT_NAMES)
ACTOR_OBS_DIM = 86
STANDING_CMD_NORM = 0.05

DEFAULT_ONNX = ""
DEFAULT_COMMAND = (0.5, 0.0, 0.0)
DEFAULT_VIEW_SPEED = 1.0
DEFAULT_REALTIME = True
DEFAULT_ATTACH_PAYLOADS = True
# Render every N physics steps. Physics still runs at SIM_TIMESTEP (200 Hz);
# only the viewer redraw is throttled. 200 Hz rendering is the usual reason a
# machine falls behind real-time and plays in slow motion — dropping to 50 Hz
# keeps the loop under one timestep on slower GPUs/drivers.
DEFAULT_RENDER_DECIMATION = 4
PRINT_EVERY = 1.0
DEFAULT_LOG_CSV = ""
DEFAULT_LOG_DECIMATION = 1
COMMAND_X_RANGE = (-1.0, 1.0)
COMMAND_Y_RANGE = (-1.0, 1.0)
COMMAND_YAW_RANGE = (-1.0, 1.0)
COMMAND_X_STEP = 0.1
COMMAND_Y_STEP = 0.1
COMMAND_YAW_STEP = 0.1


def find_latest_onnx() -> Path:
  export_root = REPO_ROOT / "logs" / "rsl_rl" / "g1_loco_teacher_nobv_stable_seed"
  if not export_root.is_dir():
    raise FileNotFoundError(f"No G1 loco training output under {export_root}")
  fallback: Path | None = None
  for run_dir in sorted(export_root.iterdir(), reverse=True):
    if not run_dir.is_dir():
      continue
    onnx_files = sorted(run_dir.glob("*.onnx"))
    if not onnx_files:
      continue
    if fallback is None:
      fallback = onnx_files[-1]
    ckpt_iters = []
    for checkpoint in run_dir.glob("model_*.pt"):
      try:
        ckpt_iters.append(int(checkpoint.stem.removeprefix("model_")))
      except ValueError:
        continue
    if ckpt_iters and max(ckpt_iters) > 0:
      return onnx_files[-1]
  if fallback is not None:
    print("[ONNX] warning: only iteration-0 exports were found")
    return fallback
  raise FileNotFoundError(f"No G1 loco ONNX under {export_root}")


def resolve_default_joint_pos() -> np.ndarray:
  joint_pos = KNEES_BENT_KEYFRAME.joint_pos or {}
  return np.asarray(
    resolve_expr(joint_pos, G1_JOINT_NAMES, 0.0),
    dtype=np.float64,
  )


def resolve_action_scale() -> np.ndarray:
  values = np.ones(NUM_POLICY_ACTIONS, dtype=np.float64)
  for pattern, value in G1_ACTION_SCALE.items():
    for idx, name in enumerate(BODY_JOINT_NAMES):
      if re.fullmatch(pattern, name):
        values[idx] = float(value)
  return values


def resolve_control_order(model: mujoco.MjModel) -> np.ndarray:
  """Return the joint index driven by each entry of ``model.ctrl``.

  G1 actuators are grouped by actuator configuration, so their control order
  differs from the kinematic joint order used by qpos, qvel and policy targets.
  """
  actuator_joint_names: list[str] = []
  for actuator_id in range(model.nu):
    transmission = int(model.actuator_trntype[actuator_id])
    if transmission != int(mujoco.mjtTrn.mjTRN_JOINT):
      raise RuntimeError(
        f"Actuator {actuator_id} does not use a joint transmission: {transmission}"
      )
    joint_id = int(model.actuator_trnid[actuator_id, 0])
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    if joint_name is None or joint_name not in G1_JOINT_NAMES:
      raise RuntimeError(
        f"Actuator {actuator_id} targets unexpected joint {joint_name!r}"
      )
    actuator_joint_names.append(joint_name)

  if len(set(actuator_joint_names)) != NUM_FULL_JOINTS:
    raise RuntimeError(
      "Expected exactly one actuator per G1 joint, got targets "
      f"{actuator_joint_names}"
    )
  return np.asarray(
    [G1_JOINT_NAMES.index(name) for name in actuator_joint_names],
    dtype=np.int64,
  )


def gait_phase_features(
  episode_time_s: float,
  command: np.ndarray,
  *,
  gait_period: float = LOCO_GAIT_PERIOD,
  gait_offset: float = LOCO_GAIT_OFFSET,
  standing_norm: float = STANDING_CMD_NORM,
) -> np.ndarray:
  """sin/cos of left/right leg phase. Standing command zeros the clock."""
  if float(np.linalg.norm(command)) < standing_norm:
    return np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
  walk = (episode_time_s % gait_period) / gait_period
  right = (walk + gait_offset) % 1.0
  two_pi = 2.0 * np.pi
  return np.array(
    [
      np.sin(two_pi * walk),
      np.cos(two_pi * walk),
      np.sin(two_pi * right),
      np.cos(two_pi * right),
    ],
    dtype=np.float32,
  )


def quat_apply_inverse_wxyz(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
  w, x, y, z = quat
  q_vec = -np.array([x, y, z], dtype=np.float64)
  t = 2.0 * np.cross(q_vec, vec)
  return vec + w * t + np.cross(q_vec, t)


def resolve_csv_path(log_csv: str, model_path: Path) -> Path | None:
  if not log_csv:
    return None
  if log_csv.lower() == "auto":
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "logs" / "sim2sim_csv" / f"{model_path.stem}_{timestamp}.csv"
  path = Path(log_csv).expanduser()
  return path if path.is_absolute() else REPO_ROOT / path


def quat_to_euler_xyz_wxyz(quat: np.ndarray) -> tuple[float, float, float]:
  w, x, y, z = quat
  roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
  pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
  yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
  return float(roll), float(pitch), float(yaw)


class PlotJugglerCsvLogger:
  """Stream sim2sim state and actuator data to a generic PlotJuggler CSV."""

  def __init__(
    self,
    path: Path,
    *,
    left_foot_id: int,
    right_foot_id: int,
    joint_to_ctrl: np.ndarray,
    flush_every: int = 200,
  ) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    self.path = path
    self._left_foot_id = left_foot_id
    self._right_foot_id = right_foot_id
    self._joint_to_ctrl = joint_to_ctrl.copy()
    self._file = path.open("w", newline="", buffering=1024 * 1024)
    self._writer = csv.writer(self._file)
    self._flush_every = max(int(flush_every), 1)
    self._rows_written = 0

    header = [
      "time",
      "sim/step",
      "policy/update",
      "command/vx_m_s",
      "command/vy_m_s",
      "command/yaw_rate_rad_s",
      "base/position/x_m",
      "base/position/y_m",
      "base/position/z_m",
      "base/orientation/qw",
      "base/orientation/qx",
      "base/orientation/qy",
      "base/orientation/qz",
      "base/orientation/roll_rad",
      "base/orientation/pitch_rad",
      "base/orientation/yaw_rad",
      "base/free_joint_linear_velocity/x_m_s",
      "base/free_joint_linear_velocity/y_m_s",
      "base/free_joint_linear_velocity/z_m_s",
      "base/free_joint_angular_velocity/x_rad_s",
      "base/free_joint_angular_velocity/y_rad_s",
      "base/free_joint_angular_velocity/z_rad_s",
      "com/position_world/x_m",
      "com/position_world/y_m",
      "com/position_world/z_m",
      "com/relative_to_base_world_axes/x_m",
      "com/relative_to_base_world_axes/y_m",
      "com/relative_to_base_world_axes/z_m",
    ]
    for side in ("left", "right"):
      header.extend(f"foot/{side}/position_world/{axis}_m" for axis in ("x", "y", "z"))
      header.extend(
        f"foot/{side}/relative_to_base_world_axes/{axis}_m" for axis in ("x", "y", "z")
      )
    header.extend(f"policy/action/{name}" for name in BODY_JOINT_NAMES)
    for signal, unit in (
      ("target_position", "rad"),
      ("position", "rad"),
      ("velocity", "rad_s"),
      ("position_error", "rad"),
      ("actuator_force", "Nm"),
      ("generalized_actuator_force", "Nm"),
    ):
      header.extend(f"joint/{signal}/{name}_{unit}" for name in G1_JOINT_NAMES)
    self._writer.writerow(header)

  def write(
    self,
    *,
    step: int,
    policy_updated: bool,
    data: mujoco.MjData,
    command: np.ndarray,
    action: np.ndarray,
    target_pos: np.ndarray,
  ) -> None:
    qpos = np.asarray(data.qpos[7:], dtype=np.float64)
    qvel = np.asarray(data.qvel[6:], dtype=np.float64)
    actuator_force = np.asarray(data.actuator_force, dtype=np.float64)[
      self._joint_to_ctrl
    ]
    generalized_force = np.asarray(data.qfrc_actuator[6:], dtype=np.float64)
    base_pos = np.asarray(data.qpos[0:3], dtype=np.float64)
    base_quat = np.asarray(data.qpos[3:7], dtype=np.float64)
    roll, pitch, yaw = quat_to_euler_xyz_wxyz(base_quat)
    com_world = np.asarray(data.subtree_com[0], dtype=np.float64)
    left_foot = np.asarray(data.xpos[self._left_foot_id], dtype=np.float64)
    right_foot = np.asarray(data.xpos[self._right_foot_id], dtype=np.float64)

    row: list[float | int] = [
      float(data.time),
      step,
      int(policy_updated),
      *command.tolist(),
      *base_pos.tolist(),
      *base_quat.tolist(),
      roll,
      pitch,
      yaw,
      *np.asarray(data.qvel[0:3], dtype=np.float64).tolist(),
      *np.asarray(data.qvel[3:6], dtype=np.float64).tolist(),
      *com_world.tolist(),
      *(com_world - base_pos).tolist(),
      *left_foot.tolist(),
      *(left_foot - base_pos).tolist(),
      *right_foot.tolist(),
      *(right_foot - base_pos).tolist(),
      *np.asarray(action, dtype=np.float64).tolist(),
      *np.asarray(target_pos, dtype=np.float64).tolist(),
      *qpos.tolist(),
      *qvel.tolist(),
      *(target_pos - qpos).tolist(),
      *actuator_force.tolist(),
      *generalized_force.tolist(),
    ]
    self._writer.writerow(row)
    self._rows_written += 1
    if self._rows_written % self._flush_every == 0:
      self._file.flush()

  def close(self) -> None:
    if not self._file.closed:
      self._file.flush()
      self._file.close()


class OnnxPolicy:
  def __init__(self, model_path: str | Path):
    import onnxruntime as ort

    self.session = ort.InferenceSession(
      str(model_path), providers=["CPUExecutionProvider"]
    )
    self.input_name = self.session.get_inputs()[0].name
    self.output_name = self.session.get_outputs()[0].name
    self.input_dim = self.session.get_inputs()[0].shape[1]
    self.output_dim = self.session.get_outputs()[0].shape[1]
    print(f"[ONNX] input={self.input_dim}, output={self.output_dim}")

  def __call__(self, obs: np.ndarray) -> np.ndarray:
    obs = np.ascontiguousarray(obs.astype(np.float32))
    return np.asarray(
      self.session.run([self.output_name], {self.input_name: obs})[0],
      dtype=np.float32,
    ).reshape(-1)


def _robot_cfg(*, attach_payloads: bool):
  from wbc_mjlab import g1_constants_custom as g1c

  cfg = get_g1_robot_cfg()
  if attach_payloads:
    g1c.ATTACH_PAYLOADS = True
    cfg.spec_fn = g1c.wrap_spec_fn_with_payloads(cfg.spec_fn)
  return cfg


def build_model(*, attach_payloads: bool = DEFAULT_ATTACH_PAYLOADS) -> mujoco.MjModel:
  entity = Entity(_robot_cfg(attach_payloads=attach_payloads))
  spec = entity.spec
  ground = spec.worldbody.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE)
  ground.size = [0.0, 0.0, 1.0]
  ground.condim = 3
  ground.friction = [0.6, 0.005, 0.0001]
  model = spec.compile()
  model.opt.timestep = SIM_TIMESTEP
  return model


def get_obs(
  data: mujoco.MjData,
  *,
  pelvis_id: int,
  command: np.ndarray,
  last_action: np.ndarray,
  default_joint_pos: np.ndarray,
  episode_time_s: float,
) -> np.ndarray:
  quat = np.asarray(data.xquat[pelvis_id], dtype=np.float64)
  ang_vel_w = np.asarray(data.cvel[pelvis_id, :3], dtype=np.float64)
  ang_vel_b = quat_apply_inverse_wxyz(quat, ang_vel_w)
  projected_gravity = quat_apply_inverse_wxyz(
    quat, np.array([0.0, 0.0, -1.0], dtype=np.float64)
  )
  joint_pos_rel = np.asarray(data.qpos[7:], dtype=np.float64) - default_joint_pos
  joint_vel_rel = np.asarray(data.qvel[6:], dtype=np.float64)
  phase = gait_phase_features(episode_time_s, command)
  return np.concatenate(
    [
      ang_vel_b * LOCO_BASE_ANG_VEL_SCALE,
      projected_gravity,
      command,
      joint_pos_rel * LOCO_JOINT_POS_SCALE,
      joint_vel_rel * LOCO_JOINT_VEL_SCALE,
      last_action,
      phase,
    ]
  ).astype(np.float32)


def install_command_controls(viewer, command: np.ndarray, on_reset) -> None:
  import glfw

  def clamp(v, lo, hi):
    return float(np.clip(v, lo, hi))

  keys = {
    glfw.KEY_UP,
    glfw.KEY_DOWN,
    glfw.KEY_LEFT,
    glfw.KEY_RIGHT,
    glfw.KEY_SPACE,
    glfw.KEY_A,
    glfw.KEY_D,
    glfw.KEY_R,
  }
  orig = viewer._key_callback

  def cb(window, key, scancode, action, mods):
    if key in keys and action in (glfw.PRESS, glfw.REPEAT):
      if key == glfw.KEY_UP:
        command[0] = clamp(command[0] + COMMAND_X_STEP, *COMMAND_X_RANGE)
      elif key == glfw.KEY_DOWN:
        command[0] = clamp(command[0] - COMMAND_X_STEP, *COMMAND_X_RANGE)
      elif key == glfw.KEY_A:
        command[1] = clamp(command[1] + COMMAND_Y_STEP, *COMMAND_Y_RANGE)
      elif key == glfw.KEY_D:
        command[1] = clamp(command[1] - COMMAND_Y_STEP, *COMMAND_Y_RANGE)
      elif key == glfw.KEY_LEFT:
        command[2] = clamp(command[2] + COMMAND_YAW_STEP, *COMMAND_YAW_RANGE)
      elif key == glfw.KEY_RIGHT:
        command[2] = clamp(command[2] - COMMAND_YAW_STEP, *COMMAND_YAW_RANGE)
      elif key == glfw.KEY_SPACE:
        command[:] = 0.0
      elif key == glfw.KEY_R:
        on_reset()
        print("[sim2sim] reset")
        return
      print(f"[command] vx={command[0]:+.2f} vy={command[1]:+.2f} yaw={command[2]:+.2f}")
      return
    orig(window, key, scancode, action, mods)

  glfw.set_key_callback(viewer.window, cb)

  original_create_overlay = viewer._create_overlay

  def create_overlay_with_command() -> None:
    original_create_overlay()
    gridpos = mujoco.mjtGridPos.mjGRID_TOPRIGHT
    if gridpos not in viewer._overlay:
      viewer._overlay[gridpos] = ["", ""]
    viewer._overlay[gridpos][0] += (
      "Command vx/vy/yaw\n"
      "Up/Down forward\n"
      "A/D lateral\n"
      "Left/Right yaw\n"
      "Space stop\n"
      "R reset\n"
    )
    viewer._overlay[gridpos][1] += (
      f"{command[0]:+.2f}  {command[1]:+.2f}  {command[2]:+.2f}\n"
      f"+/- {COMMAND_X_STEP:.1f} m/s\n"
      f"+/- {COMMAND_Y_STEP:.1f} m/s\n"
      f"+/- {COMMAND_YAW_STEP:.1f} rad/s\n"
      "zero command\n"
      "home pose\n"
    )

  viewer._create_overlay = create_overlay_with_command


def run(
  model_arg: str = "",
  log_csv: str = DEFAULT_LOG_CSV,
  log_decimation: int = DEFAULT_LOG_DECIMATION,
  *,
  attach_payloads: bool = DEFAULT_ATTACH_PAYLOADS,
  view_speed: float = DEFAULT_VIEW_SPEED,
  render_decimation: int = DEFAULT_RENDER_DECIMATION,
) -> None:
  model_path = Path(model_arg) if model_arg else find_latest_onnx()
  if not model_path.is_absolute():
    model_path = REPO_ROOT / model_path
  if not model_path.exists():
    raise FileNotFoundError(model_path)

  policy = OnnxPolicy(model_path)
  if policy.input_dim != ACTOR_OBS_DIM:
    raise RuntimeError(f"Expected {ACTOR_OBS_DIM} obs, got {policy.input_dim}")
  if policy.output_dim != NUM_POLICY_ACTIONS:
    raise RuntimeError(f"Expected {NUM_POLICY_ACTIONS} actions, got {policy.output_dim}")

  if attach_payloads:
    os.environ.setdefault("WBC_ATTACH_PAYLOADS", "1")
  model = build_model(attach_payloads=attach_payloads)
  if model.nu != NUM_FULL_JOINTS or model.nq != 7 + NUM_FULL_JOINTS:
    raise RuntimeError(
      f"Unexpected model sizes nq={model.nq} nu={model.nu} "
      f"(expected nq={7 + NUM_FULL_JOINTS} nu={NUM_FULL_JOINTS})"
    )
  data = mujoco.MjData(model)
  default_joint_pos = resolve_default_joint_pos()
  action_scale = resolve_action_scale()
  ctrl_to_joint = resolve_control_order(model)
  joint_to_ctrl = np.argsort(ctrl_to_joint)
  body_indices = np.array(
    [G1_JOINT_NAMES.index(name) for name in BODY_JOINT_NAMES], dtype=np.int64
  )
  knee_l = G1_JOINT_NAMES.index(G1_LEFT_KNEE_NAME)
  knee_r = G1_JOINT_NAMES.index(G1_RIGHT_KNEE_NAME)
  shoulder_l = G1_JOINT_NAMES.index(G1_LEFT_SHOULDER_PITCH_NAME)
  shoulder_r = G1_JOINT_NAMES.index(G1_RIGHT_SHOULDER_PITCH_NAME)
  pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
  left_foot_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link"
  )
  right_foot_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link"
  )
  ctrl_lo = model.actuator_ctrlrange[:, 0].copy()
  ctrl_hi = model.actuator_ctrlrange[:, 1].copy()
  command = np.array(DEFAULT_COMMAND, dtype=np.float64)

  last_action = np.zeros(NUM_POLICY_ACTIONS, dtype=np.float32)
  target_pos = default_joint_pos.copy()
  policy_steps = 0

  def reset_state() -> None:
    nonlocal last_action, target_pos, policy_steps
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = np.asarray(KNEES_BENT_KEYFRAME.pos, dtype=np.float64)
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:] = default_joint_pos
    data.qvel[:] = 0.0
    last_action = np.zeros(NUM_POLICY_ACTIONS, dtype=np.float32)
    target_pos = default_joint_pos.copy()
    policy_steps = 0
    mujoco.mj_forward(model, data)

  reset_state()

  import mujoco_viewer

  viewer = mujoco_viewer.MujocoViewer(model, data, mode="window")
  viewer.cam.distance = 3.0
  viewer.cam.azimuth = 90.0
  viewer.cam.elevation = -5.0
  install_command_controls(viewer, command, reset_state)

  print("[sim2sim] Up/Down=vx  A/D=vy  Left/Right=yaw  Space=stop  R=reset")
  print(f"[sim2sim] payloads={'on' if attach_payloads else 'off'}")
  next_time = time.perf_counter()
  wall_start = next_time
  sim_time = 0.0
  physics_step = 0
  next_print_time = 0.0
  if log_decimation < 1:
    raise ValueError(f"log_decimation must be >= 1, got {log_decimation}")
  if render_decimation < 1:
    raise ValueError(f"render_decimation must be >= 1, got {render_decimation}")
  view_speed = max(float(view_speed), 1e-6)
  print(
    f"[sim2sim] dt={model.opt.timestep}, decimation={POLICY_DECIMATION}, "
    f"policy_hz={1.0 / POLICY_DT:.1f}, render_decimation={render_decimation}, "
    f"render_hz={1.0 / (model.opt.timestep * render_decimation):.1f}, "
    f"view_speed={view_speed}"
  )
  csv_path = resolve_csv_path(log_csv, model_path)
  csv_logger = (
    PlotJugglerCsvLogger(
      csv_path,
      left_foot_id=left_foot_id,
      right_foot_id=right_foot_id,
      joint_to_ctrl=joint_to_ctrl,
    )
    if csv_path is not None
    else None
  )
  if csv_logger is not None:
    print(
      "[sim2sim] recording PlotJuggler CSV at "
      f"{1.0 / (model.opt.timestep * log_decimation):.1f} Hz: {csv_path}"
    )
  try:
    while viewer.is_alive:
      policy_updated = physics_step % POLICY_DECIMATION == 0
      if policy_updated:
        obs = get_obs(
          data,
          pelvis_id=pelvis_id,
          command=command,
          last_action=last_action,
          default_joint_pos=default_joint_pos,
          episode_time_s=policy_steps * POLICY_DT,
        ).reshape(1, -1)
        action = policy(obs)
        last_action = action.astype(np.float32)
        target_pos = default_joint_pos.copy()
        target_pos[body_indices] = (
          default_joint_pos[body_indices] + action * action_scale
        )
        policy_steps += 1

      knee_diff = data.qpos[7 + knee_l] - data.qpos[7 + knee_r]
      target_pos[shoulder_l] = (
        ARM_SWING_GAIN * knee_diff + default_joint_pos[shoulder_l]
      )
      target_pos[shoulder_r] = (
        -ARM_SWING_GAIN * knee_diff + default_joint_pos[shoulder_r]
      )
      # Policy targets use kinematic joint order, while MuJoCo ctrl follows
      # actuator declaration order. Reorder before applying actuator limits.
      ctrl_target = np.clip(target_pos[ctrl_to_joint], ctrl_lo, ctrl_hi)
      target_pos = ctrl_target[joint_to_ctrl]
      data.ctrl[:] = ctrl_target
      mujoco.mj_step(model, data)
      if csv_logger is not None and physics_step % log_decimation == 0:
        csv_logger.write(
          step=physics_step,
          policy_updated=policy_updated,
          data=data,
          command=command,
          action=last_action,
          target_pos=target_pos,
        )
      physics_step += 1
      if physics_step % render_decimation == 0:
        viewer.render()
      viewer.cam.lookat = [
        float(data.qpos[0]),
        float(data.qpos[1]),
        float(data.qpos[2]),
      ]
      sim_time += model.opt.timestep

      if sim_time >= next_print_time:
        next_print_time += PRINT_EVERY
        wall_elapsed = time.perf_counter() - wall_start
        rtf = sim_time / wall_elapsed if wall_elapsed > 0 else float("nan")
        print(
          f"t={sim_time:7.2f}s base_z={data.qpos[2]:.3f} "
          f"command=({command[0]:+.2f},{command[1]:+.2f},{command[2]:+.2f}) "
          f"rtf={rtf:.2f}"
        )

      if DEFAULT_REALTIME:
        next_time += model.opt.timestep / view_speed
        sleep_s = next_time - time.perf_counter()
        if sleep_s > 0:
          time.sleep(sleep_s)
        else:
          next_time = time.perf_counter()
  finally:
    if csv_logger is not None:
      csv_logger.close()
      print(f"[sim2sim] PlotJuggler CSV saved: {csv_logger.path}")
    viewer.close()


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description="G1 loco teacher ONNX sim2sim")
  p.add_argument("model_path", nargs="?", default="", help="ONNX path (empty = latest)")
  p.add_argument(
    "--log-csv",
    nargs="?",
    const="auto",
    default=DEFAULT_LOG_CSV,
    metavar="PATH",
    help=(
      "Record a PlotJuggler-compatible CSV. Pass no PATH to save automatically "
      "under logs/sim2sim_csv."
    ),
  )
  p.add_argument(
    "--log-decimation",
    type=int,
    default=DEFAULT_LOG_DECIMATION,
    help="Record every N physics steps (default: 1 = 200 Hz).",
  )
  p.add_argument(
    "--no-payloads",
    action="store_true",
    help="Skip Jetson + Dex1-1 payloads (training default is on).",
  )
  p.add_argument(
    "--view-speed",
    type=float,
    default=DEFAULT_VIEW_SPEED,
    help="Playback speed multiplier (1.0 = real-time). >1 plays faster.",
  )
  p.add_argument(
    "--render-decimation",
    type=int,
    default=DEFAULT_RENDER_DECIMATION,
    help=(
      "Render every N physics steps (default 4 = 50 Hz). Increase if the "
      "machine falls behind real-time and plays in slow motion."
    ),
  )
  return p.parse_args()


if __name__ == "__main__":
  args = parse_args()
  run(
    args.model_path,
    args.log_csv,
    args.log_decimation,
    attach_payloads=not args.no_payloads,
    view_speed=args.view_speed,
    render_decimation=args.render_decimation,
  )
