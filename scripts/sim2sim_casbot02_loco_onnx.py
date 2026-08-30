"""Casbot02 12-leg loco teacher sim2sim (ONNX policy, keyboard command).

Same structure as amp_mjlab's ``sim2sim_casbot02_leg_amp_onnx.py``, but:
  * loads the loco teacher ONNX (12 action, 180 obs);
  * builds the observation TERM-major (mjlab 1.4.0 layout: each term's 4-frame
    history contiguous), NOT time-major like the amp (mjlab 1.2.0) script.

Keyboard: Up/Down = vx, Left/Right = yaw-rate, Space = stop.
"""
from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

from mjlab.entity import Entity

from wbc_mjlab.casbot02_constants import (
  CASBOT02_23DOF_JOINT_NAMES,
  CASBOT02_LEG_ONLY_ACTION_SCALE,
  CASBOT02_LEG_ONLY_JOINT_NAMES,
  HOME_KEYFRAME,
  get_casbot02_23dof_robot_cfg,
)
from wbc_mjlab.casbot02_actions import (
  ARM_SWING_GAIN,
  LEFT_KNEE_NAME,
  LEFT_SHOULDER_NAME,
  RIGHT_KNEE_NAME,
  RIGHT_SHOULDER_NAME,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

SIM_TIMESTEP = 0.005
POLICY_DECIMATION = 4
HISTORY_LENGTH = 4
NUM_OBS_JOINTS = len(CASBOT02_LEG_ONLY_JOINT_NAMES)  # 12
NUM_POLICY_ACTIONS = len(CASBOT02_LEG_ONLY_JOINT_NAMES)  # 12
NUM_FULL_JOINTS = len(CASBOT02_23DOF_JOINT_NAMES)  # 22 (legs + arms)
OBS_JOINT_INDICES = np.array(
  [CASBOT02_23DOF_JOINT_NAMES.index(n) for n in CASBOT02_LEG_ONLY_JOINT_NAMES],
  dtype=np.int64,
)
ACTION_JOINT_INDICES = OBS_JOINT_INDICES
LEG_KNEE_L = CASBOT02_23DOF_JOINT_NAMES.index(LEFT_KNEE_NAME)
LEG_KNEE_R = CASBOT02_23DOF_JOINT_NAMES.index(RIGHT_KNEE_NAME)
ARM_SHOULDER_L = CASBOT02_23DOF_JOINT_NAMES.index(LEFT_SHOULDER_NAME)
ARM_SHOULDER_R = CASBOT02_23DOF_JOINT_NAMES.index(RIGHT_SHOULDER_NAME)
SINGLE_FRAME_OBS_SIZE = 45  # 3+3+3+12+12+12
TERM_DIMS = (3, 3, 3, 12, 12, 12)

DEFAULT_ONNX = ""
DEFAULT_COMMAND = (0.5, 0.0, 0.0)  # 默认给个前进速度, 打开就能看到走
DEFAULT_VIEW_SPEED = 1.0
DEFAULT_REALTIME = True
PRINT_EVERY = 1.0  # 终端打印关节位置的时间间隔(秒)
COMMAND_X_RANGE = (-3.5, 5.0)
COMMAND_Y_RANGE = (-1.0, 1.0)
COMMAND_YAW_RANGE = (-3.14, 3.14)
COMMAND_X_STEP = 0.1
COMMAND_Y_STEP = 0.1
COMMAND_YAW_STEP = 0.1


def find_latest_onnx() -> Path:
  export_root = REPO_ROOT / "logs" / "rsl_rl" / "casbot02_loco_teacher"
  for run_dir in sorted(export_root.iterdir(), reverse=True):
    onnx_files = list(run_dir.glob("*.onnx"))
    if onnx_files:
      return sorted(onnx_files)[-1]
  raise FileNotFoundError(f"No loco teacher ONNX under {export_root}")


def make_default_joint_pos() -> np.ndarray:
  joint_pos = HOME_KEYFRAME.joint_pos or {}
  fallback = float(joint_pos.get(".*", 0.0))
  return np.array(
    [float(joint_pos.get(n, fallback)) for n in CASBOT02_23DOF_JOINT_NAMES],
    dtype=np.float64,
  )


def make_action_scale() -> np.ndarray:
  return np.array(
    [CASBOT02_LEG_ONLY_ACTION_SCALE[n] for n in CASBOT02_LEG_ONLY_JOINT_NAMES],
    dtype=np.float64,
  )


def quat_apply_inverse_wxyz(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
  w, x, y, z = quat
  q_vec = -np.array([x, y, z], dtype=np.float64)
  t = 2.0 * np.cross(q_vec, vec)
  return vec + w * t + np.cross(q_vec, t)


class OnnxPolicy:
  def __init__(self, model_path: str | Path):
    import onnxruntime as ort
    self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
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


def build_model() -> mujoco.MjModel:
  entity = Entity(get_casbot02_23dof_robot_cfg())
  spec = entity.spec
  ground = spec.worldbody.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE)
  ground.size = [0.0, 0.0, 1.0]
  ground.condim = 3
  ground.friction = [1.0, 0.005, 0.0001]
  ground.material = "matplane"
  model = spec.compile()
  model.opt.timestep = SIM_TIMESTEP
  return model


def get_obs_frame(
  data: mujoco.MjData,
  command: np.ndarray,
  last_action: np.ndarray,
  default_obs_joint_pos: np.ndarray,
) -> np.ndarray:
  base_ang_vel = np.asarray(data.sensor("angular-velocity").data, dtype=np.float64).copy()
  quat_wxyz = np.asarray(data.qpos[3:7], dtype=np.float64)
  projected_gravity = quat_apply_inverse_wxyz(quat_wxyz, np.array([0.0, 0.0, -1.0]))
  joint_pos_rel = np.asarray(data.qpos[7:], dtype=np.float64)[OBS_JOINT_INDICES] - default_obs_joint_pos
  joint_vel_rel = np.asarray(data.qvel[6:], dtype=np.float64)[OBS_JOINT_INDICES]
  return np.concatenate(
    [base_ang_vel, projected_gravity, command, joint_pos_rel, joint_vel_rel, last_action]
  ).astype(np.float32)


def build_obs_term_major(history: deque[np.ndarray]) -> np.ndarray:
  """Concatenate the 4 single frames TERM-major (mjlab 1.4.0 layout)."""
  frames = np.array(list(history))  # (4, 45), oldest -> newest
  parts = []
  offset = 0
  for d in TERM_DIMS:
    parts.append(frames[:, offset:offset + d].reshape(-1))
    offset += d
  return np.concatenate(parts).reshape(1, -1)


def install_command_controls(viewer, command: np.ndarray) -> None:
  import glfw

  def clamp(v, lo, hi):
    return float(np.clip(v, lo, hi))

  keys = {glfw.KEY_UP, glfw.KEY_DOWN, glfw.KEY_LEFT, glfw.KEY_RIGHT, glfw.KEY_SPACE}
  orig = viewer._key_callback

  def cb(window, key, scancode, action, mods):
    if key in keys and action in (glfw.PRESS, glfw.REPEAT):
      if key == glfw.KEY_UP:
        command[0] = clamp(command[0] + COMMAND_X_STEP, *COMMAND_X_RANGE)
      elif key == glfw.KEY_DOWN:
        command[0] = clamp(command[0] - COMMAND_X_STEP, *COMMAND_X_RANGE)
      elif key == glfw.KEY_LEFT:
        command[2] = clamp(command[2] + COMMAND_YAW_STEP, *COMMAND_YAW_RANGE)
      elif key == glfw.KEY_RIGHT:
        command[2] = clamp(command[2] - COMMAND_YAW_STEP, *COMMAND_YAW_RANGE)
      elif key == glfw.KEY_SPACE:
        command[:] = 0.0
      print(f"[command] vx={command[0]:+.2f} vy={command[1]:+.2f} yaw={command[2]:+.2f}")
      return
    orig(window, key, scancode, action, mods)

  glfw.set_key_callback(viewer.window, cb)

  # 右上角 overlay 显示当前速度命令
  original_create_overlay = viewer._create_overlay

  def create_overlay_with_command() -> None:
    original_create_overlay()
    gridpos = mujoco.mjtGridPos.mjGRID_TOPRIGHT
    if gridpos not in viewer._overlay:
      viewer._overlay[gridpos] = ["", ""]
    viewer._overlay[gridpos][0] += (
      "Command vx/vy/yaw\n"
      "Up/Down forward\n"
      "Left/Right yaw\n"
      "Space stop\n"
    )
    viewer._overlay[gridpos][1] += (
      f"{command[0]:+.2f}  {command[1]:+.2f}  {command[2]:+.2f}\n"
      f"+/- {COMMAND_X_STEP:.1f} m/s\n"
      f"+/- {COMMAND_YAW_STEP:.1f} rad/s\n"
      "zero command\n"
    )

  viewer._create_overlay = create_overlay_with_command


def run(model_arg: str = "") -> None:
  model_path = Path(model_arg) if model_arg else find_latest_onnx()
  if not model_path.is_absolute():
    model_path = REPO_ROOT / model_path
  if not model_path.exists():
    raise FileNotFoundError(model_path)

  policy = OnnxPolicy(model_path)
  if policy.output_dim != NUM_POLICY_ACTIONS:
    raise RuntimeError(f"Expected {NUM_POLICY_ACTIONS} actions, got {policy.output_dim}")

  model = build_model()
  data = mujoco.MjData(model)
  default_joint_pos = make_default_joint_pos()
  default_obs_joint_pos = default_joint_pos[OBS_JOINT_INDICES]
  action_scale = make_action_scale()
  ctrl_lo, ctrl_hi = model.actuator_ctrlrange[:, 0].copy(), model.actuator_ctrlrange[:, 1].copy()
  command = np.array(DEFAULT_COMMAND, dtype=np.float64)

  # 默认站姿初始化
  mujoco.mj_resetData(model, data)
  data.qpos[0:3] = np.asarray(HOME_KEYFRAME.pos, dtype=np.float64)
  data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
  data.qpos[7:] = default_joint_pos
  data.qvel[:] = 0.0
  mujoco.mj_forward(model, data)

  history: deque[np.ndarray] = deque(maxlen=HISTORY_LENGTH)
  last_action = np.zeros(NUM_OBS_JOINTS, dtype=np.float32)
  target_pos = default_joint_pos.copy()

  import mujoco_viewer
  viewer = mujoco_viewer.MujocoViewer(model, data, mode="window")
  viewer.cam.distance = 4.0
  viewer.cam.azimuth = 45.0
  viewer.cam.elevation = -20.0
  install_command_controls(viewer, command)

  print("[sim2sim] Up/Down=vx  Left/Right=yaw  Space=stop")
  import time
  next_time = time.perf_counter()
  sim_time = 0.0
  physics_step = 0
  next_print_time = 0.0
  policy_dt = model.opt.timestep * POLICY_DECIMATION
  print(
    f"[sim2sim] dt={model.opt.timestep}, decimation={POLICY_DECIMATION}, "
    f"policy_hz={1.0 / policy_dt:.1f}"
  )
  try:
    while viewer.is_alive:
      if physics_step % POLICY_DECIMATION == 0:
        obs_frame = get_obs_frame(data, command, last_action, default_obs_joint_pos)
        if not history:  # 首步用当前帧回填整个历史
          for _ in range(HISTORY_LENGTH):
            history.append(obs_frame)
        else:
          history.append(obs_frame)

        obs = build_obs_term_major(history)
        action = policy(obs)
        last_action = action[:NUM_OBS_JOINTS].astype(np.float32)

        target_pos = default_joint_pos.copy()
        target_pos[ACTION_JOINT_INDICES] = (
          default_joint_pos[ACTION_JOINT_INDICES] + action * action_scale
        )

      # 训练端 action term 在每个物理子步重新计算肩关节目标。
      knee_diff = data.qpos[7 + LEG_KNEE_L] - data.qpos[7 + LEG_KNEE_R]
      target_pos[ARM_SHOULDER_L] = (
        ARM_SWING_GAIN * knee_diff + default_joint_pos[ARM_SHOULDER_L]
      )
      target_pos[ARM_SHOULDER_R] = (
        -ARM_SWING_GAIN * knee_diff + default_joint_pos[ARM_SHOULDER_R]
      )
      target_pos = np.clip(target_pos, ctrl_lo, ctrl_hi)

      data.ctrl[:] = target_pos
      mujoco.mj_step(model, data)
      physics_step += 1
      viewer.render()
      viewer.cam.lookat = [float(data.qpos[0]), float(data.qpos[1]), float(data.qpos[2])]
      sim_time += model.opt.timestep

      if sim_time >= next_print_time:
        next_print_time += PRINT_EVERY
        actual_qpos = np.asarray(data.qpos[7:], dtype=np.float64)
        jp_str = "  ".join(
          f"{name}={actual_qpos[i]:+.4f}"
          for i, name in enumerate(CASBOT02_23DOF_JOINT_NAMES)
        )
        print(f"t={sim_time:7.2f}s base_z={data.qpos[2]:.3f} command=({command[0]:+.2f},{command[2]:+.2f})")
        print(f"    joint_pos(rad): {jp_str}")

      if DEFAULT_REALTIME:
        next_time += model.opt.timestep / max(DEFAULT_VIEW_SPEED, 1e-6)
        sleep_s = next_time - time.perf_counter()
        if sleep_s > 0:
          time.sleep(sleep_s)
        else:
          next_time = time.perf_counter()
  finally:
    viewer.close()


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description="Casbot02 loco teacher sim2sim")
  p.add_argument("model_path", nargs="?", default="", help="ONNX path (empty = latest)")
  return p.parse_args()


if __name__ == "__main__":
  run(parse_args().model_path)
