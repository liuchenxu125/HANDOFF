"""CASBOT02AMPJ get-up-only ONNX policy in native MuJoCo.

The robot is initialized from a fallen frame in a retargeted AMP NPZ.  The
policy receives the same 4-frame, term-major observation as the Get-Up AMP
teacher, with a permanently zero velocity command and no upward assist force.

Keyboard: R resets the robot to the selected fallen frame.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import time

import mujoco
import numpy as np

from scripts.play_casbot02ampj_amp_data import (
  MotionData,
  load_motion,
  resolve_model_addresses,
)
from scripts.sim2sim_casbot02ampj_amp_onnx import (
  HISTORY_LENGTH,
  NUM_JOINTS,
  POLICY_DECIMATION,
  SIM_TIMESTEP,
  SINGLE_FRAME_OBS_SIZE,
  OnnxPolicy,
  build_model,
  build_obs_term_major,
  get_obs_frame,
  make_action_scale,
  make_default_joint_pos,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOTION = (
  REPO_ROOT
  / "src"
  / "wbc_mjlab"
  / "data"
  / "motions"
  / "casbot02ampj"
  / "preview"
  / "GetUp"
  / "fallAndGetUp3_subject1_270_1770.npz"
)
GETUP_LOG_ROOT = REPO_ROOT / "logs" / "rsl_rl" / "casbot02_ampj_getup_teacher_flat"


def find_latest_onnx() -> Path:
  if not GETUP_LOG_ROOT.is_dir():
    raise FileNotFoundError(f"No Get-Up training output under {GETUP_LOG_ROOT}")
  candidates = sorted(GETUP_LOG_ROOT.glob("*/*.onnx"), reverse=True)
  if not candidates:
    raise FileNotFoundError(f"No Get-Up ONNX under {GETUP_LOG_ROOT}")
  return candidates[0]


def roll_pitch_wxyz(quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  w, x, y, z = np.moveaxis(quat, -1, 0)
  roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
  pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
  return roll, pitch


def select_fallen_frame(motion: MotionData, requested_frame: int) -> int:
  if requested_frame >= 0:
    if requested_frame >= motion.num_frames:
      raise ValueError(
        f"frame {requested_frame} is outside [0, {motion.num_frames - 1}]"
      )
    return requested_frame

  roll, pitch = roll_pitch_wxyz(motion.root_quat_wxyz)
  fallen = (motion.root_pos_w[:, 2] < 0.45) & (
    (np.abs(roll) > 1.0) | (np.abs(pitch) > 1.0)
  )
  candidates = np.flatnonzero(fallen)
  if candidates.size == 0:
    raise ValueError(
      f"{motion.path}: no fallen frame matching root z<0.45 m and tilt>1 rad; "
      "pass --frame explicitly to override"
    )

  # Prefer a quiet fallen frame so the test measures recovery from rest rather
  # than inheriting a large falling velocity from the reference trajectory.
  motion_speed = np.linalg.norm(motion.root_lin_vel_w[candidates], axis=1)
  motion_speed += 0.25 * np.linalg.norm(
    motion.root_ang_vel_w[candidates], axis=1
  )
  return int(candidates[np.argmin(motion_speed)])


def reset_from_motion(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  motion: MotionData,
  frame: int,
  *,
  zero_velocity: bool,
) -> None:
  addresses = resolve_model_addresses(model)
  mujoco.mj_resetData(model, data)
  data.qpos[0:2] = 0.0
  data.qpos[2] = motion.root_pos_w[frame, 2]
  data.qpos[3:7] = motion.root_quat_wxyz[frame]
  data.qpos[addresses.joint_qpos] = motion.joint_pos[frame]

  if not zero_velocity:
    data.qvel[0:3] = motion.root_lin_vel_w[frame]
    data.qvel[3:6] = motion.root_ang_vel_w[frame]
    data.qvel[addresses.joint_qvel] = motion.joint_vel[frame]
  data.ctrl[:] = motion.joint_pos[frame]
  mujoco.mj_forward(model, data)


def run(
  model_path: Path,
  motion_path: Path,
  frame_arg: int,
  zero_velocity: bool,
  real_time_speed: float,
) -> None:
  if not model_path.is_file():
    raise FileNotFoundError(model_path)
  if real_time_speed <= 0.0:
    raise ValueError(f"real-time-speed must be positive, got {real_time_speed}")

  motion = load_motion(motion_path)
  fallen_frame = select_fallen_frame(motion, frame_arg)
  policy = OnnxPolicy(model_path)
  expected_input = SINGLE_FRAME_OBS_SIZE * HISTORY_LENGTH
  if policy.input_dim != expected_input or policy.output_dim != NUM_JOINTS:
    raise RuntimeError(
      f"Expected Get-Up ONNX {expected_input} -> {NUM_JOINTS}, got "
      f"{policy.input_dim} -> {policy.output_dim}"
    )

  model = build_model()
  data = mujoco.MjData(model)
  default_joint_pos = make_default_joint_pos()
  action_scale = make_action_scale()
  ctrl_low = model.actuator_ctrlrange[:, 0]
  ctrl_high = model.actuator_ctrlrange[:, 1]
  command = np.zeros(3, dtype=np.float64)
  history: deque[np.ndarray] = deque(maxlen=HISTORY_LENGTH)
  last_action = np.zeros(NUM_JOINTS, dtype=np.float32)
  target_pos = motion.joint_pos[fallen_frame].copy()

  def reset() -> None:
    nonlocal last_action, target_pos
    reset_from_motion(
      model,
      data,
      motion,
      fallen_frame,
      zero_velocity=zero_velocity,
    )
    history.clear()
    last_action = np.zeros(NUM_JOINTS, dtype=np.float32)
    target_pos = motion.joint_pos[fallen_frame].copy()

  reset()

  import glfw
  import mujoco.viewer as mj_viewer

  control = {"reset": False}

  def key_callback(keycode: int) -> None:
    if keycode == glfw.KEY_R:
      control["reset"] = True

  roll, pitch = roll_pitch_wxyz(
    motion.root_quat_wxyz[fallen_frame : fallen_frame + 1]
  )
  print(f"[ONNX] model={model_path}")
  print(f"[motion] file={motion.path}")
  print(
    f"[reset] frame={fallen_frame}, time={fallen_frame / motion.fps:.3f}s, "
    f"root_z={motion.root_pos_w[fallen_frame, 2]:.3f}m, "
    f"roll={float(roll[0]):+.3f}, pitch={float(pitch[0]):+.3f}, "
    f"zero_velocity={zero_velocity}"
  )
  print("[sim2sim] command=(0,0,0), assist_force=0 N, R=reset")

  physics_step = 0
  next_time = time.perf_counter()
  with mj_viewer.launch_passive(
    model, data, key_callback=key_callback
  ) as viewer:
    viewer.cam.distance = 3.5
    viewer.cam.azimuth = 45.0
    viewer.cam.elevation = -20.0
    while viewer.is_running():
      if control["reset"]:
        reset()
        physics_step = 0
        control["reset"] = False
        next_time = time.perf_counter()

      if physics_step % POLICY_DECIMATION == 0:
        obs_frame = get_obs_frame(
          data, command, last_action, default_joint_pos
        )
        if not history:
          for _ in range(HISTORY_LENGTH):
            history.append(obs_frame)
        else:
          history.append(obs_frame)
        action = policy(build_obs_term_major(history))
        last_action = action.copy()
        target_pos = np.clip(
          default_joint_pos + action * action_scale,
          ctrl_low,
          ctrl_high,
        )

      data.ctrl[:] = target_pos
      mujoco.mj_step(model, data)
      physics_step += 1
      viewer.cam.lookat[:] = data.qpos[0:3]
      viewer.sync()

      next_time += SIM_TIMESTEP / real_time_speed
      sleep_s = next_time - time.perf_counter()
      if sleep_s > 0.0:
        time.sleep(sleep_s)
      else:
        next_time = time.perf_counter()


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="CASBOT02AMPJ get-up-only ONNX sim2sim"
  )
  parser.add_argument(
    "model_path",
    nargs="?",
    type=Path,
    default=None,
    help="Get-Up ONNX path (default: latest exported run)",
  )
  parser.add_argument(
    "--motion",
    type=Path,
    default=DEFAULT_MOTION,
    help="NPZ supplying the fallen reset pose",
  )
  parser.add_argument(
    "--frame",
    type=int,
    default=-1,
    help="reset frame; -1 selects the quietest strict fallen frame",
  )
  parser.add_argument(
    "--zero-velocity",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="zero all reset velocities",
  )
  parser.add_argument(
    "--real-time-speed",
    type=float,
    default=1.0,
    help="wall-clock playback speed",
  )
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  model_path = find_latest_onnx() if args.model_path is None else args.model_path
  motion_path = args.motion.expanduser()
  if not model_path.is_absolute():
    model_path = (Path.cwd() / model_path).resolve()
  if not motion_path.is_absolute():
    motion_path = (Path.cwd() / motion_path).resolve()
  run(
    model_path=model_path,
    motion_path=motion_path,
    frame_arg=args.frame,
    zero_velocity=args.zero_velocity,
    real_time_speed=args.real_time_speed,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
