"""Replay a CASBOT02AMPJ 27-DoF AMP NPZ in a native MuJoCo window.

The motion is kinematically replayed: every frame writes the recorded root
pose/velocity and all 27 joint positions/velocities directly into MuJoCo.
This is intended for checking retargeted data, not for evaluating a policy or
testing whether the robot can physically track the trajectory.

Keyboard controls:

* Space: pause/resume
* R: restart from the selected start time
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time

import mujoco
import numpy as np

from scripts.sim2sim_casbot02ampj_amp_onnx import build_model
from wbc_mjlab.casbot02ampj import constants as C


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
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

REQUIRED_KEYS = (
  "fps",
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)


@dataclass(frozen=True)
class MotionData:
  path: Path
  fps: float
  joint_pos: np.ndarray
  joint_vel: np.ndarray
  root_pos_w: np.ndarray
  root_quat_wxyz: np.ndarray
  root_lin_vel_w: np.ndarray
  root_ang_vel_w: np.ndarray

  @property
  def num_frames(self) -> int:
    return int(self.joint_pos.shape[0])

  @property
  def duration(self) -> float:
    return self.num_frames / self.fps


@dataclass(frozen=True)
class ModelAddresses:
  joint_qpos: np.ndarray
  joint_qvel: np.ndarray


def load_motion(path: Path) -> MotionData:
  """Load and validate the NPZ contract used by the AMP teacher."""
  if not path.is_file():
    raise FileNotFoundError(path)

  with np.load(path, allow_pickle=False) as archive:
    missing = [key for key in REQUIRED_KEYS if key not in archive]
    if missing:
      raise ValueError(f"{path}: missing NPZ fields: {missing}")

    fps_array = np.asarray(archive["fps"])
    if fps_array.size != 1:
      raise ValueError(f"{path}: fps must be scalar, got {fps_array.shape}")
    fps = float(fps_array.item())
    if not np.isfinite(fps) or fps <= 0.0:
      raise ValueError(f"{path}: invalid fps={fps}")

    joint_pos = np.asarray(archive["joint_pos"], dtype=np.float64)
    joint_vel = np.asarray(archive["joint_vel"], dtype=np.float64)
    body_pos_w = np.asarray(archive["body_pos_w"], dtype=np.float64)
    body_quat_w = np.asarray(archive["body_quat_w"], dtype=np.float64)
    body_lin_vel_w = np.asarray(archive["body_lin_vel_w"], dtype=np.float64)
    body_ang_vel_w = np.asarray(archive["body_ang_vel_w"], dtype=np.float64)

  expected_joints = len(C.JOINT_NAMES)
  if joint_pos.ndim != 2 or joint_pos.shape[1] != expected_joints:
    raise ValueError(
      f"{path}: joint_pos must be [T,{expected_joints}], got {joint_pos.shape}"
    )
  if joint_vel.shape != joint_pos.shape:
    raise ValueError(
      f"{path}: joint_vel must have shape {joint_pos.shape}, got {joint_vel.shape}"
    )
  num_frames = joint_pos.shape[0]
  if num_frames < 2:
    raise ValueError(f"{path}: expected at least two frames, got {num_frames}")

  expected_body_shapes = {
    "body_pos_w": (num_frames, 3),
    "body_quat_w": (num_frames, 4),
    "body_lin_vel_w": (num_frames, 3),
    "body_ang_vel_w": (num_frames, 3),
  }
  root_arrays = {
    "body_pos_w": body_pos_w[:, 0, :] if body_pos_w.ndim == 3 else body_pos_w,
    "body_quat_w": (
      body_quat_w[:, 0, :] if body_quat_w.ndim == 3 else body_quat_w
    ),
    "body_lin_vel_w": (
      body_lin_vel_w[:, 0, :]
      if body_lin_vel_w.ndim == 3
      else body_lin_vel_w
    ),
    "body_ang_vel_w": (
      body_ang_vel_w[:, 0, :]
      if body_ang_vel_w.ndim == 3
      else body_ang_vel_w
    ),
  }
  for key, expected_shape in expected_body_shapes.items():
    if root_arrays[key].shape != expected_shape:
      raise ValueError(
        f"{path}: root {key} must be {expected_shape}, got "
        f"{root_arrays[key].shape}"
      )

  arrays = (joint_pos, joint_vel, *root_arrays.values())
  if not all(np.isfinite(array).all() for array in arrays):
    raise ValueError(f"{path}: motion contains NaN or Inf")

  root_quat = root_arrays["body_quat_w"]
  quat_norm = np.linalg.norm(root_quat, axis=-1)
  max_quat_error = float(np.max(np.abs(quat_norm - 1.0)))
  if max_quat_error > 2.0e-2:
    raise ValueError(
      f"{path}: root quaternion max norm error is {max_quat_error:.6f}"
    )
  root_quat = root_quat / np.maximum(quat_norm[:, None], 1.0e-12)

  return MotionData(
    path=path,
    fps=fps,
    joint_pos=joint_pos,
    joint_vel=joint_vel,
    root_pos_w=root_arrays["body_pos_w"],
    root_quat_wxyz=root_quat,
    root_lin_vel_w=root_arrays["body_lin_vel_w"],
    root_ang_vel_w=root_arrays["body_ang_vel_w"],
  )


def resolve_model_addresses(model: mujoco.MjModel) -> ModelAddresses:
  """Resolve qpos/qvel addresses without assuming MuJoCo joint ordering."""
  qpos_addresses: list[int] = []
  qvel_addresses: list[int] = []
  for name in C.JOINT_NAMES:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
      raise ValueError(f"Playback model is missing joint {name!r}")
    if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
      raise ValueError(f"Playback joint {name!r} is not a hinge")
    qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
    qvel_addresses.append(int(model.jnt_dofadr[joint_id]))
  return ModelAddresses(
    joint_qpos=np.asarray(qpos_addresses, dtype=np.int32),
    joint_qvel=np.asarray(qvel_addresses, dtype=np.int32),
  )


def write_frame(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  motion: MotionData,
  addresses: ModelAddresses,
  frame: int,
  xy_offset: np.ndarray,
) -> None:
  data.qpos[0:3] = motion.root_pos_w[frame]
  data.qpos[0:2] -= xy_offset
  data.qpos[3:7] = motion.root_quat_wxyz[frame]
  data.qpos[addresses.joint_qpos] = motion.joint_pos[frame]

  data.qvel[0:3] = motion.root_lin_vel_w[frame]
  data.qvel[3:6] = motion.root_ang_vel_w[frame]
  data.qvel[addresses.joint_qvel] = motion.joint_vel[frame]
  mujoco.mj_forward(model, data)


def time_to_frame(time_s: float, fps: float, num_frames: int) -> int:
  return int(np.clip(round(time_s * fps), 0, num_frames - 1))


def play(
  motion: MotionData,
  *,
  speed: float,
  start_time: float,
  end_time: float | None,
  loop: bool,
  follow_camera: bool,
  zero_xy: bool,
) -> None:
  if speed <= 0.0:
    raise ValueError(f"speed must be positive, got {speed}")
  if start_time < 0.0:
    raise ValueError(f"start_time must be non-negative, got {start_time}")

  start_frame = time_to_frame(start_time, motion.fps, motion.num_frames)
  end_frame = (
    motion.num_frames - 1
    if end_time is None
    else time_to_frame(end_time, motion.fps, motion.num_frames)
  )
  if end_frame < start_frame:
    raise ValueError(
      f"end time/frame precedes start: {end_time}s/{end_frame} < "
      f"{start_time}s/{start_frame}"
    )

  model = build_model()
  data = mujoco.MjData(model)
  addresses = resolve_model_addresses(model)
  xy_offset = (
    motion.root_pos_w[start_frame, :2].copy()
    if zero_xy
    else np.zeros(2, dtype=np.float64)
  )
  write_frame(model, data, motion, addresses, start_frame, xy_offset)

  import glfw
  import mujoco.viewer as mj_viewer

  control = {"paused": False, "restart": False}

  def key_callback(keycode: int) -> None:
    if keycode == glfw.KEY_SPACE:
      control["paused"] = not control["paused"]
      print("[play] paused" if control["paused"] else "[play] resumed")
    elif keycode == glfw.KEY_R:
      control["restart"] = True

  frame_dt = 1.0 / (motion.fps * speed)
  selected_duration = (end_frame - start_frame + 1) / motion.fps
  print(
    f"[play] frames={start_frame}:{end_frame} "
    f"duration={selected_duration:.3f}s speed={speed:.2f}x loop={loop}"
  )
  print("[play] Space=pause/resume  R=restart  close window=stop")

  with mj_viewer.launch_passive(
    model, data, key_callback=key_callback
  ) as viewer:
    viewer.cam.distance = 3.5
    viewer.cam.azimuth = 45.0
    viewer.cam.elevation = -20.0
    frame = start_frame
    next_frame_time = time.perf_counter()

    while viewer.is_running():
      if control["restart"]:
        frame = start_frame
        control["restart"] = False
        next_frame_time = time.perf_counter()

      if control["paused"]:
        viewer.sync()
        time.sleep(min(frame_dt, 0.02))
        next_frame_time = time.perf_counter()
        continue

      write_frame(model, data, motion, addresses, frame, xy_offset)
      if follow_camera:
        viewer.cam.lookat[:] = data.qpos[0:3]
      viewer.sync()

      frame += 1
      if frame > end_frame:
        if loop:
          frame = start_frame
        else:
          break

      next_frame_time += frame_dt
      sleep_s = next_frame_time - time.perf_counter()
      if sleep_s > 0.0:
        time.sleep(sleep_s)
      else:
        next_frame_time = time.perf_counter()


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Replay CASBOT02AMPJ 27-DoF AMP NPZ motion data"
  )
  parser.add_argument(
    "input_file",
    nargs="?",
    type=Path,
    default=DEFAULT_INPUT,
    help=f"motion NPZ (default: {DEFAULT_INPUT.relative_to(REPO_ROOT)})",
  )
  parser.add_argument("--speed", type=float, default=1.0, help="playback speed")
  parser.add_argument(
    "--start-time", type=float, default=0.0, help="selected start time in seconds"
  )
  parser.add_argument(
    "--end-time", type=float, default=None, help="selected end time in seconds"
  )
  parser.add_argument(
    "--loop",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="loop the selected interval",
  )
  parser.add_argument(
    "--follow-camera",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="keep the camera centered on the recorded root",
  )
  parser.add_argument(
    "--zero-xy",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="translate the selected first root position to world x=y=0",
  )
  parser.add_argument(
    "--check-only",
    action="store_true",
    help="validate the NPZ/model contract without opening a window",
  )
  return parser.parse_args()


def main() -> int:
  args = parse_args()
  input_path = args.input_file.expanduser()
  if not input_path.is_absolute():
    input_path = (Path.cwd() / input_path).resolve()
  motion = load_motion(input_path)
  print(
    f"[motion] {motion.path}\n"
    f"[motion] frames={motion.num_frames}, fps={motion.fps:g}, "
    f"duration={motion.duration:.3f}s, joints={motion.joint_pos.shape[1]}"
  )

  if args.check_only:
    model = build_model()
    addresses = resolve_model_addresses(model)
    data = mujoco.MjData(model)
    write_frame(
      model,
      data,
      motion,
      addresses,
      frame=0,
      xy_offset=np.zeros(2, dtype=np.float64),
    )
    print(
      f"[check] OK: model nq={model.nq}, nv={model.nv}, "
      f"nu={model.nu}, mapped_joints={len(addresses.joint_qpos)}"
    )
    return 0

  play(
    motion,
    speed=args.speed,
    start_time=args.start_time,
    end_time=args.end_time,
    loop=args.loop,
    follow_camera=args.follow_camera,
    zero_xy=args.zero_xy,
  )
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
