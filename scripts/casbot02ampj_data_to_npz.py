"""Convert CASBOT02 1000 Hz ``.data`` files to CASBOT02AMPJ AMP ``.npz``.

The raw recorder layout is the same one handled by amp_mjlab's
``casbot02_data_to_npz.py``.  This version targets HANDOFF's independent
27-joint, fixed-head, no-dexterous-hand CASBOT02AMPJ model and therefore logs
27 joint columns and 30 body columns in the exact order required by the AMP
motion loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
from tqdm import tqdm

import mjlab
from mjlab.entity import Entity
from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_conjugate,
  quat_mul,
)

from wbc_mjlab.casbot02ampj import constants as C


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = (
  REPO_ROOT
  / "src"
  / "wbc_mjlab"
  / "data"
  / "motions"
  / "casbot02ampj"
  / "amp"
  / "WalkandRun"
)

FPS_IN = 1000
FPS_OUT = 50
PITCH_STANDING_OFFSET = np.pi / 2.0
RAW_NUM_COLUMNS = 61

# Raw recorder columns.  The two 7-DoF arm blocks include wrist yaw, pitch,
# and roll.  Waist yaw is stored separately at the final column.
RAW_LEFT_LEG = slice(12, 18)
RAW_RIGHT_LEG = slice(18, 24)
RAW_LEFT_ARM = slice(24, 31)
RAW_RIGHT_ARM = slice(31, 38)
RAW_WAIST_YAW = slice(60, 61)


def euler_pyr_to_quat_xyzw(pyr: np.ndarray) -> np.ndarray:
  """Convert rows of (pitch, yaw, roll) Euler angles to xyzw quaternions."""
  pitch = pyr[:, 0]
  yaw = pyr[:, 1]
  roll = pyr[:, 2]

  cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
  cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
  cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)

  w = cr * cp * cy + sr * sp * sy
  x = sr * cp * cy - cr * sp * sy
  y = cr * sp * cy + sr * cp * sy
  z = cr * cp * sy - sr * sp * cy

  quat = np.stack((x, y, z, w), axis=1)
  norm = np.linalg.norm(quat, axis=1, keepdims=True)
  return (quat / np.clip(norm, 1.0e-8, None)).astype(np.float32)


def _make_quat_continuous(quat_wxyz: torch.Tensor) -> torch.Tensor:
  quat = quat_wxyz.clone()
  for frame in range(1, quat.shape[0]):
    if torch.dot(quat[frame - 1], quat[frame]) < 0.0:
      quat[frame] = -quat[frame]
  return quat


def _so3_derivative(rotations: torch.Tensor, dt: float) -> torch.Tensor:
  """Central-difference world angular velocity from continuous wxyz quats."""
  if rotations.shape[0] < 3:
    return torch.zeros(
      (rotations.shape[0], 3), dtype=rotations.dtype, device=rotations.device
    )

  quat_rel = quat_mul(rotations[2:], quat_conjugate(rotations[:-2]))
  omega = axis_angle_from_quat(quat_rel) / (2.0 * dt)
  return torch.cat((omega[:1], omega, omega[-1:]), dim=0)


def _load_raw_data(path: Path, skip_first_line: bool) -> np.ndarray:
  raw = np.genfromtxt(
    str(path),
    delimiter=",",
    skip_header=1 if skip_first_line else 0,
    invalid_raise=False,
  )
  if raw.ndim == 1:
    raw = raw[None, :]

  # The recorder writes a trailing comma, which genfromtxt exposes as a final
  # all-NaN column.
  while raw.shape[1] > RAW_NUM_COLUMNS and np.all(np.isnan(raw[:, -1])):
    raw = raw[:, :-1]

  valid_rows = ~np.any(np.isnan(raw[:, :RAW_NUM_COLUMNS]), axis=1)
  dropped_rows = int((~valid_rows).sum())
  raw = raw[valid_rows]
  if dropped_rows:
    print(f"  [raw] dropped {dropped_rows} rows containing NaN")

  if raw.shape[1] < RAW_NUM_COLUMNS:
    raise ValueError(
      f"{path}: expected at least {RAW_NUM_COLUMNS} columns, got {raw.shape[1]}"
    )
  if raw.shape[0] < 2:
    raise ValueError(f"{path}: expected at least two valid frames")
  return raw[:, :RAW_NUM_COLUMNS]


class Casbot02AmpjMotion:
  """Downsample and map one raw motion into the 27-joint model convention."""

  def __init__(
    self,
    input_file: Path,
    input_fps: int,
    output_fps: int,
    device: torch.device | str,
    skip_first_line: bool,
  ) -> None:
    if input_fps <= 0 or output_fps <= 0:
      raise ValueError("input_fps and output_fps must be positive")
    if input_fps % output_fps != 0:
      raise ValueError(
        f"Expected integer downsample ratio, got {input_fps} -> {output_fps}"
      )

    raw = _load_raw_data(input_file, skip_first_line)
    raw_frame_count = raw.shape[0]
    stride = input_fps // output_fps
    raw = raw[::stride]
    if raw.shape[0] < 2:
      raise ValueError(
        f"{input_file}: only {raw.shape[0]} frames remain after {stride}x downsample"
      )

    self.input_file = input_file
    self.output_fps = output_fps
    self.output_dt = 1.0 / output_fps
    self.num_frames = raw.shape[0]

    self.root_pos = torch.as_tensor(
      self._root_pos(raw), dtype=torch.float32, device=device
    )
    self.root_quat = torch.as_tensor(
      self._root_quat_wxyz(raw), dtype=torch.float32, device=device
    )
    self.root_quat = _make_quat_continuous(self.root_quat)
    self.joint_pos = torch.as_tensor(
      self._joint_pos(raw), dtype=torch.float32, device=device
    )

    self.root_lin_vel = torch.gradient(
      self.root_pos, spacing=self.output_dt, dim=0
    )[0]
    self.root_ang_vel = _so3_derivative(self.root_quat, self.output_dt)
    self.joint_vel = torch.gradient(
      self.joint_pos, spacing=self.output_dt, dim=0
    )[0]

    print(
      f"Loaded {input_file.name}: {raw_frame_count} raw frames -> "
      f"{self.num_frames} frames @ {output_fps} Hz"
    )

  def clamp_to_joint_limits(self, joint_limits: torch.Tensor) -> None:
    """Clamp small recorder excursions to the model's hard joint limits."""
    if joint_limits.shape != (len(C.JOINT_NAMES), 2):
      raise ValueError(
        f"Expected joint limits [27,2], got {tuple(joint_limits.shape)}"
      )
    clamped = torch.minimum(
      torch.maximum(self.joint_pos, joint_limits[:, 0]), joint_limits[:, 1]
    )
    delta = torch.abs(clamped - self.joint_pos)
    changed = delta > 0.0
    if torch.any(changed):
      changed_frames = int(torch.any(changed, dim=1).sum().item())
      changed_values = int(changed.sum().item())
      max_delta = float(delta.max().item())
      print(
        f"  [limits] clamped {changed_values} values in {changed_frames} frames; "
        f"max correction={max_delta:.6f} rad"
      )
      self.joint_pos = clamped
      self.joint_vel = torch.gradient(
        self.joint_pos, spacing=self.output_dt, dim=0
      )[0]

  @staticmethod
  def _root_pos(raw: np.ndarray) -> np.ndarray:
    source_pos = raw[:, 0:3].astype(np.float32)
    # Recorder: +y forward, +x left.  MuJoCo: +x forward, +y left.
    return np.stack(
      (source_pos[:, 1], -source_pos[:, 0], source_pos[:, 2]), axis=1
    ).astype(np.float32)

  @staticmethod
  def _root_quat_wxyz(raw: np.ndarray) -> np.ndarray:
    source_pyr = raw[:, 3:6].astype(np.float32)
    pitch_mean = float(np.mean(source_pyr[:, 0]))
    if abs(pitch_mean - PITCH_STANDING_OFFSET) < abs(pitch_mean):
      pitch = source_pyr[:, 0] - PITCH_STANDING_OFFSET
      print(f"  [pitch] mean={pitch_mean:.5f}: subtract pi/2")
    else:
      pitch = source_pyr[:, 0].copy()
      print(f"  [pitch] mean={pitch_mean:.5f}: no offset")

    roll = source_pyr[:, 1]
    yaw = -source_pyr[:, 2]
    quat_xyzw = euler_pyr_to_quat_xyzw(
      np.stack((pitch, yaw, roll), axis=1)
    )
    return quat_xyzw[:, (3, 0, 1, 2)].astype(np.float32)

  @staticmethod
  def _joint_pos(raw: np.ndarray) -> np.ndarray:
    joint_pos = np.concatenate(
      (
        raw[:, RAW_LEFT_LEG],
        raw[:, RAW_RIGHT_LEG],
        raw[:, RAW_WAIST_YAW],
        raw[:, RAW_LEFT_ARM],
        raw[:, RAW_RIGHT_ARM],
      ),
      axis=1,
    ).astype(np.float32)
    if joint_pos.shape[1] != len(C.JOINT_NAMES):
      raise ValueError(
        f"Expected {len(C.JOINT_NAMES)} joint columns, got {joint_pos.shape[1]}"
      )
    return joint_pos


def _build_fk_scene(
  device: str,
  output_fps: int,
) -> tuple[Simulation, Scene]:
  scene_cfg = SceneCfg(
    num_envs=1,
    entities={"robot": C.get_robot_cfg()},
  )
  scene = Scene(scene_cfg, device=device)
  mj_model = scene.compile()

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=mj_model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  robot: Entity = scene["robot"]
  if robot.joint_names != C.JOINT_NAMES:
    raise RuntimeError(
      f"Joint order mismatch:\nexpected={C.JOINT_NAMES}\nactual={robot.joint_names}"
    )
  if len(robot.body_names) != 30:
    raise RuntimeError(f"Expected 30 no-hand bodies, got {len(robot.body_names)}")
  return sim, scene


def _validate_log(log: dict[str, Any], num_frames: int) -> None:
  expected_shapes = {
    "joint_pos": (num_frames, 27),
    "joint_vel": (num_frames, 27),
    "body_pos_w": (num_frames, 30, 3),
    "body_quat_w": (num_frames, 30, 4),
    "body_lin_vel_w": (num_frames, 30, 3),
    "body_ang_vel_w": (num_frames, 30, 3),
  }
  for key, expected in expected_shapes.items():
    value = np.asarray(log[key])
    if value.shape != expected:
      raise ValueError(f"{key}: expected {expected}, got {value.shape}")
    if not np.isfinite(value).all():
      raise ValueError(f"{key}: contains NaN or Inf")

  quat_norm = np.linalg.norm(np.asarray(log["body_quat_w"]), axis=-1)
  max_quat_error = float(np.max(np.abs(quat_norm - 1.0)))
  if max_quat_error > 2.0e-3:
    raise ValueError(f"Body quaternion max norm error is {max_quat_error:.6f}")


def convert_file(
  sim: Simulation,
  scene: Scene,
  input_file: Path,
  output_path: Path,
  input_fps: int,
  output_fps: int,
  skip_first_line: bool,
  overwrite: bool,
  clamp_joint_limits: bool,
) -> None:
  if output_path.exists() and not overwrite:
    print(f"[SKIP] {output_path} already exists (use --overwrite)")
    return

  motion = Casbot02AmpjMotion(
    input_file=input_file,
    input_fps=input_fps,
    output_fps=output_fps,
    device=sim.device,
    skip_first_line=skip_first_line,
  )
  robot: Entity = scene["robot"]
  if clamp_joint_limits:
    motion.clamp_to_joint_limits(robot.data.joint_pos_limits[0])
  joint_ids, matched_names = robot.find_joints(
    C.JOINT_NAMES, preserve_order=True
  )
  if tuple(matched_names) != C.JOINT_NAMES:
    raise RuntimeError(f"Failed to resolve all 27 joints: {matched_names}")

  log: dict[str, Any] = {
    "fps": np.asarray([output_fps], dtype=np.float32),
    "joint_pos": [],
    "joint_vel": [],
    "body_pos_w": [],
    "body_quat_w": [],
    "body_lin_vel_w": [],
    "body_ang_vel_w": [],
  }

  scene.reset()
  for frame in tqdm(
    range(motion.num_frames), desc=input_file.name, unit="frame"
  ):
    root_state = robot.data.default_root_state.clone()
    root_state[:, 0:3] = motion.root_pos[frame : frame + 1]
    root_state[:, 0:2] += scene.env_origins[:, 0:2]
    root_state[:, 3:7] = motion.root_quat[frame : frame + 1]
    root_state[:, 7:10] = motion.root_lin_vel[frame : frame + 1]
    root_state[:, 10:13] = motion.root_ang_vel[frame : frame + 1]
    robot.write_root_state_to_sim(root_state)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, joint_ids] = motion.joint_pos[frame : frame + 1]
    joint_vel[:, joint_ids] = motion.joint_vel[frame : frame + 1]
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)

    log["joint_pos"].append(
      robot.data.joint_pos[0, joint_ids].cpu().numpy().copy()
    )
    log["joint_vel"].append(
      robot.data.joint_vel[0, joint_ids].cpu().numpy().copy()
    )
    log["body_pos_w"].append(
      robot.data.body_link_pos_w[0].cpu().numpy().copy()
    )
    log["body_quat_w"].append(
      robot.data.body_link_quat_w[0].cpu().numpy().copy()
    )
    log["body_lin_vel_w"].append(
      robot.data.body_link_lin_vel_w[0].cpu().numpy().copy()
    )
    log["body_ang_vel_w"].append(
      robot.data.body_link_ang_vel_w[0].cpu().numpy().copy()
    )

  for key in (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
  ):
    log[key] = np.stack(log[key], axis=0).astype(np.float32)
  _validate_log(log, motion.num_frames)

  output_path.parent.mkdir(parents=True, exist_ok=True)
  temp_path = output_path.with_name(f".{output_path.name}.tmp")
  with temp_path.open("wb") as output_file:
    np.savez(output_file, **log)
  temp_path.replace(output_path)
  print(
    f"[OK] {input_file.name} -> {output_path} "
    f"({motion.num_frames} frames, 27 joints, 30 bodies)"
  )


def main(
  input_file: str | None = None,
  input_dir: str = str(DEFAULT_DATA_DIR),
  output_name: str | None = None,
  output_dir: str = str(DEFAULT_DATA_DIR),
  input_fps: int = FPS_IN,
  output_fps: int = FPS_OUT,
  device: str = "cuda:0",
  skip_first_line: bool = False,
  overwrite: bool = False,
  clamp_joint_limits: bool = True,
) -> None:
  """Convert one file or every .data file in a directory.

  With no arguments, converts all current WalkandRun ``.data`` files beside
  their sources. Existing ``.npz`` outputs are skipped unless ``--overwrite``
  is supplied. Use ``--device cpu`` when CUDA is unavailable.
  """
  output_root = Path(output_dir).expanduser().resolve()
  if input_file is not None:
    source = Path(input_file).expanduser().resolve()
    if not source.is_file():
      raise FileNotFoundError(source)
    name = output_name or source.with_suffix(".npz").name
    if not name.endswith(".npz"):
      name += ".npz"
    file_pairs = ((source, output_root / name),)
  else:
    source_dir = Path(input_dir).expanduser().resolve()
    data_files = sorted(source_dir.glob("*.data"))
    if not data_files:
      raise FileNotFoundError(f"No .data files found in {source_dir}")
    file_pairs = tuple(
      (source, output_root / source.with_suffix(".npz").name)
      for source in data_files
    )
    print(f"Found {len(file_pairs)} .data files in {source_dir}")

  sim, scene = _build_fk_scene(device=device, output_fps=output_fps)
  for index, (source, destination) in enumerate(file_pairs, start=1):
    print(f"\n[{index}/{len(file_pairs)}] {source.name}")
    convert_file(
      sim=sim,
      scene=scene,
      input_file=source,
      output_path=destination,
      input_fps=input_fps,
      output_fps=output_fps,
      skip_first_line=skip_first_line,
      overwrite=overwrite,
      clamp_joint_limits=clamp_joint_limits,
    )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
