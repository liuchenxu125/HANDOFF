"""Convert CASBOT02 recovery CSV files to CASBOT02AMPJ AMP NPZ files.

Accepted CSV layouts:

* base position: 3 columns
* base quaternion in xyzw order: 4 columns
* either the source 29-joint XML order (36 columns total), or the direct
  CASBOT02AMPJ 27-joint order (34 columns total)

The source joint vector contains head yaw/pitch.  Those two columns are
discarded because the CASBOT02AMPJ recovery model fixes the head, yielding the
27 policy joints and 30 no-dexterous-hand bodies required by this task.
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

from scripts.casbot02ampj_data_to_npz import (
  _build_fk_scene,
  _make_quat_continuous,
  _so3_derivative,
  _validate_log,
)
from wbc_mjlab.casbot02ampj import constants as C


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RECOVERY_DIR = (
  REPO_ROOT
  / "src"
  / "wbc_mjlab"
  / "data"
  / "motions"
  / "casbot02ampj"
  / "amp"
  / "Recovery"
)

INPUT_COLUMNS = (34, 36)
INPUT_JOINT_COUNT = 29
INPUT_FPS = 30.0
OUTPUT_FPS = 50.0

SOURCE_JOINT_NAMES: tuple[str, ...] = (
  "left_leg_pelvic_pitch_joint",
  "left_leg_pelvic_roll_joint",
  "left_leg_pelvic_yaw_joint",
  "left_leg_knee_pitch_joint",
  "left_leg_ankle_pitch_joint",
  "left_leg_ankle_roll_joint",
  "right_leg_pelvic_pitch_joint",
  "right_leg_pelvic_roll_joint",
  "right_leg_pelvic_yaw_joint",
  "right_leg_knee_pitch_joint",
  "right_leg_ankle_pitch_joint",
  "right_leg_ankle_roll_joint",
  "waist_yaw_joint",
  "head_yaw_joint",
  "head_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_pitch_joint",
  "left_wrist_yaw_joint",
  "left_wrist_pitch_joint",
  "left_wrist_roll_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_pitch_joint",
  "right_wrist_yaw_joint",
  "right_wrist_pitch_joint",
  "right_wrist_roll_joint",
)
assert len(SOURCE_JOINT_NAMES) == INPUT_JOINT_COUNT

TARGET_SOURCE_INDICES: tuple[int, ...] = tuple(
  SOURCE_JOINT_NAMES.index(name) for name in C.JOINT_NAMES
)
assert len(TARGET_SOURCE_INDICES) == 27
assert set(SOURCE_JOINT_NAMES) - set(C.JOINT_NAMES) == set(C.HEAD_JOINT_NAMES)


def _quat_slerp_batch(
  quat0: torch.Tensor,
  quat1: torch.Tensor,
  blend: torch.Tensor,
) -> torch.Tensor:
  """Vectorized shortest-path slerp for wxyz quaternion batches."""
  blend = blend.unsqueeze(-1)
  dot = torch.sum(quat0 * quat1, dim=-1, keepdim=True)
  quat1 = torch.where(dot < 0.0, -quat1, quat1)
  dot = torch.abs(dot).clamp(max=1.0)

  theta = torch.acos(dot)
  sin_theta = torch.sin(theta)
  safe_sin = torch.clamp(sin_theta, min=1.0e-8)
  s0 = torch.sin((1.0 - blend) * theta) / safe_sin
  s1 = torch.sin(blend * theta) / safe_sin
  spherical = s0 * quat0 + s1 * quat1
  linear = (1.0 - blend) * quat0 + blend * quat1
  output = torch.where(dot > 0.9995, linear, spherical)
  return output / torch.linalg.norm(output, dim=-1, keepdim=True).clamp(min=1.0e-8)


def _load_csv(path: Path) -> np.ndarray:
  motion = np.loadtxt(path, delimiter=",", ndmin=2)
  if motion.shape[1] not in INPUT_COLUMNS:
    joint_columns = motion.shape[1] - 7
    raise ValueError(
      f"{path}: expected 34 or 36 columns (3 position + 4 quaternion + "
      f"27 or 29 joints), "
      f"got {motion.shape[1]} columns ({joint_columns} joint columns)"
    )
  if motion.shape[0] < 2:
    raise ValueError(f"{path}: expected at least two frames")
  if not np.isfinite(motion).all():
    raise ValueError(f"{path}: contains NaN or Inf")
  return motion.astype(np.float32)


class Casbot02AmpjCsvMotion:
  """Load 29-joint recovery CSV, reduce to 27 joints, and resample."""

  def __init__(
    self,
    input_file: Path,
    input_fps: float,
    output_fps: float,
    device: torch.device | str,
    require_recovery_coverage: bool = True,
  ) -> None:
    if input_fps <= 0.0 or output_fps <= 0.0:
      raise ValueError("input_fps and output_fps must be positive")

    raw = _load_csv(input_file)
    self.input_file = input_file
    self.input_fps = input_fps
    self.output_fps = output_fps
    self.output_dt = 1.0 / output_fps
    self.input_frames = raw.shape[0]
    self.duration = (self.input_frames - 1) / input_fps

    root_pos_input = torch.as_tensor(
      raw[:, 0:3], dtype=torch.float32, device=device
    )
    root_quat_xyzw = torch.as_tensor(
      raw[:, 3:7], dtype=torch.float32, device=device
    )
    root_quat_input = root_quat_xyzw[:, (3, 0, 1, 2)]
    root_quat_input = root_quat_input / torch.linalg.norm(
      root_quat_input, dim=-1, keepdim=True
    ).clamp(min=1.0e-8)
    root_quat_input = _make_quat_continuous(root_quat_input)

    source_joint_pos = torch.as_tensor(
      raw[:, 7:], dtype=torch.float32, device=device
    )
    if source_joint_pos.shape[1] == len(C.JOINT_NAMES):
      print("  [dimension] direct CASBOT02AMPJ target joints=27 (head fixed)")
      joint_pos_input = source_joint_pos
    elif source_joint_pos.shape[1] == INPUT_JOINT_COUNT:
      head_indices = tuple(
        SOURCE_JOINT_NAMES.index(name) for name in C.HEAD_JOINT_NAMES
      )
      head_max = float(torch.abs(source_joint_pos[:, head_indices]).max().item())
      print(
        f"  [dimension] CSV joints=29; remove head yaw/pitch -> target joints=27; "
        f"head max |q|={head_max:.6f} rad"
      )
      joint_pos_input = source_joint_pos[:, TARGET_SOURCE_INDICES]
    else:
      raise ValueError(
        f"Expected 27 or {INPUT_JOINT_COUNT} source joints, "
        f"got {source_joint_pos.shape[1]}"
      )

    times = torch.arange(
      0.0,
      self.duration,
      self.output_dt,
      dtype=torch.float32,
      device=device,
    )
    if times.numel() < 3:
      raise ValueError(
        f"{input_file}: only {times.numel()} frames after interpolation"
      )
    source_phase = times * input_fps
    index0 = torch.floor(source_phase).long().clamp(max=self.input_frames - 1)
    index1 = torch.clamp(index0 + 1, max=self.input_frames - 1)
    blend = source_phase - index0.to(source_phase.dtype)

    self.root_pos = self._lerp(
      root_pos_input[index0], root_pos_input[index1], blend
    )
    self.root_quat = _quat_slerp_batch(
      root_quat_input[index0], root_quat_input[index1], blend
    )
    self.joint_pos = self._lerp(
      joint_pos_input[index0], joint_pos_input[index1], blend
    )
    self.num_frames = times.numel()

    self.root_lin_vel = torch.gradient(
      self.root_pos, spacing=self.output_dt, dim=0
    )[0]
    self.root_ang_vel = _so3_derivative(self.root_quat, self.output_dt)
    self.joint_vel = torch.gradient(
      self.joint_pos, spacing=self.output_dt, dim=0
    )[0]

    print(
      f"  [frames] {self.input_frames} @ {input_fps:g} Hz -> "
      f"{self.num_frames} @ {output_fps:g} Hz; duration={self.duration:.3f}s"
    )
    self._report_recovery_coverage(require=require_recovery_coverage)

  @staticmethod
  def _lerp(
    value0: torch.Tensor,
    value1: torch.Tensor,
    blend: torch.Tensor,
  ) -> torch.Tensor:
    blend = blend.unsqueeze(-1)
    return value0 * (1.0 - blend) + value1 * blend

  def clamp_to_joint_limits(self, joint_limits: torch.Tensor) -> None:
    if joint_limits.shape != (len(C.JOINT_NAMES), 2):
      raise ValueError(f"Expected hard joint limits [27,2], got {joint_limits.shape}")
    clamped = torch.minimum(
      torch.maximum(self.joint_pos, joint_limits[:, 0]), joint_limits[:, 1]
    )
    delta = torch.abs(clamped - self.joint_pos)
    changed = delta > 0.0
    if torch.any(changed):
      print(
        f"  [limits] clamped {int(changed.sum().item())} values; "
        f"max correction={float(delta.max().item()):.6f} rad"
      )
      self.joint_pos = clamped
      self.joint_vel = torch.gradient(
        self.joint_pos, spacing=self.output_dt, dim=0
      )[0]

  def _report_recovery_coverage(self, *, require: bool) -> None:
    w, x, y, z = self.root_quat.unbind(dim=-1)
    roll = torch.atan2(
      2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)
    )
    pitch = torch.asin(
      torch.clamp(2.0 * (w * y - z * x), min=-1.0, max=1.0)
    )
    root_z = self.root_pos[:, 2]
    fallen = (root_z < 0.45) & (
      (torch.abs(roll) > 1.0) | (torch.abs(pitch) > 1.0)
    )
    upright = (root_z > 0.70) & (
      (torch.abs(roll) < 0.3) & (torch.abs(pitch) < 0.3)
    )
    num_fallen = int(fallen.sum().item())
    num_upright = int(upright.sum().item())
    print(
      f"  [recovery] fallen frames={num_fallen}, upright frames={num_upright}, "
      f"root z=[{float(root_z.min().item()):.3f}, "
      f"{float(root_z.max().item()):.3f}] m"
    )
    if require and (num_fallen == 0 or num_upright == 0):
      raise ValueError(
        "Recovery clip must contain both fallen (z<0.45, tilt>1.0) and "
        "upright (z>0.70, tilt<0.3) frames"
      )


def convert_file(
  sim,
  scene,
  input_file: Path,
  output_path: Path,
  input_fps: float,
  output_fps: float,
  overwrite: bool,
  clamp_joint_limits: bool,
  require_recovery_coverage: bool = True,
) -> None:
  if output_path.exists() and not overwrite:
    print(f"[SKIP] {output_path} already exists (use --overwrite True)")
    return

  motion = Casbot02AmpjCsvMotion(
    input_file=input_file,
    input_fps=input_fps,
    output_fps=output_fps,
    device=sim.device,
    require_recovery_coverage=require_recovery_coverage,
  )
  robot: Entity = scene["robot"]
  if clamp_joint_limits:
    motion.clamp_to_joint_limits(robot.data.joint_pos_limits[0])
  joint_ids, matched_names = robot.find_joints(
    C.JOINT_NAMES, preserve_order=True
  )
  if tuple(matched_names) != C.JOINT_NAMES:
    raise RuntimeError(f"Failed to resolve all 27 target joints: {matched_names}")

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
  input_dir: str = str(DEFAULT_RECOVERY_DIR),
  output_dir: str = str(DEFAULT_RECOVERY_DIR),
  output_name: str | None = None,
  input_fps: float = INPUT_FPS,
  output_fps: float = OUTPUT_FPS,
  device: str = "cpu",
  overwrite: bool = False,
  clamp_joint_limits: bool = True,
  require_recovery_coverage: bool = True,
) -> None:
  """Convert one recovery CSV or every CSV in the Recovery directory."""
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
    csv_files = sorted(source_dir.glob("*.csv"))
    if not csv_files:
      raise FileNotFoundError(f"No CSV files found in {source_dir}")
    file_pairs = tuple(
      (source, output_root / source.with_suffix(".npz").name)
      for source in csv_files
    )
    print(f"Found {len(file_pairs)} recovery CSV file(s) in {source_dir}")

  sim, scene = _build_fk_scene(device=device, output_fps=int(output_fps))
  for index, (source, destination) in enumerate(file_pairs, start=1):
    print(f"\n[{index}/{len(file_pairs)}] {source.name}")
    convert_file(
      sim=sim,
      scene=scene,
      input_file=source,
      output_path=destination,
      input_fps=input_fps,
      output_fps=output_fps,
      overwrite=overwrite,
      clamp_joint_limits=clamp_joint_limits,
      require_recovery_coverage=require_recovery_coverage,
    )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
