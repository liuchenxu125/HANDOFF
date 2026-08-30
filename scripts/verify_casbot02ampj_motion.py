"""Validate CASBOT02 AMPJ NPZ files against the derived 27-DoF model."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from mjlab.entity import Entity

from wbc_mjlab.casbot02ampj import constants as C


DEFAULT_DATA_DIR = (
  Path(__file__).resolve().parents[1]
  / "src"
  / "wbc_mjlab"
  / "data"
  / "motions"
  / "casbot02ampj"
  / "amp"
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


def _roll_pitch_wxyz(quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  w, x, y, z = np.moveaxis(quat, -1, 0)
  roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
  pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
  return roll, pitch


def _model_contract():
  entity = Entity(C.get_robot_cfg())
  model = entity.spec.compile()
  joint_ranges = np.stack(
    [
      model.jnt_range[
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
      ]
      for name in C.JOINT_NAMES
    ]
  )
  body_names = tuple(model.body(i).name for i in range(1, model.nbody))
  amp_body_indices = tuple(body_names.index(name) for name in C.AMP_BODY_NAMES)
  return model, joint_ranges, body_names, amp_body_indices


def _validate_clip(
  path: Path,
  *,
  joint_ranges: np.ndarray,
  num_bodies: int,
  recovery: bool,
) -> list[str]:
  errors: list[str] = []
  try:
    data = np.load(path, allow_pickle=False)
  except Exception as exc:
    return [f"cannot load: {exc}"]

  missing = [key for key in REQUIRED_KEYS if key not in data]
  if missing:
    return [f"missing keys: {missing}"]

  fps = np.asarray(data["fps"])
  if fps.size != 1 or not np.isfinite(fps).all() or float(fps.item()) <= 0.0:
    errors.append(f"fps must be one positive finite scalar, got {fps}")

  joint_pos = np.asarray(data["joint_pos"])
  joint_vel = np.asarray(data["joint_vel"])
  if joint_pos.ndim != 2 or joint_pos.shape[1:] != (len(C.JOINT_NAMES),):
    errors.append(f"joint_pos expected [T,27], got {joint_pos.shape}")
    return errors
  num_frames = joint_pos.shape[0]
  if num_frames < 2:
    errors.append(f"clip must contain at least 2 frames, got {num_frames}")
  if joint_vel.shape != joint_pos.shape:
    errors.append(f"joint_vel expected {joint_pos.shape}, got {joint_vel.shape}")

  body_shapes = {
    "body_pos_w": (num_frames, num_bodies, 3),
    "body_quat_w": (num_frames, num_bodies, 4),
    "body_lin_vel_w": (num_frames, num_bodies, 3),
    "body_ang_vel_w": (num_frames, num_bodies, 3),
  }
  for key, expected in body_shapes.items():
    actual = np.asarray(data[key])
    if actual.shape != expected:
      errors.append(f"{key} expected {expected}, got {actual.shape}")

  for key in REQUIRED_KEYS[1:]:
    if not np.isfinite(np.asarray(data[key])).all():
      errors.append(f"{key} contains NaN or Inf")

  below = joint_pos < (joint_ranges[:, 0] - 1.0e-3)
  above = joint_pos > (joint_ranges[:, 1] + 1.0e-3)
  if np.any(below | above):
    frame, joint = np.argwhere(below | above)[0]
    errors.append(
      f"joint limit violation at frame {frame}, {C.JOINT_NAMES[joint]}="
      f"{joint_pos[frame, joint]:.5f}, range={joint_ranges[joint].tolist()}"
    )

  body_quat = np.asarray(data["body_quat_w"])
  if body_quat.shape == body_shapes["body_quat_w"]:
    quat_norm = np.linalg.norm(body_quat, axis=-1)
    max_norm_error = float(np.max(np.abs(quat_norm - 1.0)))
    if max_norm_error > 2.0e-2:
      errors.append(f"body quaternion max norm error is {max_norm_error:.5f}")

  if recovery and not errors:
    root_pos = np.asarray(data["body_pos_w"])[:, 0]
    root_quat = body_quat[:, 0]
    roll, pitch = _roll_pitch_wxyz(root_quat)
    fallen = (root_pos[:, 2] < 0.45) & (
      (np.abs(roll) > 1.0) | (np.abs(pitch) > 1.0)
    )
    upright = (root_pos[:, 2] > 0.70) & (
      (np.abs(roll) < 0.3) & (np.abs(pitch) < 0.3)
    )
    if not np.any(fallen):
      errors.append("Recovery clip has no frame matching z<0.45 m and tilt>1.0 rad")
    if not np.any(upright):
      errors.append("Recovery clip has no upright frame matching z>0.70 m and tilt<0.3 rad")

  return errors


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "data_dir",
    nargs="?",
    type=Path,
    default=DEFAULT_DATA_DIR,
    help="AMP root containing WalkandRun/ and Recovery/",
  )
  args = parser.parse_args()
  data_dir = args.data_dir.resolve()

  model, joint_ranges, body_names, amp_indices = _model_contract()
  print(
    f"[model] hinges={model.njnt - 1}, actions={model.nu}, "
    f"bodies={len(body_names)}, source={C.SOURCE_XML}"
  )
  print(f"[model] AMP body indices={amp_indices}")

  all_errors: list[str] = []
  total = 0
  for folder_name in ("WalkandRun", "Recovery"):
    folder = data_dir / folder_name
    # MotionLoader reads only direct children of each category directory.
    files = sorted(folder.glob("*.npz")) if folder.is_dir() else []
    if not files:
      all_errors.append(f"{folder}: no .npz files")
      continue
    for path in files:
      total += 1
      errors = _validate_clip(
        path,
        joint_ranges=joint_ranges,
        num_bodies=len(body_names),
        recovery=folder_name == "Recovery",
      )
      if errors:
        for error in errors:
          all_errors.append(f"{path}: {error}")
      else:
        with np.load(path, allow_pickle=False) as clip:
          print(
            f"[ok] {path.relative_to(data_dir)}: "
            f"frames={clip['joint_pos'].shape[0]}, fps={float(clip['fps']):g}"
          )

  if all_errors:
    print("\nValidation failed:")
    for error in all_errors:
      print(f"  - {error}")
    return 1

  print(f"\nValidated {total} motion clips successfully.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
