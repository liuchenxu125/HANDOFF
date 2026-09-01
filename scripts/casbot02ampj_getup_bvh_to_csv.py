"""Crop LAFAN1 get-up clips and batch-retarget them to CASBOT02AMPJ 27DoF.

Run this script in the GMR environment.  It is headless and does not construct
the GMR viewer.  Output CSV layout is root xyz, root quaternion xyzw, followed
by all 27 policy joints in ``casbot02ampj.constants.JOINT_NAMES`` order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GMR_ROOT = Path.home() / "Desktop" / "GMR"
DEFAULT_RAW_DIR = DEFAULT_GMR_ROOT / "assets" / "lafan1"
DEFAULT_CROPPED_DIR = DEFAULT_RAW_DIR / "amp"
DEFAULT_OUTPUT_DIR = (
  REPO_ROOT
  / "src"
  / "wbc_mjlab"
  / "data"
  / "motions"
  / "casbot02ampj"
  / "retarget_csv"
  / "GetUp"
)
MANIFEST_PATH = (
  REPO_ROOT
  / "src"
  / "wbc_mjlab"
  / "data"
  / "motions"
  / "casbot02ampj"
  / "amp"
  / "GetUp"
  / "manifest.json"
)
ROBOT_XML = (
  REPO_ROOT
  / "src"
  / "wbc_mjlab"
  / "CASBOT02_ENCOS_7dof_shell_20251015"
  / "Serial"
  / "xml"
  / "CASBOT02AMPJ_27dof_gmr.xml"
)
IK_CONFIG = (
  REPO_ROOT
  / "src"
  / "wbc_mjlab"
  / "casbot02ampj"
  / "retarget"
  / "bvh_lafan1_to_casbot02ampj_27dof.json"
)
ROBOT_KEY = "casbot02ampj_27dof"

JOINT_NAMES = (
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


def _crop_bvh(source: Path, output: Path, start: int, end: int) -> None:
  """Keep a zero-indexed inclusive frame interval and update ``Frames:``."""
  lines = source.read_text(encoding="utf-8").splitlines()
  motion_index = next(i for i, line in enumerate(lines) if line.strip() == "MOTION")
  frames_index = next(
    i for i in range(motion_index + 1, len(lines))
    if lines[i].strip().startswith("Frames:")
  )
  frame_time_index = next(
    i for i in range(frames_index + 1, len(lines))
    if lines[i].strip().startswith("Frame Time:")
  )
  total_frames = int(lines[frames_index].split(":", 1)[1].strip())
  frame_lines = [line for line in lines[frame_time_index + 1 :] if line.strip()]
  if len(frame_lines) != total_frames:
    raise ValueError(
      f"{source}: header says {total_frames} frames, found {len(frame_lines)}"
    )
  if start < 0 or end < start or end >= total_frames:
    raise ValueError(f"{source}: invalid inclusive interval [{start}, {end}]")
  selected = frame_lines[start : end + 1]
  output.parent.mkdir(parents=True, exist_ok=True)
  content = (
    lines[:frames_index]
    + [f"Frames: {len(selected)}", lines[frame_time_index]]
    + selected
  )
  output.write_text("\n".join(content) + "\n", encoding="utf-8")
  print(f"[crop] {source.name}[{start}:{end}] -> {output.name} ({len(selected)} frames)")


def _load_gmr(gmr_root: Path):
  sys.path.insert(0, str(gmr_root))
  from general_motion_retargeting import params
  from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting
  from general_motion_retargeting.utils.lafan1 import load_bvh_file

  params.ROBOT_XML_DICT[ROBOT_KEY] = ROBOT_XML
  params.IK_CONFIG_DICT.setdefault("bvh_lafan1", {})[ROBOT_KEY] = IK_CONFIG
  return GeneralMotionRetargeting, load_bvh_file


def _retarget_clip(
  bvh_path: Path,
  output_path: Path,
  retargeter_cls,
  load_bvh_file,
) -> None:
  frames, human_height = load_bvh_file(str(bvh_path), format="lafan1")
  retargeter = retargeter_cls(
    src_human="bvh_lafan1",
    tgt_robot=ROBOT_KEY,
    actual_human_height=human_height,
    verbose=False,
  )
  model = retargeter.model
  joint_qpos_addresses = []
  for name in JOINT_NAMES:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
      raise ValueError(f"27DoF GMR model is missing joint {name!r}")
    joint_qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
  if model.nq != 7 + len(JOINT_NAMES):
    raise ValueError(f"Expected nq=34 (free root + 27 hinges), got {model.nq}")

  rows = []
  for human_frame in tqdm(frames, desc=bvh_path.stem, unit="frame"):
    qpos = retargeter.retarget(human_frame)
    root_xyzw = qpos[3:7][[1, 2, 3, 0]]
    rows.append(
      np.concatenate((qpos[:3], root_xyzw, qpos[joint_qpos_addresses]))
    )
  motion = np.asarray(rows, dtype=np.float64)
  if motion.shape != (len(frames), 34) or not np.isfinite(motion).all():
    raise ValueError(f"Invalid retarget output shape/content: {motion.shape}")
  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.savetxt(output_path, motion, delimiter=",", fmt="%.10f")
  print(f"[OK] {bvh_path.name} -> {output_path} {motion.shape}")


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--gmr-root", type=Path, default=DEFAULT_GMR_ROOT)
  parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
  parser.add_argument("--cropped-dir", type=Path, default=DEFAULT_CROPPED_DIR)
  parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
  parser.add_argument("--overwrite", action="store_true")
  parser.add_argument(
    "--allow-missing",
    action="store_true",
    help="Convert available clips and report missing sources instead of failing first.",
  )
  args = parser.parse_args()

  for required in (ROBOT_XML, IK_CONFIG, MANIFEST_PATH):
    if not required.is_file():
      raise FileNotFoundError(required)
  manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

  prepared: list[tuple[str, Path]] = []
  missing: list[str] = []
  for clip in manifest["clips"]:
    name = clip["name"]
    cropped = args.cropped_dir.expanduser() / f"{name}.bvh"
    if not cropped.is_file():
      source = args.raw_dir.expanduser() / clip["source"]
      if source.is_file():
        _crop_bvh(source, cropped, int(clip["start"]), int(clip["end"]))
      else:
        missing.append(f"{name}: source {source}")
        continue
    prepared.append((name, cropped))

  if missing and not args.allow_missing:
    print("Missing source BVH files:")
    for item in missing:
      print(f"  - {item}")
    print("Re-run with --allow-missing to process the available subset.")
    return 2

  retargeter_cls, load_bvh_file = _load_gmr(args.gmr_root.expanduser())
  output_dir = args.output_dir.expanduser()
  for name, bvh_path in prepared:
    output_path = output_dir / f"{name}.csv"
    if output_path.exists() and not args.overwrite:
      print(f"[skip] {output_path} (use --overwrite)")
      continue
    _retarget_clip(
      bvh_path, output_path, retargeter_cls, load_bvh_file
    )

  if missing:
    print("\nConverted the available subset; still missing:")
    for item in missing:
      print(f"  - {item}")
    return 2
  print(f"\nRetargeted all {len(prepared)} positive-weight get-up clips.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
