"""Validate the active CASBOT02AMPJ get-up-only AMP dataset."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.verify_casbot02ampj_motion import (
  _model_contract,
  _roll_pitch_wxyz,
  _validate_clip,
)


GETUP_DIR = (
  Path(__file__).resolve().parents[1]
  / "src"
  / "wbc_mjlab"
  / "data"
  / "motions"
  / "casbot02ampj"
  / "amp"
  / "GetUp"
)


def main() -> int:
  manifest_path = GETUP_DIR / "manifest.json"
  manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
  expected_names = tuple(clip["name"] for clip in manifest["clips"])
  actual_files = sorted(GETUP_DIR.glob("*.npz"))
  actual_names = tuple(path.stem for path in actual_files)

  errors: list[str] = []
  missing = sorted(set(expected_names) - set(actual_names))
  extra = sorted(set(actual_names) - set(expected_names))
  if missing:
    errors.append(f"missing manifest clips: {missing}")
  if extra:
    errors.append(f"unexpected active clips: {extra}")

  _model, joint_ranges, body_names, amp_indices = _model_contract()
  print(f"[model] joints=27, bodies={len(body_names)}, AMP indices={amp_indices}")
  total_fallen = 0
  total_upright = 0
  for path in actual_files:
    clip_errors = _validate_clip(
      path,
      joint_ranges=joint_ranges,
      num_bodies=len(body_names),
      recovery=False,
    )
    errors.extend(f"{path.name}: {error}" for error in clip_errors)
    if not clip_errors:
      with np.load(path, allow_pickle=False) as clip:
        root_pos = np.asarray(clip["body_pos_w"])[:, 0]
        root_quat = np.asarray(clip["body_quat_w"])[:, 0]
        roll, pitch = _roll_pitch_wxyz(root_quat)
        fallen = (root_pos[:, 2] < 0.45) & (
          (np.abs(roll) > 1.0) | (np.abs(pitch) > 1.0)
        )
        upright = (root_pos[:, 2] > 0.70) & (
          (np.abs(roll) < 0.3) & (np.abs(pitch) < 0.3)
        )
        total_fallen += int(fallen.sum())
        total_upright += int(upright.sum())
        print(
          f"[ok] {path.name}: frames={clip['joint_pos'].shape[0]}, "
          f"fps={float(np.asarray(clip['fps']).item()):g}, "
          f"fallen={int(fallen.sum())}, upright={int(upright.sum())}"
        )

  if actual_files and total_fallen == 0:
    errors.append("dataset has no fallen frame (z<0.45 m and tilt>1.0 rad)")
  if actual_files and total_upright == 0:
    errors.append("dataset has no upright frame (z>0.70 m and tilt<0.3 rad)")

  if errors:
    print("\nGet-up dataset validation failed:")
    for error in errors:
      print(f"  - {error}")
    print(
      "Generate the dataset with scripts/casbot02ampj_getup_bvh_to_csv.py "
      "followed by scripts/casbot02ampj_getup_csv_to_npz.py."
    )
    return 1

  print(f"\nValidated all {len(actual_files)} get-up clips successfully.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
