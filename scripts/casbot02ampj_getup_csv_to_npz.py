"""Convert direct 27DoF GMR get-up CSV files into 50 Hz AMP NPZ files."""

from __future__ import annotations

from pathlib import Path

import mjlab
import tyro

from scripts.casbot02ampj_csv_to_npz import main as convert_csv


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = (
  REPO_ROOT
  / "src"
  / "wbc_mjlab"
  / "data"
  / "motions"
  / "casbot02ampj"
  / "retarget_csv"
  / "GetUp"
)
DEFAULT_OUTPUT_DIR = (
  REPO_ROOT
  / "src"
  / "wbc_mjlab"
  / "data"
  / "motions"
  / "casbot02ampj"
  / "amp"
  / "GetUp"
)


def main(
  input_dir: str = str(DEFAULT_INPUT_DIR),
  output_dir: str = str(DEFAULT_OUTPUT_DIR),
  device: str = "cpu",
  overwrite: bool = False,
  clamp_joint_limits: bool = True,
) -> None:
  convert_csv(
    input_dir=input_dir,
    output_dir=output_dir,
    input_fps=30.0,
    output_fps=50.0,
    device=device,
    overwrite=overwrite,
    clamp_joint_limits=clamp_joint_limits,
    require_recovery_coverage=False,
  )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
