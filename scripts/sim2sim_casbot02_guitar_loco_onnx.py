"""CASBOT02-Guitar sim2sim: same policy/observations, correct rigid payload."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from scripts import sim2sim_casbot02_loco_onnx as base
from wbc_mjlab.casbot02_guitar.constants import get_robot_cfg
from wbc_mjlab.casbot02_guitar.motor_config import LEG_ACTION_SCALE


def build_model():
  return base.build_model(get_robot_cfg())


def main():
  args = base.parse_args()
  model_path = args.model_path or str(base.find_latest_onnx("casbot02_guitar_loco_teacher"))
  base.run(
    model_path, args.log_csv, args.log_decimation, args.gait_period,
    robot_cfg=get_robot_cfg(),
    action_scale=np.array([LEG_ACTION_SCALE[n] for n in base.CASBOT02_LEG_ONLY_JOINT_NAMES]),
    arm_swing_gain=0.0,
  )


if __name__ == "__main__":
  main()
