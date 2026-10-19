"""Export a Casbot02 task's actor to ONNX (loco teacher / distill student).

Usage:
    PYTHONPATH=. uv run python scripts/export_casbot02_onnx.py \
        Casbot02-Loco-Teacher-Flat \
        logs/rsl_rl/casbot02_loco_teacher/<run>/model_20000.pt

Writes ``<run>/<run-dir-name>.onnx`` next to the checkpoint.
Phase-enabled policies store ``gait_period`` from the saved ``params/env.yaml``
in ONNX metadata. For a standalone checkpoint, pass its training value with
``--gait-period``.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))


def positive_period(value: str | float) -> float:
  period = float(value)
  if not math.isfinite(period) or period <= 0:
    raise ValueError("gait_period must be finite and positive")
  return period


def checkpoint_gait_period(checkpoint: Path, env_cfg, override: float | None) -> float | None:
  if override is not None:
    return positive_period(override)
  snapshot = checkpoint.parent / "params" / "env.yaml"
  if snapshot.exists():
    # BaseLoader reads mjlab's Python-tagged YAML as plain data, without
    # constructing Python objects or importing the saved function references.
    saved = yaml.load(snapshot.read_text(), Loader=yaml.BaseLoader)
    phase = saved["observations"]["actor"]["terms"].get("phase")
    if phase is None:
      return None
    try:
      return positive_period(phase["params"]["gait_period"])
    except (KeyError, TypeError, ValueError) as exc:
      raise ValueError(f"Invalid/missing actor phase gait_period in {snapshot}; "
                       "pass the model's training value with --gait-period") from exc
  if "phase" in env_cfg.observations["actor"].terms:
    raise ValueError(f"Training snapshot missing: {snapshot}; "
                     "pass the model's training value with --gait-period")
  return None


def write_gait_period(model_path: Path, period: float) -> None:
  import onnx

  model = onnx.load(str(model_path))
  properties = {entry.key: entry.value for entry in model.metadata_props}
  properties["gait_period"] = str(positive_period(period))
  onnx.helper.set_model_props(model, properties)
  onnx.checker.check_model(model)
  onnx.save(model, str(model_path))


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("task_id")
  parser.add_argument("checkpoint_path", type=Path)
  parser.add_argument("--gait-period", type=positive_period,
                      help="Explicit training period, e.g. for a checkpoint without params/env.yaml")
  args = parser.parse_args()
  task_id = args.task_id
  checkpoint_path = args.checkpoint_path.expanduser().resolve()
  if not checkpoint_path.is_file():
    parser.error(f"Checkpoint not found: {checkpoint_path}")

  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper

  env_cfg = load_env_cfg(task_id, play=True)
  try:
    gait_period = checkpoint_gait_period(checkpoint_path, env_cfg, args.gait_period)
  except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as exc:
    parser.error(str(exc))
  rl_cfg = load_rl_cfg(task_id)
  runner_cls = load_runner_cls(task_id)
  rl_dict = asdict(rl_cfg)

  env_cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
  env = RslRlVecEnvWrapper(env)

  try:
    print(f"Instantiating runner: {runner_cls.__name__}")
    runner = runner_cls(env, rl_dict, device="cpu")
    runner.load(str(checkpoint_path), map_location="cpu")

    export_dir = checkpoint_path.parent
    filename = f"{export_dir.name}.onnx"
    model_path = export_dir / filename
    print(f"Exporting to {model_path}")
    runner.export_policy_to_onnx(str(export_dir), filename=filename)
    if gait_period is not None:
      write_gait_period(model_path, gait_period)
      print(f"ONNX metadata: gait_period={gait_period}")
  finally:
    env.close()
  print("Done!")


if __name__ == "__main__":
  main()
