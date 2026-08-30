"""Export a Casbot02 task's actor to ONNX (loco teacher / distill student).

Usage:
    PYTHONPATH=. uv run python scripts/export_casbot02_onnx.py \
        Casbot02-Loco-Teacher-Flat \
        logs/rsl_rl/casbot02_loco_teacher/<run>/model_20000.pt

Writes ``<run>/<run-dir-name>.onnx`` next to the checkpoint.
"""
from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))


def main() -> None:
  if len(sys.argv) < 3:
    print("Usage: python export_casbot02_onnx.py <task_id> <checkpoint_path>")
    sys.exit(1)
  task_id = sys.argv[1]
  checkpoint_path = sys.argv[2]

  if not Path(checkpoint_path).exists():
    print(f"Error: checkpoint not found: {checkpoint_path}")
    sys.exit(1)

  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper

  env_cfg = load_env_cfg(task_id, play=True)
  rl_cfg = load_rl_cfg(task_id)
  runner_cls = load_runner_cls(task_id)
  rl_dict = asdict(rl_cfg)

  env_cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu")
  env = RslRlVecEnvWrapper(env)

  print(f"Instantiating runner: {runner_cls.__name__}")
  runner = runner_cls(env, rl_dict, device="cpu")
  runner.load(checkpoint_path)

  export_dir = Path(checkpoint_path).parent
  filename = f"{export_dir.name}.onnx"
  print(f"Exporting to {export_dir / filename}")
  runner.export_policy_to_onnx(str(export_dir), filename=filename)
  print("Done!")


if __name__ == "__main__":
  main()
