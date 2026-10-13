"""G1 contact shaping, phase reflection and deployment observation checks."""

from collections import deque
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace as NS
import importlib.util

import numpy as np
import torch
import yaml

from wbc_mjlab.casbot02_config import casbot02_loco_teacher_env_cfg
from wbc_mjlab.casbot02_symmetry import mirror_loco_observation
from wbc_mjlab.observations import loco_phase_features
from wbc_mjlab.rewards import gait_phase_contact


def test_rewards_use_g1_gates_with_casbot_height_targets():
  root = Path(__file__).resolve().parents[1]
  path = next((root / "logs/rsl_rl/g1_loco_teacher_nobv_stable_seed").glob("*/params/env.yaml"))
  reference = yaml.load(path.read_text(), Loader=yaml.BaseLoader)["rewards"]
  cfg = casbot02_loco_teacher_env_cfg()
  assert "swing_height_curve" not in cfg.rewards
  for name in ("air_time", "turning_air_time", "feet_air_time"):
    assert cfg.rewards[name].weight == 0.0
  for name in ("foot_clearance", "foot_swing_height", "gait_phase_contact"):
    term = cfg.rewards[name]
    assert term.weight == float(reference[name]["weight"])
    for key, value in reference[name]["params"].items():
      if key not in ("asset_cfg", "target_height"):
        assert str(term.params[key]) == value
  # Height targets have been tuned for the larger Casbot02 feet.
  assert cfg.rewards["foot_clearance"].params["target_height"] == 0.09
  assert cfg.rewards["foot_swing_height"].params["target_height"] == 0.12
  assert cfg.rewards["foot_clearance"].params["asset_cfg"].site_names == ("left_foot", "right_foot")


def test_contact_reward_prefers_alternation_and_double_support_when_standing():
  # Phase .25: left stance, right swing. Phase .75 reverses the roles.
  command = NS(is_standing_env=torch.tensor([False, False, False, True]))
  force = torch.tensor([
    [[0., 0., 100.], [0., 0., 100.]],
    [[0., 0., 100.], [0., 0., 0.]],
    [[0., 0., 0.], [0., 0., 100.]],
    [[0., 0., 100.], [0., 0., 100.]],
  ])
  env = NS(episode_length_buf=torch.tensor([25, 25, 75, 25]), step_dt=.01,
           command_manager=NS(get_term=lambda _: command),
           scene={"feet": NS(data=NS(force=force))})
  torch.testing.assert_close(.5 * gait_phase_contact(env, "feet"), torch.tensor([.5, 1., 1., 1.]))


def test_phase_layout_and_symmetry_are_consistent():
  cfg = casbot02_loco_teacher_env_cfg()
  for name in ("actor", "critic"):
    group = cfg.observations[name]
    assert group.history_length is None
    assert group.terms["phase"].history_length == 0
    assert all(v.history_length == 4 for k, v in group.terms.items() if k != "phase")
  for size in (180, 192, 184, 196):
    obs = torch.arange(size, dtype=torch.float32).unsqueeze(0)
    mirrored = mirror_loco_observation(obs)
    torch.testing.assert_close(mirror_loco_observation(mirrored), obs)
    if size in (184, 196):
      torch.testing.assert_close(mirrored[:, 180:184], obs[:, [182, 183, 180, 181]])
  assert asdict(cfg.observations["actor"].terms["phase"])["noise"] is None


def test_sim2sim_phase_matches_training_and_preserves_legacy_input():
  root = Path(__file__).resolve().parents[1]
  spec = importlib.util.spec_from_file_location("casbot_sim", root / "scripts/sim2sim_casbot02_loco_onnx.py")
  sim = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(sim)
  for step, standing in ((0, False), (25, False), (75, False), (125, False), (25, True)):
    env = NS(episode_length_buf=torch.tensor([step]), step_dt=.01,
             command_manager=NS(get_term=lambda _: NS(is_standing_env=torch.tensor([standing]))))
    cmd = np.array([0., 0., 0. if standing else .5])
    phase = sim.gait_phase_features(step * .01, cmd)
    np.testing.assert_allclose(phase, loco_phase_features(env).numpy()[0], atol=1e-6)
    history = deque([np.zeros(45, dtype=np.float32)] * 4)
    assert sim.build_obs_term_major(history).shape == (1, 180)
    obs = sim.build_obs_term_major(history, phase)
    assert obs.shape == (1, 184)
    np.testing.assert_array_equal(obs[0, 180:], phase)
