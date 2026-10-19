"""G1 contact shaping, phase reflection and deployment observation checks."""

from collections import deque
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace as NS
import importlib.util

import numpy as np
import torch
import yaml
from tensordict import TensorDict
from mjlab.managers.observation_manager import ObservationManager

from wbc_mjlab.casbot02_config import casbot02_loco_teacher_env_cfg, casbot02_student_env_cfg
from wbc_mjlab.casbot02_symmetry import compute_symmetric_states, mirror_loco_observation
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
  for name in ("foot_clearance", "foot_swing_height"):
    term = cfg.rewards[name]
    assert term.weight == float(reference[name]["weight"])
    for key, value in reference[name]["params"].items():
      if key not in ("asset_cfg", "target_height"):
        assert str(term.params[key]) == value
  # Height targets have been tuned for the larger Casbot02 feet.
  assert cfg.rewards["foot_clearance"].params["target_height"] == 0.07
  assert cfg.rewards["foot_swing_height"].params["target_height"] == 0.10
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
    assert all(v.history_length == 4 for v in group.terms.values())
  for size, length in ((180, None), (192, None), (184, 0), (196, 0), (196, 4), (208, 4)):
    obs = torch.arange(size, dtype=torch.float32).unsqueeze(0)
    mirrored = mirror_loco_observation(obs, phase_history_length=length)
    torch.testing.assert_close(mirror_loco_observation(mirrored, phase_history_length=length), obs)
    if length is not None:
      end = 180 + 4 * max(1, length)
      expected = obs[:, 180:end].reshape(1, -1, 4)[..., [2, 3, 0, 1]].flatten(1)
      torch.testing.assert_close(mirrored[:, 180:end], expected)
      if size == end + 12:
        expected_velocity = obs[:, end:].reshape(1, 4, 3) * torch.tensor([1., -1., 1.])
        torch.testing.assert_close(mirrored[:, end:], expected_velocity.flatten(1))
  assert asdict(cfg.observations["actor"].terms["phase"])["noise"] is None
  student = casbot02_student_env_cfg()
  assert "phase" not in student.observations["actor"].terms


def test_phase_augmentation_disambiguates_old_critic_from_new_actor():
  cfg = casbot02_loco_teacher_env_cfg()
  for actor_size, critic_size, length in ((196, 208, 4), (184, 196, 0)):
    for group in cfg.observations.values():
      group.terms["phase"].history_length = length
    obs = TensorDict({"actor": torch.arange(actor_size).float().reshape(1, -1),
                      "critic": torch.arange(critic_size).float().reshape(1, -1)}, batch_size=[1])
    augmented, _ = compute_symmetric_states(NS(cfg=cfg), obs=obs)
    for key in obs.keys():
      torch.testing.assert_close(augmented[key][:1], obs[key])
      torch.testing.assert_close(augmented[key][1:], mirror_loco_observation(
        obs[key], phase_history_length=length))


def test_sim2sim_phase_matches_training_and_preserves_legacy_input():
  root = Path(__file__).resolve().parents[1]
  spec = importlib.util.spec_from_file_location("casbot_sim", root / "scripts/sim2sim_casbot02_loco_onnx.py")
  sim = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(sim)
  cfg = casbot02_loco_teacher_env_cfg()
  reward = cfg.rewards["gait_phase_contact"]
  params = cfg.observations["actor"].terms["phase"].params
  assert params == cfg.observations["critic"].terms["phase"].params
  assert params["gait_period"] == reward.params["gait_period"]
  assert params["gait_offset"] == reward.params["gait_offset"] == 0.5
  assert reward.weight == 1.0
  assert params["gait_period"] == 1.2
  assert reward.params["stance_ratio"] == 0.5
  assert np.isclose(params["gait_period"] * reward.params["stance_ratio"], 0.6)
  assert np.isclose(params["gait_period"] * (1 - reward.params["stance_ratio"]), 0.6)
  for step, standing in ((0, False), (25, False), (75, False), (125, False),
                         (200, False), (400, False), (25, True)):
    env = NS(episode_length_buf=torch.tensor([step]), step_dt=.01,
             command_manager=NS(get_term=lambda _: NS(is_standing_env=torch.tensor([standing]))))
    cmd = np.array([0., 0., 0. if standing else .5])
    phase = sim.gait_phase_features(step * .01, cmd)
    np.testing.assert_allclose(phase, loco_phase_features(env, **params).numpy()[0], atol=1e-6)
    legacy_phase = sim.gait_phase_features(step * .01, cmd, gait_period=1.0)
    np.testing.assert_allclose(legacy_phase, loco_phase_features(env).numpy()[0], atol=1e-6)
    history = deque([np.zeros(45, dtype=np.float32)] * 4)
    assert sim.build_obs_term_major(history).shape == (1, 180)
    obs = sim.build_obs_term_major(history, phase)
    assert obs.shape == (1, 184)
    np.testing.assert_array_equal(obs[0, 180:], phase)


def _frame_term(env, name):
  return env.frames[name]


def test_phase_history_sim2sim_matches_observation_manager_and_reset():
  root = Path(__file__).resolve().parents[1]
  spec = importlib.util.spec_from_file_location("casbot_sim", root / "scripts/sim2sim_casbot02_loco_onnx.py")
  sim = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(sim)
  cfg = casbot02_loco_teacher_env_cfg(play=True)
  dims = dict(base_ang_vel=3, projected_gravity=3, command=3,
              joint_pos=12, joint_vel=12, actions=12, phase=4, base_lin_vel=3)
  env = NS(num_envs=1, device="cpu", frames={name: torch.zeros(1, d) for name, d in dims.items()})
  # Keep the real configured term order, history lengths and flattening;
  # supply deterministic frames without requiring a physics environment.
  for group in cfg.observations.values():
    for name, term in group.terms.items():
      term.func = _frame_term
      term.params = {"name": name}
  manager = ObservationManager(cfg.observations, env)
  history = deque(maxlen=4)
  phase_history = deque(maxlen=4)
  for step in range(9):
    if step == 6:
      manager.reset(torch.tensor([0]))
      history.clear()
      phase_history.clear()
    command = np.array([.5 if 1 <= step < 5 else 0., 0., 0.])
    phase = sim.gait_phase_features(step * .02, command)
    frame = np.arange(45, dtype=np.float32) + step * 100
    frame[6:9] = command
    offset = 0
    for name, width in list(dims.items())[:6]:
      env.frames[name] = torch.from_numpy(frame[offset:offset + width].copy()).reshape(1, -1)
      offset += width
    env.frames["phase"] = torch.from_numpy(phase).reshape(1, -1)
    if not history:
      history.extend([frame] * 4)
      phase_history.extend([phase] * 4)
    else:
      history.append(frame)
      phase_history.append(phase)
    actual = manager.compute(update_history=True)
    deployed = sim.build_obs_term_major(history, np.asarray(phase_history))
    assert actual["actor"].shape == (1, 196)
    assert actual["critic"].shape == (1, 208)
    np.testing.assert_allclose(deployed, actual["actor"].numpy(), atol=1e-6)
    np.testing.assert_allclose(deployed[0, 180:].reshape(4, 4), np.asarray(phase_history))
    np.testing.assert_allclose(actual["critic"][:, :196].numpy(), deployed)
