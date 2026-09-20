"""Command lifecycle checks without running physics."""

from types import SimpleNamespace

import torch

from wbc_mjlab.casbot02_config import casbot02_loco_teacher_env_cfg


def make_command(n=1024):
  cfg = casbot02_loco_teacher_env_cfg().commands["twist"]
  cfg.rel_standing_envs = 0.0
  cfg.rel_turning_envs = 0.0
  cfg.rel_forward_envs = 0.0
  cfg.rel_heading_envs = 0.0
  cfg.rel_startup_envs = 1.0
  cfg.init_velocity_prob = 0.0
  robot = SimpleNamespace(data=SimpleNamespace(
    heading_w=torch.zeros(n),
    root_link_lin_vel_b=torch.zeros(n, 3),
    root_link_ang_vel_b=torch.zeros(n, 3),
  ))
  env = SimpleNamespace(num_envs=n, device="cpu", step_dt=0.02,
                        scene={"robot": robot})
  return cfg.build(env)


def test_startup_holds_then_walks_both_directions_and_resets_subset():
  torch.manual_seed(42)
  cmd = make_command()
  ids = torch.arange(cmd.num_envs)
  cmd.reset(ids)
  assert cmd.is_startup_env.all() and cmd.is_standing_env.all()
  assert torch.count_nonzero(cmd.command) == 0
  hold = cmd.time_left - cmd.startup_walk_duration
  assert ((hold >= 2) & (hold <= 3)).all()
  assert ((cmd.startup_walk_duration >= 3) & (cmd.startup_walk_duration <= 4)).all()
  target = cmd.startup_target_vx.clone()
  assert ((target.abs() >= 0.3) & (target.abs() <= 0.6)).all()
  assert 0.45 < (target > 0).float().mean() < 0.55
  for _ in range(99):
    cmd.compute(0.02)
  assert cmd.is_standing_env.all()
  assert torch.count_nonzero(cmd.command) == 0
  for _ in range(55):
    cmd.compute(0.02)
  assert not cmd.is_standing_env.any()
  torch.testing.assert_close(cmd.command[:, 0], target)
  assert torch.count_nonzero(cmd.command[:, 1:]) == 0
  # A reset must restart the hold, without disturbing other environments.
  cmd.reset(ids[:16])
  assert cmd.is_standing_env[:16].all()
  assert torch.count_nonzero(cmd.command[:16]) == 0
  torch.testing.assert_close(cmd.command[16:, 0], target[16:])


def test_startup_respects_warmup_and_clears_state_when_resampled_out():
  cmd = make_command(64)
  cmd.cfg.ranges.lin_vel_x = (-0.3, 0.3)
  ids = torch.arange(cmd.num_envs)
  cmd.reset(ids)
  torch.testing.assert_close(cmd.startup_target_vx.abs(), torch.full((64,), 0.3))
  cmd.cfg.rel_startup_envs = 0.0
  cmd.cfg.rel_standing_envs = 1.0
  cmd._resample(ids)
  cmd._update_command()
  assert not cmd.is_startup_env.any()
  assert torch.count_nonzero(cmd.startup_target_vx) == 0
  assert cmd.is_standing_env.all()
  assert torch.count_nonzero(cmd.command) == 0
