"""CPU checks for the stand-then-walk command lifecycle (no physics engine)."""
import unittest
from types import SimpleNamespace

import torch

from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg
from wbc_mjlab.casbot02_commands import Casbot02VelocityCommand, Casbot02VelocityCommandCfg
from wbc_mjlab.casbot02_config import casbot02_loco_teacher_env_cfg


class StartupCommandTest(unittest.TestCase):
  def make_command(self, count=10000, only_startup=True):
    # Test the retained startup helper independently of the active G1 sampler.
    cfg = Casbot02VelocityCommandCfg(
      entity_name='robot', resampling_time_range=(3., 8.),
      heading_command=True, rel_standing_envs=.1, rel_heading_envs=.2,
      rel_forward_envs=.2, rel_turning_envs=.2, rel_startup_envs=.1,
      ranges=Casbot02VelocityCommandCfg.Ranges(
        lin_vel_x=(-1., 1.), lin_vel_y=(0., 0.), ang_vel_z=(-1., 1.),
        heading=(-torch.pi, torch.pi),
      ),
    )
    if only_startup:
      cfg.rel_standing_envs = cfg.rel_turning_envs = 0.0
      cfg.rel_forward_envs = cfg.rel_heading_envs = 0.0
      cfg.rel_startup_envs = 1.0
    cfg.init_velocity_prob = 1.0  # Prove exclusion even with injection enabled.
    writes = []
    quat = torch.zeros(count, 4)
    quat[:, 0] = 1
    robot = SimpleNamespace(
      data=SimpleNamespace(
        root_link_pos_w=torch.zeros(count, 3), root_link_quat_w=quat,
        root_link_lin_vel_b=torch.zeros(count, 3),
        root_link_ang_vel_b=torch.zeros(count, 3), heading_w=torch.zeros(count),
      ),
      write_root_state_to_sim=lambda state, ids: writes.append(ids.clone()),
    )
    env = SimpleNamespace(num_envs=count, device='cpu', step_dt=.02,
                          scene={'robot': robot})
    return Casbot02VelocityCommand(cfg, env), writes

  def test_full_lifecycle(self):
    torch.manual_seed(42)
    cmd, writes = self.make_command()
    ids = torch.arange(cmd.num_envs)
    cmd.reset(ids)
    total = cmd.time_left.clone()
    stand = total - cmd.startup_walk_duration
    self.assertTrue(((stand >= 2) & (stand <= 3)).all())
    self.assertTrue(((cmd.startup_walk_duration >= 3) & (cmd.startup_walk_duration <= 4)).all())
    self.assertTrue(cmd.is_standing_env.all())
    self.assertTrue((cmd.command == 0).all())
    # Speed is locked in when standing starts: 0.3 ~ min(0.6, curriculum).
    sampled = cmd.startup_target_vx.clone()
    self.assertTrue((sampled.abs() >= .3).all())
    self.assertTrue((sampled.abs() <= .6).all())
    self.assertLess(abs((sampled > 0).float().mean().item() - .5), .02)
    # A later curriculum change must not rewrite the already-sampled launch speed.
    cmd.cfg.ranges.lin_vel_x = (-.4, .5)
    elapsed = 0.0
    started = torch.full_like(total, -1)
    for _ in range(351):
      cmd.compute(.02)
      elapsed += .02
      first = (started < 0) & ~cmd.is_standing_env & (cmd.command_counter == 1)
      started[first] = elapsed
      walking = ~cmd.is_standing_env
      self.assertTrue(torch.equal(cmd.command[walking, 0], sampled[walking]))
      self.assertTrue((cmd.command[:, 1:] == 0).all())
      self.assertTrue((cmd.command[cmd.is_standing_env] == 0).all())
      if elapsed < 5:
        self.assertTrue((cmd.command_counter == 1).all())
      self.assertTrue((cmd.command_counter[total > elapsed + .001] == 1).all())
      self.assertTrue((cmd.command_counter[total < elapsed - .001] == 2).all())
    self.assertTrue((started >= stand - 1e-4).all())
    self.assertTrue((started <= stand + .021).all())
    self.assertTrue((cmd.command_counter == 2).all())
    self.assertEqual(writes, [])

  def test_mixed_sampling_and_reset(self):
    torch.manual_seed(19)
    cmd, writes = self.make_command(30000, only_startup=False)
    cmd.reset(torch.arange(cmd.num_envs))
    startup = cmd.is_startup_env
    self.assertLess(abs(startup.float().mean().item() - .1), .01)
    self.assertLess(abs(cmd.is_heading_env.float().mean().item() - .2), .01)
    self.assertFalse((startup & (cmd.is_heading_env | cmd.is_forward_env |
                                cmd.is_turning_env | cmd.is_world_env)).any())
    self.assertFalse(startup[torch.cat(writes)].any())
    ids = startup.nonzero().flatten()[:5]
    untouched = torch.ones(cmd.num_envs, dtype=torch.bool)
    untouched[ids] = False
    saved = cmd.command[untouched].clone()
    # Episode reset must cancel old startup state when sampled into another mode.
    cmd.cfg.rel_startup_envs = 0
    cmd.reset(ids)
    self.assertFalse(cmd.is_startup_env[ids].any())
    self.assertTrue((cmd.startup_walk_duration[ids] == 0).all())
    self.assertTrue((cmd.startup_target_vx[ids] == 0).all())
    self.assertTrue((cmd.command_counter[ids] == 1).all())
    self.assertTrue(torch.equal(cmd.command[untouched], saved))

  def test_small_curriculum_limit(self):
    cmd, writes = self.make_command(100)
    cmd.cfg.ranges.lin_vel_x = (-.1, .2)
    cmd.reset(torch.arange(cmd.num_envs))
    self.assertTrue((cmd.startup_target_vx >= -.1).all())
    self.assertTrue((cmd.startup_target_vx <= .2).all())
    cmd.time_left[:] = cmd.startup_walk_duration
    cmd._update_command()
    self.assertTrue((cmd.command[:, 0] >= -.1).all())
    self.assertTrue((cmd.command[:, 0] <= .2).all())
    self.assertEqual(writes, [])

  def test_curriculum_between_min_and_cap(self):
    cmd, writes = self.make_command(200)
    cmd.cfg.ranges.lin_vel_x = (-.5, .5)
    cmd.reset(torch.arange(cmd.num_envs))
    speed = cmd.startup_target_vx.abs()
    self.assertTrue((speed >= .3).all())
    self.assertTrue((speed <= .5).all())
    self.assertEqual(writes, [])

  def test_startup_speed_capped_at_max(self):
    cmd, writes = self.make_command(200)
    cmd.reset(torch.arange(cmd.num_envs))
    speed = cmd.startup_target_vx.abs()
    self.assertTrue((speed >= .3).all())
    self.assertTrue((speed <= .6).all())
    self.assertEqual(writes, [])

  def test_warmup_locks_to_curriculum_limit(self):
    cmd, writes = self.make_command(64)
    cmd.cfg.ranges.lin_vel_x = (-.3, .3)
    cmd.reset(torch.arange(cmd.num_envs))
    torch.testing.assert_close(cmd.startup_target_vx.abs(), torch.full((64,), .3))
    self.assertEqual(writes, [])


class G1CommandAllocationTest(unittest.TestCase):
  def test_active_sampler_matches_g1_and_preserves_speed_curriculum(self):
    cfg = casbot02_loco_teacher_env_cfg()
    command = cfg.commands['twist']
    self.assertIs(type(command), UniformVelocityCommandCfg)
    self.assertEqual(command.rel_standing_envs, .1)
    self.assertEqual(command.rel_heading_envs, .3)
    self.assertEqual(command.rel_forward_envs, .2)
    self.assertEqual(command.rel_world_envs, 0.)
    self.assertEqual(command.init_velocity_prob, 0.)
    self.assertEqual(command.resampling_time_range, (3., 8.))
    self.assertEqual(command.ranges.lin_vel_y, (0., 0.))
    stages = cfg.curriculum['command_vel'].params['velocity_stages']
    self.assertEqual([stage['step'] for stage in stages], [0, 60000, 120000, 192000])

  def test_native_sampling_has_independent_overlapping_modes(self):
    torch.manual_seed(42)
    count = 30000
    cfg = casbot02_loco_teacher_env_cfg().commands['twist']
    env = SimpleNamespace(
      num_envs=count, device='cpu', step_dt=.02,
      scene={'robot': SimpleNamespace(data=SimpleNamespace(heading_w=torch.zeros(count)))},
    )
    cmd = cfg.build(env)
    cmd.reset(torch.arange(count))
    for mask, expected in ((cmd.is_standing_env, .1), (cmd.is_heading_env, .3),
                           (cmd.is_forward_env, .2)):
      self.assertLess(abs(mask.float().mean().item() - expected), .01)
    overlap = cmd.is_forward_env & cmd.is_heading_env
    self.assertLess(abs(overlap.float().mean().item() - .06), .01)
    cmd._update_command()
    self.assertTrue((cmd.command[cmd.is_standing_env] == 0).all())
    forward_only = cmd.is_forward_env & ~cmd.is_heading_env & ~cmd.is_standing_env
    self.assertTrue((cmd.command[forward_only, 0] >= .3).all())
    self.assertTrue((cmd.command[forward_only, 1:] == 0).all())
    heading_forward = overlap & ~cmd.is_standing_env
    self.assertTrue((cmd.command[heading_forward, 2].abs() > 0).all())


if __name__ == '__main__':
  unittest.main()
