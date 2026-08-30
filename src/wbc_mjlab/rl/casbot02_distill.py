"""Casbot02 two-teacher distillation PPO.

This is the distillation "glue" that plugs the explicit-blend KL loss
(``two_teacher_blend``) into an RSL-RL PPO update. The student actor/critic are
standard MLPs; two frozen teachers (AMP for turn, loco for forward/back) provide
per-dim Gaussian targets. Each update:

    loss = PPO(surrogate + value_loss - entropy)
         + gate_loco(cmd) * coef_loco * KL(pi_s || pi_loco)
         + gate_amp (cmd) * coef_amp  * KL(pi_s || pi_amp)

The gate is a function of the current velocity command (vx, vy, wz), which the
runner exposes via a dedicated ``command`` obs group ([B, 3]) on the student env.
"""

from __future__ import annotations

import math

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups

from wbc_mjlab.rl.frozen_teacher import FrozenRslRlTeacher
from wbc_mjlab.rl.two_teacher_blend import two_teacher_kl_loss


class Casbot02DistillPPO(PPO):
  """PPO + explicit-blend KL distillation from two frozen teachers.

  The two teachers are ``FrozenRslRlTeacher`` instances whose ``forward``
  caches ``(mean, std)`` in ``output_distribution_params``. The AMP teacher has
  the term-major->time-major input permutation enabled (``use_time_major_input``).
  """

  def __init__(
    self,
    actor,
    critic,
    storage: RolloutStorage,
    loco_teacher,
    amp_teacher,
    command_obs_group: str = "command",
    dagger_coef: float = 0.4,
    dagger_coef_min: float = 0.2,
    dagger_coef_anneal_steps: int = 60_000,
    device: str = "cpu",
    **ppo_kwargs,
  ) -> None:
    super().__init__(actor, critic, storage, device=device, **ppo_kwargs)
    self.loco_teacher = loco_teacher.to(device)
    self.amp_teacher = amp_teacher.to(device)
    self.command_obs_group = command_obs_group

    self.dagger_coef_init = float(dagger_coef)
    self.dagger_coef = float(dagger_coef)
    self.dagger_coef_min = float(dagger_coef_min)
    self.dagger_coef_anneal_steps = int(dagger_coef_anneal_steps)
    self.update_counter = 0

  def _current_dagger_coef(self) -> float:
    if self.dagger_coef_anneal_steps <= 0 or self.update_counter >= self.dagger_coef_anneal_steps:
      return self.dagger_coef_min
    progress = self.update_counter / self.dagger_coef_anneal_steps
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return self.dagger_coef_min + (self.dagger_coef_init - self.dagger_coef_min) * cosine

  def _teacher_params(self, teacher, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
      teacher(obs, stochastic_output=False)
    return tuple(p.detach() for p in teacher.output_distribution_params)

  def update(self) -> dict[str, float]:
    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    mean_distill = 0.0

    self.dagger_coef = self._current_dagger_coef()

    generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
    for batch in generator:
      # Standard PPO forward (mirrors rsl_rl PPO.update).
      self.actor(batch.observations, stochastic_output=True)
      actions_log_prob = self.actor.get_output_log_prob(batch.actions)
      values = self.critic(batch.critic_observations if batch.critic_observations is not None else batch.observations)
      distribution_params = self.actor.output_distribution_params
      entropy = self.actor.output_entropy

      if self.desired_kl is not None and self.schedule == "adaptive":
        with torch.inference_mode():
          kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)
          kl_mean = torch.mean(kl)
          if kl_mean > self.desired_kl * 2.0:
            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
          elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
          for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

      ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
      surrogate = -torch.squeeze(batch.advantages) * ratio
      surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      if self.use_clipped_value_loss:
        value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
        value_loss = torch.max((values - batch.returns).pow(2), (value_clipped - batch.returns).pow(2)).mean()
      else:
        value_loss = (batch.returns - values).pow(2).mean()

      loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

      # Distillation: explicit-blend KL from the two frozen teachers.
      student_mu, student_std = distribution_params
      loco_mu, loco_std = self._teacher_params(self.loco_teacher, batch.observations)
      amp_mu, amp_std = self._teacher_params(self.amp_teacher, batch.observations)
      cmd = batch.observations[self.command_obs_group]  # [B, 3] current command
      distill_loss, distill_logs = two_teacher_kl_loss(
        student_mu, student_std,
        loco_mu, loco_std,
        amp_mu, amp_std,
        cmd,
        coef_loco=self.dagger_coef,
        coef_amp=self.dagger_coef,
      )
      loss = loss + distill_loss

      self.optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      self.optimizer.step()

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy.mean().item()
      mean_distill += distill_loss.item()

    num_updates = self.num_learning_epochs * self.num_mini_batches
    self.update_counter += 1

    return {
      "value": mean_value_loss / num_updates,
      "surrogate": mean_surrogate_loss / num_updates,
      "entropy": mean_entropy / num_updates,
      "distill": mean_distill / num_updates,
      "dagger_coef": self.dagger_coef,
    }

  @staticmethod
  def construct_algorithm(obs, env, cfg: dict, device: str) -> "Casbot02DistillPPO":
    """Construct the student actor/critic/storage + load the two frozen teachers."""
    actor_class = resolve_callable(cfg["actor"].pop("class_name"))
    critic_class = resolve_callable(cfg["critic"].pop("class_name"))
    cfg["algorithm"].pop("class_name", None)  # resolved by the runner already
    cfg["algorithm"].pop("share_cnn_encoders", None)
    cfg["algorithm"].setdefault("rnd_cfg", None)
    cfg["algorithm"].setdefault("symmetry_cfg", None)

    cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])

    actor = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
    critic = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)

    storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

    # Frozen teachers. Both are plain MLPs (512/256/128 -> 12) over the same
    # 180-dim obs; the AMP teacher additionally needs the term-major->time-major
    # input permutation (it was trained in mjlab 1.2.0).
    teacher_kwargs = dict(
      obs_dim=cfg["teacher_obs_dim"],
      hidden_dims=tuple(cfg["teacher_hidden_dims"]),
      output_dim=cfg["teacher_output_dim"],
      activation=cfg.get("teacher_activation", "elu"),
      min_std=cfg.get("teacher_min_std", 0.05),
    )
    loco_teacher = FrozenRslRlTeacher(**teacher_kwargs).load_handoff_checkpoint(
      cfg["loco_teacher_checkpoint"], device
    )
    amp_teacher = FrozenRslRlTeacher(**teacher_kwargs).load_checkpoint(
      cfg["amp_teacher_checkpoint"], device
    ).use_time_major_input()

    return Casbot02DistillPPO(
      actor=actor,
      critic=critic,
      storage=storage,
      loco_teacher=loco_teacher,
      amp_teacher=amp_teacher,
      command_obs_group=cfg.get("command_obs_group", "command"),
      dagger_coef=cfg.get("dagger_coef", 0.4),
      dagger_coef_min=cfg.get("dagger_coef_min", 0.2),
      dagger_coef_anneal_steps=cfg.get("dagger_coef_anneal_steps", 60_000),
      device=device,
      **cfg["algorithm"],
      multi_gpu_cfg=cfg.get("multi_gpu"),
    )


__all__ = ["Casbot02DistillPPO"]
