"""RL configs for the Casbot02 loco teacher and the distill student."""

from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def _mlp_actor() -> RslRlModelCfg:
  """Plain MLP actor (512/256/128 -> 12), matching the AMP teacher's arch."""
  return RslRlModelCfg(
    hidden_dims=(512, 256, 128),
    activation="elu",
    obs_normalization=True,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 1.0,
      "std_type": "scalar",
    },
  )


def _mlp_critic() -> RslRlModelCfg:
  return RslRlModelCfg(
    hidden_dims=(512, 256, 128),
    activation="elu",
    obs_normalization=True,
  )


def _ppo_algorithm(class_name: str = "rsl_rl.algorithms:PPO") -> RslRlPpoAlgorithmCfg:
  return RslRlPpoAlgorithmCfg(
    class_name=class_name,
    num_learning_epochs=5,
    num_mini_batches=4,
    learning_rate=1.0e-3,
    schedule="adaptive",
    gamma=0.99,
    lam=0.95,
    entropy_coef=0.005,
    desired_kl=0.01,
    max_grad_norm=1.0,
    value_loss_coef=1.0,
    use_clipped_value_loss=True,
    clip_param=0.2,
  )


def casbot02_loco_teacher_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Plain-PPO runner for the 12-leg Casbot02 velocity-tracking loco teacher."""
  return RslRlOnPolicyRunnerCfg(
    seed=42,
    num_steps_per_env=24,
    max_iterations=20_001,
    obs_groups={"actor": ("actor",), "critic": ("critic",)},
    save_interval=1000,
    experiment_name="casbot02_loco_teacher",
    run_name="casbot02_loco_teacher",
    logger="tensorboard",
    wandb_project="casbot02",
    actor=_mlp_actor(),
    critic=_mlp_critic(),
    algorithm=_ppo_algorithm(),
  )


@dataclass
class Casbot02DistillRunnerCfg(RslRlOnPolicyRunnerCfg):
  """Runner config for the 2-teacher distillation student."""

  # Frozen teacher checkpoint paths (filled by the caller / CLI).
  loco_teacher_checkpoint: str = ""
  amp_teacher_checkpoint: str = ""
  # Teacher architecture (both are plain MLPs over the same 180-dim obs).
  teacher_obs_dim: int = 180
  teacher_hidden_dims: tuple[int, ...] = (512, 256, 128)
  teacher_output_dim: int = 12
  teacher_activation: str = "elu"
  teacher_min_std: float = 0.05
  # Distillation knobs.
  command_obs_group: str = "command"
  dagger_coef: float = 0.4
  dagger_coef_min: float = 0.2
  dagger_coef_anneal_steps: int = 60_000


def casbot02_student_runner_cfg() -> Casbot02DistillRunnerCfg:
  """Distillation student runner: PPO + explicit-blend KL from AMP & loco."""
  return Casbot02DistillRunnerCfg(
    seed=42,
    num_steps_per_env=24,
    max_iterations=20_001,
    obs_groups={"actor": ("actor",), "critic": ("critic",)},
    save_interval=1000,
    experiment_name="casbot02_distill_student",
    run_name="casbot02_distill_student",
    logger="tensorboard",
    wandb_project="casbot02",
    actor=_mlp_actor(),
    critic=_mlp_critic(),
    algorithm=_ppo_algorithm(class_name="wbc_mjlab.rl.casbot02_distill:Casbot02DistillPPO"),
    loco_teacher_checkpoint="",  # fill in: logs/.../model_*.pt
    amp_teacher_checkpoint="",   # fill in: AMP model_8000.pt
  )


__all__ = [
  "casbot02_loco_teacher_runner_cfg",
  "Casbot02DistillRunnerCfg",
  "casbot02_student_runner_cfg",
]
