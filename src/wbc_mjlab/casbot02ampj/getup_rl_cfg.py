"""AMP-PPO runner configuration for the CASBOT02 get-up-only teacher."""

from __future__ import annotations

from pathlib import Path

from mjlab.rl import RslRlModelCfg

from wbc_mjlab.amp_ppo_config import AmpAlgorithmCfg, AmpRunnerCfg
from wbc_mjlab.casbot02ampj import constants as C


_GETUP_DATA_DIR = (
  Path(__file__).resolve().parents[1]
  / "data"
  / "motions"
  / "casbot02ampj"
  / "amp"
  / "GetUp"
)


def casbot02_ampj_getup_teacher_flat_runner_cfg() -> AmpRunnerCfg:
  """Use HANDOFF AMP-PPO semantics with a get-up-only expert dataset."""
  experiment_name = "casbot02_ampj_getup_teacher_flat"
  return AmpRunnerCfg(
    seed=42,
    num_steps_per_env=24,
    max_iterations=500_000,
    obs_groups={"actor": ("actor",), "critic": ("critic",)},
    save_interval=1000,
    experiment_name=experiment_name,
    run_name=experiment_name,
    logger="tensorboard",
    wandb_project="casbot02",
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=AmpAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      amp_loss_coef=1.0,
      amp_grad_pen_lambda=10.0,
    ),
    amp_motion_files=str(_GETUP_DATA_DIR),
    amp_body_names=C.AMP_BODY_NAMES,
    amp_anchor_name=C.ANCHOR_BODY_NAME,
    amp_reward_coef=0.1,
    amp_task_reward_lerp=0.75,
    amp_discr_hidden_dims=(1024, 512, 256),
    amp_replay_buffer_size=100_000,
  )


__all__ = ["casbot02_ampj_getup_teacher_flat_runner_cfg"]
