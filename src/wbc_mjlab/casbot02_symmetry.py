"""Left-right symmetry transforms for the Casbot02 loco teacher.

The loco actor observation is term-major with four history frames::

  [base_ang_vel(3), projected_gravity(3), command(3),
   joint_pos(12), joint_vel(12), last_action(12)] x 4 = 180

The critic appends four frames of body-frame base linear velocity, for 192
dimensions in total.  Reflection is about the robot sagittal (x-z) plane.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


HISTORY_LENGTH = 4
ACTOR_OBS_DIM = 180
CRITIC_OBS_DIM = 192
NUM_ACTIONS = 12

# Joint order per leg: hip pitch, hip roll, hip yaw, knee pitch,
# ankle pitch, ankle roll.  Pitch joints keep their sign under a sagittal
# reflection; roll/yaw joints change sign after swapping left and right.
_JOINT_REFLECTION_SIGN = (1.0, -1.0, -1.0, 1.0, 1.0, -1.0)


def _mirror_vector_history(
  value: torch.Tensor,
  sign: tuple[float, ...],
) -> torch.Tensor:
  """Mirror a term stored as ``[history, term_dim]`` in the last axis."""
  term_dim = len(sign)
  expected_dim = HISTORY_LENGTH * term_dim
  if value.shape[-1] != expected_dim:
    raise ValueError(
      f"Expected a {expected_dim}-D history term, got {value.shape[-1]}"
    )
  sign_tensor = value.new_tensor(sign)
  return (
    value.reshape(*value.shape[:-1], HISTORY_LENGTH, term_dim)
    * sign_tensor
  ).reshape_as(value)


def mirror_leg_joints(value: torch.Tensor) -> torch.Tensor:
  """Swap the two six-DoF legs and reflect roll/yaw joint coordinates."""
  if value.shape[-1] != NUM_ACTIONS:
    raise ValueError(f"Expected {NUM_ACTIONS} leg values, got {value.shape[-1]}")
  sign = value.new_tensor(_JOINT_REFLECTION_SIGN)
  left = value[..., :6]
  right = value[..., 6:]
  return torch.cat((right * sign, left * sign), dim=-1)


def _mirror_joint_history(value: torch.Tensor) -> torch.Tensor:
  expected_dim = HISTORY_LENGTH * NUM_ACTIONS
  if value.shape[-1] != expected_dim:
    raise ValueError(
      f"Expected a {expected_dim}-D joint history term, got {value.shape[-1]}"
    )
  frames = value.reshape(*value.shape[:-1], HISTORY_LENGTH, NUM_ACTIONS)
  return mirror_leg_joints(frames).reshape_as(value)


def mirror_loco_observation(obs: torch.Tensor) -> torch.Tensor:
  """Mirror one flattened actor (180-D) or critic (192-D) observation."""
  if obs.shape[-1] not in (ACTOR_OBS_DIM, CRITIC_OBS_DIM):
    raise ValueError(
      "Casbot02 loco symmetry expects actor/critic observation dimensions "
      f"{ACTOR_OBS_DIM}/{CRITIC_OBS_DIM}, got {obs.shape[-1]}"
    )

  mirrored = obs.clone()
  offset = 0

  # Angular velocity is an axial vector.  Under y -> -y its x/z components
  # flip sign while y is preserved.
  width = HISTORY_LENGTH * 3
  mirrored[..., offset : offset + width] = _mirror_vector_history(
    obs[..., offset : offset + width], (-1.0, 1.0, -1.0)
  )
  offset += width

  # Projected gravity is a polar vector: only y changes sign.
  mirrored[..., offset : offset + width] = _mirror_vector_history(
    obs[..., offset : offset + width], (1.0, -1.0, 1.0)
  )
  offset += width

  # Command = body-frame vx, vy and axial yaw rate wz.
  mirrored[..., offset : offset + width] = _mirror_vector_history(
    obs[..., offset : offset + width], (1.0, -1.0, -1.0)
  )
  offset += width

  joint_width = HISTORY_LENGTH * NUM_ACTIONS
  for _ in range(3):  # joint_pos, joint_vel, last_action
    mirrored[..., offset : offset + joint_width] = _mirror_joint_history(
      obs[..., offset : offset + joint_width]
    )
    offset += joint_width

  if obs.shape[-1] == CRITIC_OBS_DIM:
    # Privileged body-frame base linear velocity is a polar vector.
    mirrored[..., offset : offset + width] = _mirror_vector_history(
      obs[..., offset : offset + width], (1.0, -1.0, 1.0)
    )
    offset += width

  if offset != obs.shape[-1]:
    raise RuntimeError(
      f"Casbot02 symmetry consumed {offset} values from {obs.shape[-1]}"
    )
  return mirrored


@torch.no_grad()
def compute_symmetric_states(
  env: ManagerBasedRlEnv,
  obs: TensorDict | None = None,
  actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
  """Append the sagittally mirrored sample to an RSL-RL mini-batch."""
  del env  # The fixed Casbot02 layout is validated explicitly above.

  obs_aug = None
  if obs is not None:
    batch_size = obs.batch_size[0]
    obs_aug = obs.repeat(2)
    for key, value in obs.items():
      obs_aug[key][:batch_size] = value
      obs_aug[key][batch_size:] = mirror_loco_observation(value)

  actions_aug = None
  if actions is not None:
    actions_aug = torch.cat((actions, mirror_leg_joints(actions)), dim=0)

  return obs_aug, actions_aug


__all__ = [
  "ACTOR_OBS_DIM",
  "CRITIC_OBS_DIM",
  "HISTORY_LENGTH",
  "NUM_ACTIONS",
  "compute_symmetric_states",
  "mirror_leg_joints",
  "mirror_loco_observation",
]
