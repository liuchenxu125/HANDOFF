"""CASBOT02 27-DoF full-body model used by the AMP recovery teacher.

The source MJCF contains actuated head and finger joints.  The policy-facing
model keeps only 12 leg, one waist-yaw, and 14 arm joints.  The head joints
are removed so the head is rigid, while the complete dexterous-hand body
subtrees and palm meshes are removed because the recovery robot has no hands.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg


SOURCE_XML: Path = (
  Path(__file__).resolve().parents[1]
  / "CASBOT02_ENCOS_7dof_shell_20251015"
  / "Serial"
  / "xml"
  / "CASBOT02_ENCOS_7dof_shell_20251015.xml"
)
assert SOURCE_XML.exists(), SOURCE_XML


LEG_JOINT_NAMES: tuple[str, ...] = (
  "left_leg_pelvic_pitch_joint",
  "left_leg_pelvic_roll_joint",
  "left_leg_pelvic_yaw_joint",
  "left_leg_knee_pitch_joint",
  "left_leg_ankle_pitch_joint",
  "left_leg_ankle_roll_joint",
  "right_leg_pelvic_pitch_joint",
  "right_leg_pelvic_roll_joint",
  "right_leg_pelvic_yaw_joint",
  "right_leg_knee_pitch_joint",
  "right_leg_ankle_pitch_joint",
  "right_leg_ankle_roll_joint",
)

WAIST_JOINT_NAMES: tuple[str, ...] = ("waist_yaw_joint",)

LEFT_ARM_JOINT_NAMES: tuple[str, ...] = (
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_pitch_joint",
  "left_wrist_yaw_joint",
  "left_wrist_pitch_joint",
  "left_wrist_roll_joint",
)

RIGHT_ARM_JOINT_NAMES: tuple[str, ...] = (
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_pitch_joint",
  "right_wrist_yaw_joint",
  "right_wrist_pitch_joint",
  "right_wrist_roll_joint",
)

JOINT_NAMES: tuple[str, ...] = (
  *LEG_JOINT_NAMES,
  *WAIST_JOINT_NAMES,
  *LEFT_ARM_JOINT_NAMES,
  *RIGHT_ARM_JOINT_NAMES,
)
assert len(JOINT_NAMES) == 27

HEAD_JOINT_NAMES: tuple[str, ...] = ("head_yaw_joint", "head_pitch_joint")
_FINGER_TOKENS: tuple[str, ...] = (
  "thumb",
  "index",
  "middle",
  "ring",
  "pinky",
)

# Each named body is the root of one finger subtree.  Deleting these ten roots
# removes all 32 finger bodies and their joints while keeping wrist roll as the
# end of each 7-DoF arm.
HAND_ROOT_BODY_NAMES: tuple[str, ...] = (
  "left_thumb_metacarpal_Link",
  "left_index_proximal_Link",
  "left_middle_proximal_Link",
  "left_ring_proximal_Link",
  "left_pinky_proximal_Link",
  "right_thumb_metacarpal_link",
  "right_index_proximal_link",
  "right_middle_proximal_link",
  "right_ring_proximal_link",
  "right_pinky_proximal_link",
)
HAND_BASE_MESH_NAMES: tuple[str, ...] = ("left_base_link", "right_base_link")

ROOT_BODY_NAME = "base_link"
ANCHOR_BODY_NAME = "waist_yaw_link"
FEET_BODY_NAMES: tuple[str, ...] = (
  "left_leg_ankle_roll_link",
  "right_leg_ankle_roll_link",
)
FEET_SITE_NAMES: tuple[str, ...] = ("left_foot", "right_foot")
FOOT_GEOM_NAMES: tuple[str, ...] = (
  "left_foot_collision",
  "right_foot_collision",
)

# These non-parent body pairs overlap at their physical joint housings in the
# source mesh.  MuJoCo would otherwise report large self-contact forces even
# in the nominal standing pose, making the AMP self-collision reward a
# constant false positive.  All other robot self-collisions remain enabled.
STRUCTURAL_CONTACT_EXCLUDES: tuple[tuple[str, str], ...] = (
  ("left_leg_knee_pitch_link", "left_leg_ankle_roll_link"),
  ("right_leg_knee_pitch_link", "right_leg_ankle_roll_link"),
  ("left_wrist_yaw_link", "left_wrist_roll_link"),
  ("right_wrist_yaw_link", "right_wrist_roll_link"),
)

# Same semantic key bodies as G1's AMP teacher: root, hip-roll, knee, foot,
# shoulder-roll, elbow, and wrist-yaw, in left-then-right order.
AMP_BODY_NAMES: tuple[str, ...] = (
  ROOT_BODY_NAME,
  "left_leg_pelvic_roll_link",
  "left_leg_knee_pitch_link",
  "left_leg_ankle_roll_link",
  "right_leg_pelvic_roll_link",
  "right_leg_knee_pitch_link",
  "right_leg_ankle_roll_link",
  "left_shoulder_roll_link",
  "left_elbow_pitch_link",
  "left_wrist_yaw_link",
  "right_shoulder_roll_link",
  "right_elbow_pitch_link",
  "right_wrist_yaw_link",
)
assert len(AMP_BODY_NAMES) == 13


def _find_body(spec: mujoco.MjSpec, name: str):
  for body in spec.bodies:
    if body.name == name:
      return body
  raise ValueError(f"Body {name!r} not found in {SOURCE_XML}")


def _remove_dexterous_hands(spec: mujoco.MjSpec) -> None:
  """Remove both finger trees, palm geoms, and their now-unused meshes."""
  before_body_names = {body.name for body in spec.bodies}
  for name in HAND_ROOT_BODY_NAMES:
    spec.delete(_find_body(spec, name))

  after_body_names = {body.name for body in spec.bodies}
  removed_body_names = before_body_names - after_body_names
  if len(removed_body_names) != 32:
    raise ValueError(
      f"Expected to remove 32 hand bodies, removed {len(removed_body_names)}"
    )
  remaining_hand_bodies = {
    name for name in after_body_names
    if any(token in name.lower() for token in _FINGER_TOKENS)
  }
  if remaining_hand_bodies:
    raise ValueError(f"Hand bodies remain after removal: {remaining_hand_bodies}")

  # The hand base is represented by two geoms directly attached to each wrist
  # roll body rather than by a separate body, so remove those geoms explicitly.
  removed_palm_geoms = 0
  for wrist_name, mesh_name in zip(
    ("left_wrist_roll_link", "right_wrist_roll_link"), HAND_BASE_MESH_NAMES
  ):
    wrist = _find_body(spec, wrist_name)
    palm_geoms = [geom for geom in wrist.geoms if geom.meshname == mesh_name]
    if len(palm_geoms) != 2:
      raise ValueError(
        f"Expected two {mesh_name} geoms on {wrist_name}, got {len(palm_geoms)}"
      )
    for geom in palm_geoms:
      spec.delete(geom)
      removed_palm_geoms += 1
  if removed_palm_geoms != 4:
    raise ValueError(f"Expected to remove 4 palm geoms, removed {removed_palm_geoms}")

  removed_meshes: list[str] = []
  for mesh in list(spec.meshes):
    name = mesh.name
    if name in HAND_BASE_MESH_NAMES or any(
      token in name.lower() for token in _FINGER_TOKENS
    ):
      removed_meshes.append(name)
      spec.delete(mesh)
  if len(removed_meshes) != 34:
    raise ValueError(
      f"Expected to remove 34 hand mesh assets, removed {len(removed_meshes)}"
    )


def _remove_uncontrolled_joints(spec: mujoco.MjSpec) -> None:
  """Fix the two head joints after the hand subtrees have been removed."""
  keep = set(JOINT_NAMES)
  removed: list[str] = []
  for joint in list(spec.joints):
    name = joint.name
    if name == "base_link_free_joint" or name in keep:
      continue
    if name in HEAD_JOINT_NAMES:
      removed.append(name)
      spec.delete(joint)
      continue
    raise ValueError(f"Unexpected uncontrolled joint {name!r} in {SOURCE_XML}")

  if set(removed) != set(HEAD_JOINT_NAMES):
    raise ValueError(f"Expected to remove both head joints, removed {removed}")


def _add_runtime_frames(spec: mujoco.MjSpec) -> None:
  foot_site_pos = (0.0348, 0.0, -0.0648)
  for body_name, geom_name, site_name in zip(
    FEET_BODY_NAMES, FOOT_GEOM_NAMES, FEET_SITE_NAMES
  ):
    body = _find_body(spec, body_name)
    collision_geoms = [geom for geom in body.geoms if geom.contype != 0]
    if len(collision_geoms) != 1:
      raise ValueError(
        f"Expected one collision geom on {body_name}, got {len(collision_geoms)}"
      )
    collision_geoms[0].name = geom_name
    site = body.add_site(name=site_name)
    site.pos = foot_site_pos

  root = _find_body(spec, ROOT_BODY_NAME)
  imu = root.add_site(name="imu")
  imu.pos = (0.0, 0.0, 0.0)

  spec.add_sensor(
    name="angular-velocity",
    type=mujoco.mjtSensor.mjSENS_GYRO,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname="imu",
    noise=0.005,
    cutoff=34.9,
  )
  spec.add_sensor(
    name="linear-velocity",
    type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname="imu",
    noise=0.001,
    cutoff=30.0,
  )
  spec.add_sensor(
    name="root_angmom",
    type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
    objtype=mujoco.mjtObj.mjOBJ_BODY,
    objname=ROOT_BODY_NAME,
  )


def _exclude_structural_overlap_contacts(spec: mujoco.MjSpec) -> None:
  for body1, body2 in STRUCTURAL_CONTACT_EXCLUDES:
    spec.add_exclude(
      name=f"exclude_{body1}_{body2}",
      bodyname1=body1,
      bodyname2=body2,
    )


def get_spec() -> mujoco.MjSpec:
  """Load the source asset and derive the fixed-head/no-hand 27-DoF spec."""
  spec = mujoco.MjSpec.from_file(str(SOURCE_XML))

  for geom in list(spec.geoms):
    if geom.name == "ground":
      spec.delete(geom)
  for actuator in list(spec.actuators):
    spec.delete(actuator)

  _remove_dexterous_hands(spec)
  _remove_uncontrolled_joints(spec)
  _add_runtime_frames(spec)
  _exclude_structural_overlap_contacts(spec)
  return spec


NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0


def _stiffness(armature: float) -> float:
  return armature * NATURAL_FREQ**2


def _damping(armature: float) -> float:
  return 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ


MOTOR_DELAY_MIN_LAG = 0
MOTOR_DELAY_MAX_LAG = 2
MOTOR_DELAY_UPDATE_PERIOD = 4
MOTOR_DELAY_HOLD_PROB = 0.5


def _with_motor_delay(cfg: BuiltinPositionActuatorCfg) -> BuiltinPositionActuatorCfg:
  return replace(
    cfg,
    delay_min_lag=MOTOR_DELAY_MIN_LAG,
    delay_max_lag=MOTOR_DELAY_MAX_LAG,
    delay_hold_prob=MOTOR_DELAY_HOLD_PROB,
    delay_update_period=MOTOR_DELAY_UPDATE_PERIOD,
    delay_per_env_phase=True,
  )


LEG_HEAVY_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "left_leg_pelvic_pitch_joint",
    "left_leg_pelvic_roll_joint",
    "left_leg_knee_pitch_joint",
    "right_leg_pelvic_pitch_joint",
    "right_leg_pelvic_roll_joint",
    "right_leg_knee_pitch_joint",
  ),
  stiffness=_stiffness(0.07),
  damping=_damping(0.07),
  effort_limit=150.0,
  armature=0.07,
  frictionloss=0.01,
)

LEG_LIGHT_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "left_leg_pelvic_yaw_joint",
    "left_leg_ankle_pitch_joint",
    "left_leg_ankle_roll_joint",
    "right_leg_pelvic_yaw_joint",
    "right_leg_ankle_pitch_joint",
    "right_leg_ankle_roll_joint",
  ),
  stiffness=_stiffness(0.029),
  damping=_damping(0.029),
  effort_limit=120.0,
  armature=0.029,
  frictionloss=0.01,
)

WAIST_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=WAIST_JOINT_NAMES,
  stiffness=_stiffness(0.0245),
  damping=_damping(0.0245),
  effort_limit=60.0,
  armature=0.0245,
  frictionloss=0.01,
)

ARM_HEAVY_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_elbow_pitch_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_elbow_pitch_joint",
  ),
  stiffness=_stiffness(0.033),
  damping=_damping(0.033),
  effort_limit=40.0,
  armature=0.033,
  frictionloss=0.01,
)

# Wrist pitch/roll join the previous light-arm group, as explicitly selected
# for the 27-DoF teacher.
ARM_LIGHT_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "left_shoulder_yaw_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_shoulder_yaw_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
  ),
  stiffness=_stiffness(0.0245),
  damping=_damping(0.0245),
  effort_limit=30.0,
  armature=0.0245,
  frictionloss=0.01,
)


HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.92),
  joint_pos={
    "left_leg_pelvic_pitch_joint": -0.185,
    "left_leg_pelvic_roll_joint": 0.0,
    "left_leg_pelvic_yaw_joint": 0.0,
    "left_leg_knee_pitch_joint": 0.36,
    "left_leg_ankle_pitch_joint": -0.175,
    "left_leg_ankle_roll_joint": 0.0,
    "right_leg_pelvic_pitch_joint": -0.185,
    "right_leg_pelvic_roll_joint": 0.0,
    "right_leg_pelvic_yaw_joint": 0.0,
    "right_leg_knee_pitch_joint": 0.36,
    "right_leg_ankle_pitch_joint": -0.175,
    "right_leg_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0,
    "left_shoulder_pitch_joint": 0.1,
    "left_shoulder_roll_joint": 0.175,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_pitch_joint": -0.35,
    "left_wrist_yaw_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_roll_joint": 0.0,
    "right_shoulder_pitch_joint": 0.1,
    "right_shoulder_roll_joint": -0.175,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_pitch_joint": -0.35,
    "right_wrist_yaw_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)


ARTICULATION = EntityArticulationInfoCfg(
  actuators=tuple(
    _with_motor_delay(cfg)
    for cfg in (
      LEG_HEAVY_ACTUATOR,
      LEG_LIGHT_ACTUATOR,
      WAIST_ACTUATOR,
      ARM_HEAVY_ACTUATOR,
      ARM_LIGHT_ACTUATOR,
    )
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    spec_fn=get_spec,
    articulation=ARTICULATION,
    sort_actuators=True,
  )


ACTION_SCALE: dict[str, float] = {}
for actuator_cfg in ARTICULATION.actuators:
  assert isinstance(actuator_cfg, BuiltinPositionActuatorCfg)
  assert actuator_cfg.effort_limit is not None
  for joint_name in actuator_cfg.target_names_expr:
    ACTION_SCALE[joint_name] = (
      0.25 * actuator_cfg.effort_limit / actuator_cfg.stiffness
    )

if set(ACTION_SCALE) != set(JOINT_NAMES):
  missing = set(JOINT_NAMES) - set(ACTION_SCALE)
  extra = set(ACTION_SCALE) - set(JOINT_NAMES)
  raise ValueError(f"Actuator coverage mismatch: missing={missing}, extra={extra}")


__all__ = [
  "ACTION_SCALE",
  "AMP_BODY_NAMES",
  "ANCHOR_BODY_NAME",
  "FEET_BODY_NAMES",
  "FEET_SITE_NAMES",
  "FOOT_GEOM_NAMES",
  "HAND_BASE_MESH_NAMES",
  "HAND_ROOT_BODY_NAMES",
  "HOME_KEYFRAME",
  "JOINT_NAMES",
  "ROOT_BODY_NAME",
  "SOURCE_XML",
  "STRUCTURAL_CONTACT_EXCLUDES",
  "get_robot_cfg",
  "get_spec",
]
