"""CASBOT02 23DOF constants."""

from dataclasses import replace
from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

##
# MJCF and assets.
##

CASBOT02_23DOF_XML: Path = (
  Path(__file__).parent / "CASBOT02_7dof" / "xml" / "CASBOT_02_23dof.xml"
)
assert CASBOT02_23DOF_XML.exists(), CASBOT02_23DOF_XML

CASBOT02_23DOF_JOINT_NAMES: tuple[str, ...] = (
  "leg_l1_joint",
  "leg_l2_joint",
  "leg_l3_joint",
  "leg_l4_joint",
  "leg_l5_joint",
  "leg_l6_joint",
  "leg_r1_joint",
  "leg_r2_joint",
  "leg_r3_joint",
  "leg_r4_joint",
  "leg_r5_joint",
  "leg_r6_joint",
  "upper_left_1_joint",
  "upper_left_2_joint",
  "upper_left_3_joint",
  "upper_left_4_joint",
  "upper_left_5_joint",
  "upper_right_1_joint",
  "upper_right_2_joint",
  "upper_right_3_joint",
  "upper_right_4_joint",
  "upper_right_5_joint",
)

CASBOT02_MOTOR_COMM_DELAY_MIN_LAG = 0
CASBOT02_MOTOR_COMM_DELAY_MAX_LAG = 2
CASBOT02_MOTOR_COMM_DELAY_UPDATE_PERIOD = 4
CASBOT02_MOTOR_COMM_DELAY_HOLD_PROB = 0.5

CASBOT02_23DOF_AMP_BODY_NAMES: tuple[str, ...] = (
  "torso",
  "leg_l2_link",
  "leg_l4_link",
  "leg_l6_link",
  "leg_r2_link",
  "leg_r4_link",
  "leg_r6_link",
  "left_shoulder_roll_link",
  "left_elbow_pitch_link",
  "left_wrist_yaw_link",
  "right_shoulder_roll_link",
  "right_elbow_pitch_link",
  "right_wrist_yaw_link",
)

CASBOT02_LEG_AMP_BODY_NAMES: tuple[str, ...] = (
  "torso",
  "leg_l2_link",
  "leg_l4_link",
  "leg_l6_link",
  "leg_r2_link",
  "leg_r4_link",
  "leg_r6_link",
)

CASBOT02_LEG_JOINT_NAMES: tuple[str, ...] = (
  "leg_l1_joint",
  "leg_l2_joint",
  "leg_l3_joint",
  "leg_l4_joint",
  "leg_l5_joint",
  "leg_l6_joint",
  "leg_r1_joint",
  "leg_r2_joint",
  "leg_r3_joint",
  "leg_r4_joint",
  "leg_r5_joint",
  "leg_r6_joint",
)

CASBOT02_LEG_ONLY_JOINT_NAMES: tuple[str, ...] = CASBOT02_LEG_JOINT_NAMES

CASBOT02_FOOT_GEOM_NAMES: tuple[str, ...] = (
  "left_foot_collision",
  "right_foot_collision",
)

CASBOT02_22DOF_NO_WAIST_JOINT_NAMES: tuple[str, ...] = tuple(
  name for name in CASBOT02_23DOF_JOINT_NAMES if name != "waist_yaw_joint"
)


def get_spec() -> mujoco.MjSpec:
  # mjlab 1.4.0 / mujoco 3.8.1: MjSpec.from_file resolves the XML's ``meshdir``
  # (``../meshes/``) automatically, so no manual ``spec.assets`` wiring is needed
  # (unlike the mjlab 1.2.0 ``update_assets`` pattern this was ported from).
  spec = mujoco.MjSpec.from_file(str(CASBOT02_23DOF_XML))

  # The source XML is a standalone MuJoCo scene. In mjlab the terrain is provided
  # by SceneCfg, so keep only the robot body tree from the XML.
  for geom in list(spec.geoms):
    if geom.name == "ground":
      spec.delete(geom)

  # Replace XML torque motors with mjlab position actuators below. This keeps the
  # action semantics consistent with JointPositionActionCfg.
  for act in list(spec.actuators):
    spec.delete(act)

  # 给脚加 site(Casbot02 源 XML 没有 site), 供 foot_height_scan /
  # feet_clearance 使用。位置是脚部碰撞 mesh 的平整底面中心：底面 z=-0.0648 m，
  # 接触区域 x 范围约 [-0.09394, 0.16354] m，中心 x≈0.0348 m。
  foot_sole_site_pos = [0.0348, 0.0, -0.0648]
  for body_name, site_name in (
    ("leg_l6_link", "left_foot"),
    ("leg_r6_link", "right_foot"),
  ):
    for body in spec.worldbody.find_all("body"):
      if body.name == body_name:
        site = body.add_site(name=site_name)
        site.pos = foot_sole_site_pos
        break

  return spec


##
# Actuator config.
##

NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0


def _stiffness(armature: float) -> float:
  return armature * NATURAL_FREQ**2


def _damping(armature: float) -> float:
  return 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ


def _with_motor_comm_delay(
  base_cfg: BuiltinPositionActuatorCfg,
) -> BuiltinPositionActuatorCfg:
  # 第15次修改: CASBOT02 关节 position target 加 0-10ms 电机通讯延迟。
  # (mjlab 1.4.0 无 DelayedActuatorCfg 包装类, 延迟字段直接在 actuator cfg 上。)
  return replace(
    base_cfg,
    delay_min_lag=CASBOT02_MOTOR_COMM_DELAY_MIN_LAG,
    delay_max_lag=CASBOT02_MOTOR_COMM_DELAY_MAX_LAG,
    delay_hold_prob=CASBOT02_MOTOR_COMM_DELAY_HOLD_PROB,
    delay_update_period=CASBOT02_MOTOR_COMM_DELAY_UPDATE_PERIOD,
    delay_per_env_phase=True,
  )


CASBOT02_LEG_HEAVY_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "leg_l1_joint",
    "leg_l2_joint",
    "leg_l4_joint",
    "leg_r1_joint",
    "leg_r2_joint",
    "leg_r4_joint",
  ),
  stiffness=_stiffness(0.07),
  damping=_damping(0.07),
  effort_limit=120.0,
  armature=0.07,
  frictionloss=0.01,
)

CASBOT02_LEG_LIGHT_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "leg_l3_joint",
    "leg_l5_joint",
    "leg_l6_joint",
    "leg_r3_joint",
    "leg_r5_joint",
    "leg_r6_joint",
  ),
  stiffness=_stiffness(0.029),#原值0.039
  damping=_damping(0.029),  # 真机电机 kd 上限 5.0
  effort_limit=80.0,
  armature=0.029,
  frictionloss=0.01,
)

CASBOT02_WAIST_YAW_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_yaw_joint",),
  stiffness=_stiffness(0.0245),
  damping=_damping(0.0245),
  effort_limit=60.0,
  armature=0.0245,
  frictionloss=0.01,
)

CASBOT02_ARM_HEAVY_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "upper_left_1_joint",
    "upper_left_2_joint",
    "upper_left_4_joint",
    "upper_right_1_joint",
    "upper_right_2_joint",
    "upper_right_4_joint",
  ),
  stiffness=_stiffness(0.033),
  damping=_damping(0.033),
  effort_limit=40.0,
  armature=0.033,
  frictionloss=0.01,
)

CASBOT02_ARM_LIGHT_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "upper_left_3_joint",
    "upper_left_5_joint",
    "upper_right_3_joint",
    "upper_right_5_joint",
  ),
  stiffness=_stiffness(0.0245),
  damping=_damping(0.0245),
  effort_limit=30.0,
  armature=0.0245,
  frictionloss=0.01,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.92),
  joint_pos={
    "leg_l1_joint": -0.185,
    "leg_l2_joint": 0.0,
    "leg_l3_joint": 0.0,
    "leg_l4_joint": 0.36,
    "leg_l5_joint": -0.175,
    "leg_l6_joint": 0.0,
    "leg_r1_joint": -0.185,
    "leg_r2_joint": 0.0,
    "leg_r3_joint": 0.0,
    "leg_r4_joint": 0.36,
    "leg_r5_joint": -0.175,
    "leg_r6_joint": 0.0,
    "upper_left_1_joint": 0.1,
    "upper_left_2_joint": 0.175,
    "upper_left_3_joint": 0.0,
    "upper_left_4_joint": -0.35,
    "upper_left_5_joint": 0.0,
    "upper_right_1_joint": 0.1,
    "upper_right_2_joint": -0.175,
    "upper_right_3_joint": 0.0,
    "upper_right_4_joint": -0.35,
    "upper_right_5_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Final config.
##

CASBOT02_23DOF_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    _with_motor_comm_delay(CASBOT02_LEG_HEAVY_ACTUATOR),
    _with_motor_comm_delay(CASBOT02_LEG_LIGHT_ACTUATOR),
    _with_motor_comm_delay(CASBOT02_ARM_HEAVY_ACTUATOR),
    _with_motor_comm_delay(CASBOT02_ARM_LIGHT_ACTUATOR),
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_casbot02_23dof_robot_cfg() -> EntityCfg:
  """Get a fresh CASBOT02 23DOF robot configuration instance."""
  return EntityCfg(
    init_state=HOME_KEYFRAME,
    spec_fn=get_spec,
    articulation=CASBOT02_23DOF_ARTICULATION,
    sort_actuators=True,
  )


CASBOT02_23DOF_ACTION_SCALE: dict[str, float] = {}
for a in CASBOT02_23DOF_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  assert e is not None
  for n in a.target_names_expr:
    CASBOT02_23DOF_ACTION_SCALE[n] = 0.25 * e / s

CASBOT02_22DOF_NO_WAIST_ACTION_SCALE: dict[str, float] = {
  name: CASBOT02_23DOF_ACTION_SCALE[name]
  for name in CASBOT02_22DOF_NO_WAIST_JOINT_NAMES
}

# 12 腿关节的 action scale（手臂由摆臂公式控制，不经网络）
CASBOT02_LEG_ONLY_ACTION_SCALE: dict[str, float] = {
  name: CASBOT02_23DOF_ACTION_SCALE[name]
  for name in CASBOT02_LEG_ONLY_JOINT_NAMES
}


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_casbot02_23dof_robot_cfg())

  viewer.launch(robot.spec.compile())
