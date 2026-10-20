"""Independent guitar motor/default-pose snapshot of the current Casbot02.

Edit this file for guitar-specific motor tuning. No actuator objects, action
scale dictionaries or home-pose dictionaries are shared with the base robot.
"""

from dataclasses import replace

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535
DAMPING_RATIO = 2.0
MOTOR_COMM_DELAY_MIN_LAG = 0
MOTOR_COMM_DELAY_MAX_LAG = 2
MOTOR_COMM_DELAY_UPDATE_PERIOD = 4
MOTOR_COMM_DELAY_HOLD_PROB = 0.5


def _stiffness(armature: float) -> float:
  return armature * NATURAL_FREQ**2


def _damping(armature: float) -> float:
  return 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ


def _with_motor_comm_delay(cfg: BuiltinPositionActuatorCfg) -> BuiltinPositionActuatorCfg:
  return replace(
    cfg, delay_min_lag=MOTOR_COMM_DELAY_MIN_LAG,
    delay_max_lag=MOTOR_COMM_DELAY_MAX_LAG,
    delay_update_period=MOTOR_COMM_DELAY_UPDATE_PERIOD,
    delay_hold_prob=MOTOR_COMM_DELAY_HOLD_PROB, delay_per_env_phase=True,
  )


LEG_HEAVY_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "leg_l1_joint", "leg_l2_joint", "leg_l4_joint",
    "leg_r1_joint", "leg_r2_joint", "leg_r4_joint",
  ),
  stiffness=_stiffness(0.07), damping=_damping(0.07),
  effort_limit=120.0, armature=0.07, frictionloss=0.01,
)
LEG_LIGHT_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "leg_l3_joint", "leg_l5_joint", "leg_l6_joint",
    "leg_r3_joint", "leg_r5_joint", "leg_r6_joint",
  ),
  stiffness=_stiffness(0.029), damping=5.0,
  effort_limit=80.0, armature=0.029, frictionloss=0.01,
)
ARM_HEAVY_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "upper_left_1_joint", "upper_left_2_joint", "upper_left_4_joint",
    "upper_right_1_joint", "upper_right_2_joint", "upper_right_4_joint",
  ),
  stiffness=_stiffness(0.033), damping=_damping(0.033),
  effort_limit=40.0, armature=0.033, frictionloss=0.01,
)
ARM_LIGHT_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "upper_left_3_joint", "upper_left_5_joint",
    "upper_right_3_joint", "upper_right_5_joint",
  ),
  stiffness=_stiffness(0.0245), damping=_damping(0.0245),
  effort_limit=30.0, armature=0.0245, frictionloss=0.01,
)

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.92),
  joint_pos={
    "leg_l1_joint": -0.185, "leg_l2_joint": 0.0, "leg_l3_joint": 0.0,
    "leg_l4_joint": 0.36, "leg_l5_joint": -0.175, "leg_l6_joint": 0.0,
    "leg_r1_joint": -0.185, "leg_r2_joint": 0.0, "leg_r3_joint": 0.0,
    "leg_r4_joint": 0.36, "leg_r5_joint": -0.175, "leg_r6_joint": 0.0,
    "upper_left_1_joint": 0.1, "upper_left_2_joint": 0.175,
    "upper_left_3_joint": 0.0, "upper_left_4_joint": -0.35,
    "upper_left_5_joint": 0.0,
    "upper_right_1_joint": 0.1, "upper_right_2_joint": -0.175,
    "upper_right_3_joint": 0.0, "upper_right_4_joint": -0.35,
    "upper_right_5_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

ARTICULATION = EntityArticulationInfoCfg(
  actuators=tuple(_with_motor_comm_delay(cfg) for cfg in (
    LEG_HEAVY_ACTUATOR, LEG_LIGHT_ACTUATOR, ARM_HEAVY_ACTUATOR, ARM_LIGHT_ACTUATOR,
  )),
  soft_joint_pos_limit_factor=0.9,
)

ACTION_SCALE = {
  name: 0.25 * cfg.effort_limit / cfg.stiffness
  for cfg in ARTICULATION.actuators for name in cfg.target_names_expr
}
LEG_ACTION_SCALE = {name: value for name, value in ACTION_SCALE.items() if name.startswith("leg_")}
