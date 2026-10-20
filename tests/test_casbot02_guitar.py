"""Rigid guitar payload must not silently change locomotion training settings."""

from dataclasses import asdict
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
import torch

from mjlab.entity import Entity
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.rl.runner import MjlabOnPolicyRunner
from wbc_mjlab import casbot02_constants as C
from wbc_mjlab.casbot02_config import casbot02_loco_teacher_env_cfg
from wbc_mjlab.casbot02_rl_cfg import casbot02_loco_teacher_runner_cfg
from wbc_mjlab.casbot02_guitar.constants import (
  COLLISION_PREFIX, EXPECTED_COLLISION_PARTS, SOURCE_URDF, WAIST_BODY_NAME,
  VISUAL_MESH_NAME, get_robot_cfg, get_spec,
)
from wbc_mjlab.casbot02_guitar.env_cfg import casbot02_guitar_loco_teacher_env_cfg
from wbc_mjlab.casbot02_guitar.rl_cfg import casbot02_guitar_loco_teacher_runner_cfg
from wbc_mjlab.casbot02_guitar.actions import GuitarLegWithFixedArmsAction, GuitarLegWithFixedArmsActionCfg
from wbc_mjlab.casbot02_guitar import motor_config as M


@pytest.mark.parametrize("play", [False, True])
def test_all_environment_parameters_inherit_current_casbot02(play):
  base = asdict(casbot02_loco_teacher_env_cfg(play=play))
  guitar = asdict(casbot02_guitar_loco_teacher_env_cfg(play=play))
  assert guitar["scene"]["entities"]["robot"]["spec_fn"] is get_spec
  guitar["scene"]["entities"]["robot"]["spec_fn"] = C.get_spec
  # TerrainCfg's default factory creates a fresh (equivalent) lambda each time.
  a, b = base["scene"]["terrain"]["spec_fn"], guitar["scene"]["terrain"]["spec_fn"]
  assert a.__code__ is b.__code__
  guitar["scene"]["terrain"]["spec_fn"] = a
  assert isinstance(casbot02_guitar_loco_teacher_env_cfg().actions["joint_pos"], GuitarLegWithFixedArmsActionCfg)
  assert guitar == base
  assert casbot02_loco_teacher_env_cfg().scene.entities["robot"].spec_fn is C.get_spec


def test_only_runner_log_names_change():
  base = asdict(casbot02_loco_teacher_runner_cfg())
  guitar = asdict(casbot02_guitar_loco_teacher_runner_cfg())
  for field in ("experiment_name", "run_name"):
    assert guitar[field] == "casbot02_guitar_loco_teacher"
    guitar[field] = base[field]
  assert guitar == base


def test_motor_configuration_and_defaults_are_independent():
  baseline = C.get_casbot02_23dof_robot_cfg()
  guitar = get_robot_cfg()
  assert asdict(guitar.init_state) == asdict(baseline.init_state)
  assert asdict(guitar.articulation) == asdict(baseline.articulation)
  assert M.LEG_ACTION_SCALE == C.CASBOT02_LEG_ONLY_ACTION_SCALE
  assert guitar.init_state is not baseline.init_state
  assert guitar.init_state.joint_pos is not baseline.init_state.joint_pos
  for g, b in zip(guitar.articulation.actuators, baseline.articulation.actuators):
    assert g is not b
  guitar.init_state.joint_pos["upper_left_1_joint"] = 9.
  guitar.articulation.actuators[0].stiffness = 1.
  assert get_robot_cfg().init_state.joint_pos["upper_left_1_joint"] == .1
  assert baseline.init_state.joint_pos["upper_left_1_joint"] == .1
  assert get_robot_cfg().articulation.actuators[0].stiffness == baseline.articulation.actuators[0].stiffness


def test_arm_targets_do_not_follow_knees(monkeypatch):
  from mjlab.envs.mdp.actions import JointPositionAction
  from scripts import sim2sim_casbot02_loco_onnx as sim

  # Exercise the real apply method, without constructing a simulation.
  monkeypatch.setattr(JointPositionAction, "apply_actions", lambda self: None)
  term = object.__new__(GuitarLegWithFixedArmsAction)
  term._arm_ids = torch.arange(12, 22)
  term._arm_default = torch.tensor([[M.HOME_KEYFRAME.joint_pos[n] for n in C.CASBOT02_23DOF_JOINT_NAMES[12:]]])
  calls = []
  term._entity = SimpleNamespace(set_joint_position_target=lambda target, **kw: calls.append(target.clone()))
  for _ in range(3):
    term.apply_actions()
  for value in calls:
    torch.testing.assert_close(value, term._arm_default)
  default = sim.make_default_joint_pos(get_robot_cfg())
  data = SimpleNamespace(qpos=np.zeros(29))
  for difference in (-1., 0., 1.):
    data.qpos[7 + sim.LEG_KNEE_L] = difference
    target = default.copy()
    sim.apply_arm_targets(data, target, default, gain=0.)
    np.testing.assert_array_equal(target[12:], default[12:])
  target = default.copy()
  sim.apply_arm_targets(data, target, default)
  assert target[sim.ARM_SHOULDER_L] == default[sim.ARM_SHOULDER_L] + .5


def test_independent_task_registration():
  task = "Casbot02-Guitar-Loco-Teacher-Flat"
  for play in (False, True):
    cfg = load_env_cfg(task, play=play)
    assert cfg.scene.entities["robot"].spec_fn is get_spec
  assert load_rl_cfg(task).experiment_name == "casbot02_guitar_loco_teacher"
  assert load_runner_cls(task) is MjlabOnPolicyRunner


@pytest.fixture(scope="module")
def models():
  return Entity(C.get_casbot02_23dof_robot_cfg()).spec.compile(), Entity(get_robot_cfg()).spec.compile()


def test_mass_full_inertia_and_joint_layout(models):
  base, guitar = models
  link = ET.parse(SOURCE_URDF).find(f"./link[@name='{WAIST_BODY_NAME}']/inertial")
  waist = guitar.body(WAIST_BODY_NAME).id
  original_waist = base.body(WAIST_BODY_NAME).id
  mass = float(link.find("mass").get("value"))
  np.testing.assert_allclose(guitar.body_mass[waist], mass)
  np.testing.assert_allclose(guitar.body_ipos[waist], np.fromstring(link.find("origin").get("xyz"), sep=" "))
  np.testing.assert_allclose(guitar.body_mass.sum() - base.body_mass.sum(), mass - base.body_mass[original_waist])
  np.testing.assert_allclose(guitar.body_mass.sum(), 53.620321)
  inertia = link.find("inertia")
  # URDF provides only the upper triangle of its symmetric inertia tensor.
  expected = np.array([
    [float(inertia.get("ixx")), float(inertia.get("ixy")), float(inertia.get("ixz"))],
    [float(inertia.get("ixy")), float(inertia.get("iyy")), float(inertia.get("iyz"))],
    [float(inertia.get("ixz")), float(inertia.get("iyz")), float(inertia.get("izz"))],
  ])
  R = np.empty(9)
  mujoco.mju_quat2Mat(R, guitar.body_iquat[waist])
  R = R.reshape(3, 3)
  np.testing.assert_allclose(R @ np.diag(guitar.body_inertia[waist]) @ R.T, expected, atol=1e-8)
  assert guitar.nu == base.nu == 22
  assert guitar.njnt == base.njnt == 23
  for kind, count in ((mujoco.mjtObj.mjOBJ_JOINT, base.njnt), (mujoco.mjtObj.mjOBJ_ACTUATOR, base.nu)):
    assert [mujoco.mj_id2name(guitar, kind, i) for i in range(count)] == [
      mujoco.mj_id2name(base, kind, i) for i in range(count)]
  for name in ("left_foot", "right_foot"):
    np.testing.assert_array_equal(guitar.site(name).pos, base.site(name).pos)
  np.testing.assert_array_equal(guitar.actuator_gainprm, base.actuator_gainprm)
  np.testing.assert_array_equal(guitar.actuator_biasprm, base.actuator_biasprm)
  np.testing.assert_array_equal(guitar.actuator_forcerange, base.actuator_forcerange)
  np.testing.assert_array_equal(guitar.dof_armature, base.dof_armature)


def test_guitar_colliders_and_no_home_self_contact(models):
  base, guitar = models
  names = [mujoco.mj_id2name(guitar, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(guitar.ngeom)]
  assert sum(n.startswith(COLLISION_PREFIX) for n in names) == EXPECTED_COLLISION_PARTS
  visual = guitar.mesh(VISUAL_MESH_NAME).id
  visual_geoms = np.flatnonzero(guitar.geom_dataid == visual)
  assert len(visual_geoms) == 1
  assert not guitar.geom_contype[visual_geoms[0]]
  assert not guitar.geom_conaffinity[visual_geoms[0]]
  for model in (base, guitar):
    data = mujoco.MjData(model)
    data.qpos[:3] = C.HOME_KEYFRAME.pos
    data.qpos[7:] = [C.HOME_KEYFRAME.joint_pos[n] for n in C.CASBOT02_23DOF_JOINT_NAMES]
    mujoco.mj_forward(model, data)
    assert data.ncon == 0


def test_sim2sim_uses_guitar_and_keeps_baseline_default():
  from scripts import sim2sim_casbot02_guitar_loco_onnx as guitar_sim
  from scripts import sim2sim_casbot02_loco_onnx as base_sim

  guitar = guitar_sim.build_model()
  base = base_sim.build_model()
  np.testing.assert_allclose(guitar.body_mass.sum() - base.body_mass.sum(), 2.96639)
  assert guitar.geom("ground").matid == base.geom("ground").matid
  assert guitar.opt.timestep == base.opt.timestep == .005


def test_sim2sim_entry_disables_swing_and_uses_independent_scale(monkeypatch):
  from scripts import sim2sim_casbot02_guitar_loco_onnx as sim

  monkeypatch.setattr(sim.base, "parse_args", lambda: SimpleNamespace(
    model_path="guitar.onnx", log_csv="", log_decimation=1, gait_period=1.2))
  calls = []
  monkeypatch.setattr(sim.base, "run", lambda *args, **kwargs: calls.append(kwargs))
  sim.main()
  assert calls[0]["arm_swing_gain"] == 0.
  assert calls[0]["robot_cfg"].spec_fn is get_spec
  np.testing.assert_array_equal(calls[0]["action_scale"], [
    M.LEG_ACTION_SCALE[n] for n in C.CASBOT02_LEG_ONLY_JOINT_NAMES])
