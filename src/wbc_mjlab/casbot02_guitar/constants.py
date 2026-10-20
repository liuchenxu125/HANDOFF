"""CASBOT02 locomotion robot carrying the supplied rigid guitar assembly.

The supplied URDF, unlike its stale MJCF, contains the guitar mass/inertia and
mesh. Only that waist assembly is imported: the existing loco robot's joints,
motors, default pose, foot sites, and other geometry remain unchanged.
"""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from mjlab.entity import EntityCfg
from wbc_mjlab import casbot02_constants as C
from . import motor_config as M


SOURCE_ROOT = Path(__file__).resolve().parents[3] / "CASBOT02_ENCOS_7dof_shell_guitar" / "Serial"
SOURCE_URDF = SOURCE_ROOT / "urdf" / "CASBOT02_ENCOS_7dof_shell_20251015.urdf"
WAIST_BODY_NAME = "waist_yaw_link"
VISUAL_MESH_NAME = "casbot02_guitar_waist_visual"
COLLISION_PREFIX = "casbot02_guitar_collision_"
EXPECTED_COLLISION_PARTS = 8


def _origin(node: ET.Element | None) -> tuple[np.ndarray, np.ndarray]:
  """URDF fixed-axis RPY to body-frame position and wxyz quaternion."""
  pos = np.fromstring(node.get("xyz", "0 0 0"), sep=" ") if node is not None else np.zeros(3)
  angles = np.fromstring(node.get("rpy", "0 0 0"), sep=" ") if node is not None else np.zeros(3)
  if pos.shape != (3,) or angles.shape != (3,):
    raise ValueError("Guitar URDF origin requires three xyz and rpy values")
  cr, cp, cy = np.cos(angles * .5)
  sr, sp, sy = np.sin(angles * .5)
  quat = np.array([
    cr * cp * cy + sr * sp * sy,
    sr * cp * cy - cr * sp * sy,
    cr * sp * cy + sr * cp * sy,
    cr * cp * sy - sr * sp * cy,
  ])
  return pos, quat


def _rotation(quat: np.ndarray) -> np.ndarray:
  result = np.empty(9)
  mujoco.mju_quat2Mat(result, quat)
  return result.reshape(3, 3)


@lru_cache(maxsize=4)
def _collision_components(
  mesh_path: str,
  mesh_mtime: int,
  base_mesh_path: str,
  base_mtime: int,
  pos: tuple[float, ...],
  quat: tuple[float, ...],
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
  """Separate the guitar's CAD components from the unchanged torso housing.

  In this supplied assembly every added solid is in front of the original
  waist's maximum x. Keeping the original torso collider plus these components
  avoids a single torso+guitar convex hull filling the arm clearance. Tiny
  degenerate two-triangle CAD fragments remain visible but are not colliders.
  This is component separation, not a general concave-mesh decomposition.
  """
  del mesh_mtime, base_mtime  # File timestamps invalidate the cache.
  import trimesh  # Already required by the project's yourdfpy dependency.

  assembly = trimesh.load_mesh(mesh_path, process=True)
  base = trimesh.load_mesh(base_mesh_path, process=True)
  front = float(base.bounds[1, 0])
  R = _rotation(np.asarray(quat))
  components = []
  for component in assembly.split(only_watertight=False):
    vertices = np.asarray(component.vertices) @ R.T + np.asarray(pos)
    if vertices[:, 0].min() <= front + 1e-6:
      continue
    if len(vertices) < 4 or np.linalg.matrix_rank(vertices - vertices[0], tol=1e-9) < 3:
      continue
    components.append((vertices.copy(), np.asarray(component.faces).copy()))
  if len(components) != EXPECTED_COLLISION_PARTS:
    raise ValueError(
      f"Expected {EXPECTED_COLLISION_PARTS} separate guitar collision solids, "
      f"found {len(components)} in {mesh_path}. Check the supplied CAD layout."
    )
  return tuple(components)


def get_spec() -> mujoco.MjSpec:
  """Base loco MJCF plus URDF waist mass/inertia and rigid guitar geometry."""
  if not SOURCE_URDF.is_file():
    raise FileNotFoundError(f"Guitar model URDF missing: {SOURCE_URDF}")
  link = ET.parse(SOURCE_URDF).find(f"./link[@name='{WAIST_BODY_NAME}']")
  if link is None:
    raise ValueError(f"{SOURCE_URDF} does not contain {WAIST_BODY_NAME}")
  inertial = link.find("inertial")
  visual = link.find("visual")
  collision = link.find("collision")
  if inertial is None or visual is None or collision is None:
    raise ValueError("Guitar waist requires inertial, visual, and collision data")
  inertia_node = inertial.find("inertia")
  mass_node = inertial.find("mass")
  visual_mesh = visual.find("geometry/mesh")
  collision_mesh = collision.find("geometry/mesh")
  if any(node is None for node in (inertia_node, mass_node, visual_mesh, collision_mesh)):
    raise ValueError("Guitar waist inertia/mass/mesh metadata is incomplete")
  visual_path = (SOURCE_URDF.parent / visual_mesh.get("filename")).resolve()
  collision_path = (SOURCE_URDF.parent / collision_mesh.get("filename")).resolve()
  for node in (visual_mesh, collision_mesh):
    scale = np.fromstring(node.get("scale", "1 1 1"), sep=" ")
    if scale.shape != (3,) or not np.allclose(scale, 1.):
      raise ValueError("Guitar CAD component separation expects unit mesh scale")
  com, inertia_quat = _origin(inertial.find("origin"))
  I = np.array([
    [float(inertia_node.get("ixx")), float(inertia_node.get("ixy")), float(inertia_node.get("ixz"))],
    [float(inertia_node.get("ixy")), float(inertia_node.get("iyy")), float(inertia_node.get("iyz"))],
    [float(inertia_node.get("ixz")), float(inertia_node.get("iyz")), float(inertia_node.get("izz"))],
  ])
  R = _rotation(inertia_quat)
  I = R @ I @ R.T
  eigenvalues = np.linalg.eigvalsh(I)
  mass = float(mass_node.get("value"))
  if mass <= 0. or eigenvalues[0] <= 0. or eigenvalues[2] > eigenvalues[0] + eigenvalues[1]:
    raise ValueError("Guitar URDF mass/inertia is not physically valid")

  spec = C.get_spec()
  waist = spec.body(WAIST_BODY_NAME)
  waist.mass = mass  # Already torso+guitar; do not add the guitar a second time.
  waist.ipos = com
  waist.iquat = [1., 0., 0., 0.]
  waist.inertia = [0., 0., 0.]
  waist.fullinertia = [I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]]

  old_geoms = [g for g in waist.geoms if g.meshname == WAIST_BODY_NAME]
  visuals = [g for g in old_geoms if not (g.contype or g.conaffinity)]
  colliders = [g for g in old_geoms if g.contype or g.conaffinity]
  if len(visuals) != 1 or len(colliders) != 1:
    raise ValueError("Expected one original waist visual and one waist collider")
  spec.add_mesh(name=VISUAL_MESH_NAME, file=str(visual_path))
  visuals[0].meshname = VISUAL_MESH_NAME
  visuals[0].pos, visuals[0].quat = _origin(visual.find("origin"))

  cpos, cquat = _origin(collision.find("origin"))
  base_mesh = C.CASBOT02_23DOF_XML.parent.parent / "meshes" / "waist_yaw_link.STL"
  components = _collision_components(
    str(collision_path), collision_path.stat().st_mtime_ns,
    str(base_mesh), base_mesh.stat().st_mtime_ns, tuple(cpos), tuple(cquat),
  )
  template = colliders[0]
  contact_params = {key: getattr(template, key) for key in (
    "contype", "conaffinity", "condim", "priority", "friction", "solref", "solimp", "solmix", "margin", "gap"
  )}
  for i, (vertices, faces) in enumerate(components):
    name = f"{COLLISION_PREFIX}{i}"
    spec.add_mesh(name=name, uservert=vertices.ravel().tolist(), userface=faces.ravel().tolist())
    waist.add_geom(
      name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name,
      group=0, density=0., **contact_params,
    )
  return spec


def get_robot_cfg() -> EntityCfg:
  """Independent motors/default pose, initially identical to current Casbot02."""
  return EntityCfg(
    init_state=deepcopy(M.HOME_KEYFRAME), spec_fn=get_spec,
    articulation=deepcopy(M.ARTICULATION), sort_actuators=True,
  )
