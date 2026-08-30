"""CASBOT02AMPJ 27-DoF AMP-teacher sim2sim with an ONNX policy.

The actor observation follows mjlab's term-major four-frame layout:
angular velocity, projected gravity, velocity command, relative joint
position, relative joint velocity, and the previous action.  Every joint in
``casbot02ampj.constants.JOINT_NAMES`` is controlled by the policy; the head
is fixed and the dexterous-hand bodies are absent from the derived model.

Keyboard: Up/Down = vx, Left/Right = yaw rate, Space = stop.
"""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

from mjlab.entity import Entity

from wbc_mjlab.casbot02ampj import constants as C


REPO_ROOT = Path(__file__).resolve().parent.parent

SIM_TIMESTEP = 0.005
POLICY_DECIMATION = 4
HISTORY_LENGTH = 4
NUM_JOINTS = len(C.JOINT_NAMES)
SINGLE_FRAME_OBS_SIZE = 9 + 3 * NUM_JOINTS
TERM_DIMS = (3, 3, 3, NUM_JOINTS, NUM_JOINTS, NUM_JOINTS)

DEFAULT_COMMAND = (0.5, 0.0, 0.0)
COMMAND_X_RANGE = (-2.0, 3.0)
COMMAND_YAW_RANGE = (-0.7, 0.7)
COMMAND_X_STEP = 0.1
COMMAND_YAW_STEP = 0.1


def find_latest_onnx() -> Path:
  export_root = REPO_ROOT / "logs" / "rsl_rl" / "casbot02_ampj_teacher_flat"
  if not export_root.is_dir():
    raise FileNotFoundError(f"No AMPJ training output under {export_root}")

  fallback: Path | None = None
  for run_dir in sorted(export_root.iterdir(), reverse=True):
    onnx_files = sorted(run_dir.glob("*.onnx"))
    if not onnx_files:
      continue
    if fallback is None:
      fallback = onnx_files[-1]

    checkpoint_iterations = []
    for checkpoint in run_dir.glob("model_*.pt"):
      try:
        checkpoint_iterations.append(int(checkpoint.stem.removeprefix("model_")))
      except ValueError:
        continue
    if checkpoint_iterations and max(checkpoint_iterations) > 0:
      return onnx_files[-1]

  if fallback is not None:
    print(
      "[ONNX] warning: only iteration-0 exports were found; "
      "the selected policy is untrained"
    )
    return fallback
  raise FileNotFoundError(f"No CASBOT02AMPJ ONNX under {export_root}")


def make_default_joint_pos() -> np.ndarray:
  joint_pos = C.HOME_KEYFRAME.joint_pos or {}
  fallback = float(joint_pos.get(".*", 0.0))
  return np.asarray(
    [float(joint_pos.get(name, fallback)) for name in C.JOINT_NAMES],
    dtype=np.float64,
  )


def make_action_scale() -> np.ndarray:
  return np.asarray([C.ACTION_SCALE[name] for name in C.JOINT_NAMES])


def quat_apply_inverse_wxyz(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
  w, x, y, z = quat
  inverse_vec = -np.asarray((x, y, z), dtype=np.float64)
  cross = 2.0 * np.cross(inverse_vec, vec)
  return vec + w * cross + np.cross(inverse_vec, cross)


class OnnxPolicy:
  def __init__(self, model_path: Path):
    import onnxruntime as ort

    self.session = ort.InferenceSession(
      str(model_path), providers=["CPUExecutionProvider"]
    )
    model_input = self.session.get_inputs()[0]
    model_output = self.session.get_outputs()[0]
    self.input_name = model_input.name
    self.output_name = model_output.name
    self.input_dim = model_input.shape[1]
    self.output_dim = model_output.shape[1]
    print(f"[ONNX] input={self.input_dim}, output={self.output_dim}")

  def __call__(self, obs: np.ndarray) -> np.ndarray:
    obs = np.ascontiguousarray(obs.astype(np.float32))
    output = self.session.run(
      [self.output_name], {self.input_name: obs}
    )[0]
    return np.asarray(output, dtype=np.float32).reshape(-1)


def build_model() -> mujoco.MjModel:
  entity = Entity(C.get_robot_cfg())
  spec = entity.spec

  checker = spec.add_texture(name="texplane")
  checker.type = mujoco.mjtTexture.mjTEXTURE_2D
  checker.builtin = mujoco.mjtBuiltin.mjBUILTIN_CHECKER
  checker.rgb1 = (0.2, 0.3, 0.4)
  checker.rgb2 = (0.1, 0.15, 0.2)
  checker.mark = mujoco.mjtMark.mjMARK_CROSS
  checker.markrgb = (0.8, 0.8, 0.8)
  checker.width = 512
  checker.height = 512

  ground_material = spec.add_material(name="matplane")
  ground_material.textures[
    mujoco.mjtTextureRole.mjTEXROLE_RGB.value
  ] = checker.name
  ground_material.texrepeat = (1.0, 1.0)
  ground_material.texuniform = True
  ground_material.reflectance = 0.0

  ground = spec.worldbody.add_geom(
    name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE
  )
  ground.size = (0.0, 0.0, 1.0)
  ground.condim = 3
  ground.friction = (1.0, 0.005, 0.0001)
  ground.material = ground_material.name
  model = spec.compile()
  model.opt.timestep = SIM_TIMESTEP

  actuator_names = tuple(model.actuator(i).name for i in range(model.nu))
  if actuator_names != C.JOINT_NAMES:
    raise RuntimeError(
      "Actuator order differs from the training action order:\n"
      f"expected={C.JOINT_NAMES}\nactual={actuator_names}"
    )
  return model


def get_obs_frame(
  data: mujoco.MjData,
  command: np.ndarray,
  last_action: np.ndarray,
  default_joint_pos: np.ndarray,
) -> np.ndarray:
  base_ang_vel = np.asarray(
    data.sensor("angular-velocity").data, dtype=np.float64
  ).copy()
  quat_wxyz = np.asarray(data.qpos[3:7], dtype=np.float64)
  projected_gravity = quat_apply_inverse_wxyz(
    quat_wxyz, np.asarray((0.0, 0.0, -1.0), dtype=np.float64)
  )
  joint_pos_rel = np.asarray(data.qpos[7:], dtype=np.float64) - default_joint_pos
  joint_vel = np.asarray(data.qvel[6:], dtype=np.float64)
  frame = np.concatenate(
    (
      base_ang_vel,
      projected_gravity,
      command,
      joint_pos_rel,
      joint_vel,
      last_action,
    )
  )
  if frame.shape != (SINGLE_FRAME_OBS_SIZE,):
    raise RuntimeError(
      f"Expected one observation frame of {SINGLE_FRAME_OBS_SIZE}, got {frame.shape}"
    )
  return frame.astype(np.float32)


def build_obs_term_major(history: deque[np.ndarray]) -> np.ndarray:
  frames = np.asarray(history)  # oldest -> newest
  parts: list[np.ndarray] = []
  offset = 0
  for dim in TERM_DIMS:
    parts.append(frames[:, offset : offset + dim].reshape(-1))
    offset += dim
  return np.concatenate(parts).reshape(1, -1)


def install_command_controls(viewer, command: np.ndarray) -> None:
  import glfw

  keys = {
    glfw.KEY_UP,
    glfw.KEY_DOWN,
    glfw.KEY_LEFT,
    glfw.KEY_RIGHT,
    glfw.KEY_SPACE,
  }
  original_callback = viewer._key_callback

  def callback(window, key, scancode, action, mods):
    if key in keys and action in (glfw.PRESS, glfw.REPEAT):
      if key == glfw.KEY_UP:
        command[0] = np.clip(
          command[0] + COMMAND_X_STEP, *COMMAND_X_RANGE
        )
      elif key == glfw.KEY_DOWN:
        command[0] = np.clip(
          command[0] - COMMAND_X_STEP, *COMMAND_X_RANGE
        )
      elif key == glfw.KEY_LEFT:
        command[2] = np.clip(
          command[2] + COMMAND_YAW_STEP, *COMMAND_YAW_RANGE
        )
      elif key == glfw.KEY_RIGHT:
        command[2] = np.clip(
          command[2] - COMMAND_YAW_STEP, *COMMAND_YAW_RANGE
        )
      else:
        command[:] = 0.0
      print(
        f"[command] vx={command[0]:+.2f} vy={command[1]:+.2f} "
        f"yaw={command[2]:+.2f}"
      )
      return
    original_callback(window, key, scancode, action, mods)

  glfw.set_key_callback(viewer.window, callback)


def run(model_arg: str = "") -> None:
  model_path = Path(model_arg) if model_arg else find_latest_onnx()
  if not model_path.is_absolute():
    model_path = REPO_ROOT / model_path
  if not model_path.is_file():
    raise FileNotFoundError(model_path)

  print(f"[ONNX] model={model_path}")
  policy = OnnxPolicy(model_path)
  expected_input = SINGLE_FRAME_OBS_SIZE * HISTORY_LENGTH
  if policy.input_dim != expected_input or policy.output_dim != NUM_JOINTS:
    raise RuntimeError(
      f"Expected ONNX {expected_input} -> {NUM_JOINTS}, got "
      f"{policy.input_dim} -> {policy.output_dim}"
    )

  model = build_model()
  data = mujoco.MjData(model)
  default_joint_pos = make_default_joint_pos()
  action_scale = make_action_scale()
  ctrl_low = model.actuator_ctrlrange[:, 0]
  ctrl_high = model.actuator_ctrlrange[:, 1]
  command = np.asarray(DEFAULT_COMMAND, dtype=np.float64)

  mujoco.mj_resetData(model, data)
  data.qpos[0:3] = np.asarray(C.HOME_KEYFRAME.pos, dtype=np.float64)
  data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
  data.qpos[7:] = default_joint_pos
  mujoco.mj_forward(model, data)

  history: deque[np.ndarray] = deque(maxlen=HISTORY_LENGTH)
  last_action = np.zeros(NUM_JOINTS, dtype=np.float32)
  target_pos = default_joint_pos.copy()

  import mujoco_viewer

  viewer = mujoco_viewer.MujocoViewer(model, data, mode="window")
  viewer.cam.distance = 4.0
  viewer.cam.azimuth = 45.0
  viewer.cam.elevation = -20.0
  install_command_controls(viewer, command)

  import time

  physics_step = 0
  next_time = time.perf_counter()
  policy_dt = SIM_TIMESTEP * POLICY_DECIMATION
  print("[sim2sim] Up/Down=vx  Left/Right=yaw  Space=stop")
  print(
    f"[sim2sim] dt={SIM_TIMESTEP}, decimation={POLICY_DECIMATION}, "
    f"policy_hz={1.0 / policy_dt:.1f}"
  )
  try:
    while viewer.is_alive:
      if physics_step % POLICY_DECIMATION == 0:
        frame = get_obs_frame(
          data, command, last_action, default_joint_pos
        )
        if not history:
          for _ in range(HISTORY_LENGTH):
            history.append(frame)
        else:
          history.append(frame)
        action = policy(build_obs_term_major(history))
        last_action = action.copy()
        target_pos = np.clip(
          default_joint_pos + action * action_scale,
          ctrl_low,
          ctrl_high,
        )

      data.ctrl[:] = target_pos
      mujoco.mj_step(model, data)
      physics_step += 1
      viewer.render()
      viewer.cam.lookat = data.qpos[0:3]

      next_time += SIM_TIMESTEP
      sleep_s = next_time - time.perf_counter()
      if sleep_s > 0.0:
        time.sleep(sleep_s)
      else:
        next_time = time.perf_counter()
  finally:
    viewer.close()


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="CASBOT02AMPJ 27-DoF AMP teacher sim2sim"
  )
  parser.add_argument(
    "model_path", nargs="?", default="", help="ONNX path (empty = latest)"
  )
  return parser.parse_args()


if __name__ == "__main__":
  run(parse_args().model_path)
