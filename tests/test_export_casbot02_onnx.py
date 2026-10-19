"""Export the training clock, without changing the policy or its other metadata."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import onnx
import pytest
from onnx import TensorProto, helper

spec = importlib.util.spec_from_file_location(
  "casbot_export", Path(__file__).resolve().parents[1] / "scripts/export_casbot02_onnx.py"
)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


@pytest.mark.parametrize("period", [1.0, 4.0 / 3.0, 1.2])
def test_metadata_uses_saved_actor_clock_instead_of_live_or_reward_clock(tmp_path, period):
  params = tmp_path / "params"
  params.mkdir()
  (params / "env.yaml").write_text(
    "observations:\n  actor:\n    terms:\n      phase:\n        params:\n"
    f"          gait_period: {period}\n"
    "rewards:\n  gait_phase_contact:\n    params:\n      gait_period: 9.0\n"
    "unrelated: !!python/tuple [1, 2]\n"
  )
  cfg = NS(observations={"actor": NS(terms={"phase": NS(params={"gait_period": 2.0})})})
  assert export.checkpoint_gait_period(tmp_path / "model.pt", cfg, None) == period


def test_missing_snapshot_does_not_silently_label_old_model_with_live_period(tmp_path):
  cfg = NS(observations={"actor": NS(terms={"phase": NS(params={"gait_period": 1.2})})})
  with pytest.raises(ValueError, match="Training snapshot missing"):
    export.checkpoint_gait_period(tmp_path / "model.pt", cfg, None)
  assert export.checkpoint_gait_period(tmp_path / "model.pt", cfg, 1.0) == 1.0


def test_policy_without_phase_does_not_get_a_live_phase_period(tmp_path):
  (tmp_path / "params").mkdir()
  (tmp_path / "params/env.yaml").write_text("observations:\n  actor:\n    terms: {}\n")
  cfg = NS(observations={"actor": NS(terms={"phase": object()})})
  assert export.checkpoint_gait_period(tmp_path / "model.pt", cfg, None) is None


def test_metadata_update_preserves_policy_graph_and_other_properties(tmp_path):
  graph = helper.make_graph(
    [helper.make_node("Identity", ["obs"], ["actions"])], "policy",
    [helper.make_tensor_value_info("obs", TensorProto.FLOAT, [1, 2])],
    [helper.make_tensor_value_info("actions", TensorProto.FLOAT, [1, 2])],
  )
  model = helper.make_model(graph)
  helper.set_model_props(model, {"history_ordering": "term", "gait_period": "1.0"})
  path = tmp_path / "policy.onnx"
  onnx.save(model, path)
  export.write_gait_period(path, 1.2)
  updated = onnx.load(path)
  assert updated.graph.SerializeToString() == model.graph.SerializeToString()
  assert {p.key: p.value for p in updated.metadata_props} == {
    "history_ordering": "term", "gait_period": "1.2",
  }


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_period_is_rejected(value):
  with pytest.raises(ValueError, match="finite and positive"):
    export.positive_period(value)
