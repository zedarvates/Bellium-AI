import importlib.util
import json
from pathlib import Path

import pytest

from bellium.micro_nn import mlp
from bellium.micro_nn.features import binary_router_values, effort_classifier_values, error_classifier_values
from bellium.micro_nn.legacy import classify_legacy, load_legacy_model
from legacy.micro_nn import calibration


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_corrupt_model_is_rejected_before_replacing_a_file(tmp_path, value):
    model = {"layers": [1, 2], "weights": [[value, 0.]], "biases": [[0., 0.]], "activations": ["softmax"]}
    target = tmp_path / "model.json"
    target.write_text("preserve", encoding="utf-8")
    with pytest.raises(ValueError):
        mlp.save_mlp(target, model)
    assert target.read_text() == "preserve"


@pytest.mark.parametrize("name", ["inpaint_router", "tool_router"])
@pytest.mark.parametrize("accuracy", [0.84, float("nan")])
def test_failed_training_gate_preserves_previous_weights(name, accuracy, tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts" / f"train_{name}.py"
    spec = importlib.util.spec_from_file_location(name, script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = tmp_path / "v0.json"
    target.write_text("previous weights", encoding="utf-8")
    monkeypatch.setattr(module, "DEST", target)
    monkeypatch.setattr(module, "make_samples", lambda *args: [])
    monkeypatch.setattr(module, "train_classifier", lambda *args, **kwargs: {})
    monkeypatch.setattr(module, "accuracy", lambda *args: accuracy)
    with pytest.raises(SystemExit):
        module.main()
    assert target.read_text() == "previous weights"


def test_calibration_is_applied_without_changing_weights(tmp_path, monkeypatch):
    model = load_legacy_model("binary_router")
    model_path = tmp_path / "binary_router.json"
    model_path.write_text(json.dumps(model), encoding="utf-8")
    before = model_path.read_bytes()
    monkeypatch.setattr(calibration, "_MODELS_DIR", tmp_path)
    query = binary_router_values(0.1, 1.0, True)
    original = classify_legacy("binary_router", query)
    calibration.save_temperature("binary_router", 3.0)
    softened = classify_legacy("binary_router", query)
    assert softened.confidence < original.confidence
    assert model_path.read_bytes() == before
    model_path.write_text("changed model", encoding="utf-8")
    with pytest.raises(ValueError, match="match"):
        calibration.load_temperature("binary_router")


def test_unavailable_log_adapter_performs_no_automatic_calibration():
    assert calibration.calibrate_from_logs("binary_router") is None


def test_feature_extractors_preserve_published_contracts():
    assert effort_classifier_values("const x = 1;")["is_code"] == 1.0
    assert error_classifier_values("at main (main.js:10)")["has_traceback"] == 1.0
