import copy
import random

import pytest

from bellium.micro_nn.calibration import fit_temperature
from bellium.micro_nn.mlp import load_mlp
from bellium.specialists import tool_router
from scripts.train_tool_router import make_samples


def _query():
    return {"features": {"has_code": 1.0, "has_files": 1.0, "has_error": 0.0,
                         "wants_mutation": 0.0, "wants_external_state": 0.0, "criticality": 0.32}}


@pytest.fixture
def identity():
    model = load_mlp(tool_router.MODEL_PATH)
    return fit_temperature(model, make_samples(100, random.Random(318)),
                           min_temperature=1.0, max_temperature=1.0)


def test_explicit_identity_preserves_default_decisions_and_exposes_trace(identity):
    query = _query()
    before = copy.deepcopy(identity)
    raw = tool_router.classify_tool_need(query)
    calibrated = tool_router.classify_tool_need(query, calibration=identity)
    assert raw.output["label"] == calibrated.output["label"]
    assert raw.output["probabilities"] == calibrated.output["probabilities"]
    assert raw.confidence == calibrated.confidence
    assert "calibration" not in raw.output
    assert calibrated.output["calibration"]["temperature"] == 1.0
    assert identity == before
    default_hybrid = tool_router.route_tool(query)
    optional_hybrid = tool_router.route_tool(query, calibration=identity)
    assert default_hybrid.output["tool"] == optional_hybrid.output["tool"]
    assert default_hybrid.confidence == optional_hybrid.confidence


def test_supported_input_rejects_mismatched_model_artifact(identity):
    identity["model_fingerprint"] = "wrong-model"
    with pytest.raises(ValueError, match="does not match"):
        tool_router.classify_tool_need(_query(), calibration=identity)


def test_veto_precedes_even_an_invalid_calibration(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("deterministic veto must not consult calibration")
    monkeypatch.setattr(tool_router, "predict_calibrated", forbidden)
    query = _query()
    query["features"]["wants_mutation"] = 0.5
    result = tool_router.route_tool(query, calibration={"schema": "invalid"})
    assert result.output["tool"] == "escalate"
    assert result.output["micro_nn"] is None


def test_out_of_domain_abstains_before_even_an_invalid_calibration(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("unsupported domain must not consult calibration")
    monkeypatch.setattr(tool_router, "predict_calibrated", forbidden)
    query = _query()
    query["features"]["has_code"] = 0.5
    result = tool_router.classify_tool_need(query, calibration={"schema": "invalid"})
    assert result.abstained and result.output["reason"] == "outside_training_domain"
    assert "calibration" not in result.output
