import copy
import math

import pytest

from bellium.micro_nn.calibration import (
    fit_temperature, model_fingerprint, predict_calibrated, predict_logits, probability_metrics,
)
from bellium.micro_nn.mlp import predict_mlp


def _model():
    return {"layers": [1, 2], "weights": [[4.0, -4.0]], "biases": [[0.0, 0.0]],
            "activations": ["softmax"], "labels": ["first", "second"]}


def _samples():
    return [([1.0], 0)] * 40 + [([1.0], 1)] * 10


def test_fit_corrects_known_overconfidence_without_mutating_model_or_samples():
    model, samples = _model(), _samples()
    before = copy.deepcopy((model, samples))
    calibration = fit_temperature(model, samples)
    assert (model, samples) == before
    assert calibration["temperature"] == pytest.approx(8.0 / math.log(4.0), rel=1e-6)
    assert calibration["class_support"] == [40, 10]
    assert calibration["calibration_nll_after"] < calibration["calibration_nll_before"]
    assert predict_calibrated(model, [1.0], calibration)[0] == pytest.approx(0.8, abs=1e-6)
    assert not calibration["at_search_bound"]
    assert fit_temperature(model, samples) == calibration


def test_identity_keeps_existing_inference_and_ranking():
    model = _model()
    calibration = fit_temperature(model, _samples(), min_temperature=1.0, max_temperature=1.0)
    for vector in ([1.0], [-1.0], [0.0]):
        assert predict_calibrated(model, vector, calibration) == predict_mlp(model, vector)
    softened = fit_temperature(model, _samples())
    for vector in ([1.0], [-1.0], [0.0]):
        raw = predict_mlp(model, vector)
        calibrated = predict_calibrated(model, vector, softened)
        assert max(range(2), key=raw.__getitem__) == max(range(2), key=calibrated.__getitem__)


def test_stable_loss_does_not_clip_underflowed_probabilities():
    result = probability_metrics([[1000.0, -1000.0]], [1])
    assert result["nll"] == 2000.0
    assert result["brier"] == 2.0
    assert result["ece"] == 1.0


def test_metrics_have_correct_bins_and_abstention_denominators():
    # Confidence 0.8, one correct and one incorrect; plus one 0.5 tie abstention.
    rows = [[math.log(4.0), 0.0], [math.log(4.0), 0.0], [0.0, 0.0]]
    result = probability_metrics(rows, [0, 1, 1], bins=2)
    assert result["class_support"] == [1, 2]
    assert result["accepted"] == 2 and result["abstained"] == 1
    assert result["coverage"] == 2 / 3 and result["selective_accuracy"] == 0.5
    assert result["ece"] == pytest.approx(0.7 - 1 / 3)
    assert result["reliability_bins"][0]["samples"] == 0
    assert result["reliability_bins"][0]["mean_confidence"] is None


def test_model_changes_invalidate_calibration_and_dict_order_does_not():
    model = _model()
    calibration = fit_temperature(model, _samples())
    assert model_fingerprint(dict(reversed(list(model.items())))) == model_fingerprint(model)
    model["weights"][0][0] += 0.01
    with pytest.raises(ValueError, match="does not match"):
        predict_calibrated(model, [1.0], calibration)


@pytest.mark.parametrize("change", [{"schema": "other"}, {"classes": 3},
                                     {"temperature": 0}, {"temperature": True}])
def test_invalid_artifact_is_rejected(change):
    model = _model()
    artifact = fit_temperature(model, _samples())
    artifact.update(change)
    with pytest.raises(ValueError):
        predict_calibrated(model, [1.0], artifact)


@pytest.mark.parametrize("logits,labels", [
    ([], []), ([[1, 0]], []), ([[1, 0], [0, 1]], [0]),
    ([[1, 0], [1, 0, 0]], [0, 1]), ([[1]], [0]),
    ([[math.nan, 0]], [0]), ([[True, 0]], [0]),
    ([[1, 0]], [True]), ([[1, 0]], [2]), ([[1, 0]], [-1]),
])
def test_invalid_metric_data_is_rejected(logits, labels):
    with pytest.raises(ValueError):
        probability_metrics(logits, labels)


@pytest.mark.parametrize("options", [{"temperature": 0}, {"temperature": math.inf},
                                      {"bins": 0}, {"bins": True},
                                      {"confidence_threshold": math.nan}])
def test_invalid_metric_controls_are_rejected(options):
    with pytest.raises(ValueError):
        probability_metrics([[1.0, 0.0]], [0], **options)


def test_fit_requires_every_class_and_softmax():
    with pytest.raises(ValueError, match="support"):
        fit_temperature(_model(), [([1.0], 0)])
    model = _model()
    model["activations"] = ["linear"]
    with pytest.raises(ValueError, match="softmax"):
        predict_logits(model, [1.0])


@pytest.mark.parametrize("options", [{"min_temperature": 2.0}, {"max_temperature": 0.5},
                                      {"iterations": True}, {"iterations": 0}])
def test_invalid_search_controls_are_rejected(options):
    with pytest.raises(ValueError):
        fit_temperature(_model(), _samples(), **options)
