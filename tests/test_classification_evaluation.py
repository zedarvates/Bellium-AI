import math

import pytest

from bellium.micro_nn.evaluate import classification_gate, classification_metrics


def _model():
    return {"layers": [1, 2], "weights": [[-1.0, 1.0]],
            "biases": [[0.0, 0.0]], "activations": ["softmax"]}


def test_confusion_and_abstention_denominators():
    samples = [([-2.0], 0), ([2.0], 1), ([2.0], 0), ([0.0], 1)]
    result = classification_metrics(_model(), samples)
    assert result["confusion_true_rows"] == [[1, 1], [1, 1]]
    assert result["class_support"] == [2, 2]
    assert result["class_recall"] == [0.5, 0.5]
    assert result["class_abstentions"] == [0, 1]
    assert result["accuracy"] == 0.5
    assert result["coverage"] == 0.75
    assert result["selective_accuracy"] == 2 / 3


def test_no_accepted_prediction_has_no_selective_accuracy():
    result = classification_metrics(_model(), [([0.0], 0)])
    assert result["selective_accuracy"] is None
    assert result["class_recall"][1] is None
    assert result["class_support"][1] == 0
    assert not classification_gate(result)["passed"]


def test_majority_accuracy_cannot_hide_minority_failure():
    samples = [([-2.0], 0)] * 100 + [([-2.0], 1)] * 10
    metrics = classification_metrics(_model(), samples)
    gate = classification_gate(metrics, min_accuracy=0.85)
    assert metrics["accuracy"] > 0.85
    assert gate["passed"] is False
    assert gate["reasons"] == ["class_1_recall_below_threshold"]


def test_supported_good_classes_pass():
    metrics = classification_metrics(_model(), [([-2.0], 0)] * 10 + [([2.0], 1)] * 10)
    assert classification_gate(metrics)["passed"]


@pytest.mark.parametrize("value", [True, -0.1, 1.1, math.nan, math.inf])
def test_invalid_confidence_threshold(value):
    with pytest.raises(ValueError):
        classification_metrics(_model(), [([0.0], 0)], confidence_threshold=value)


@pytest.mark.parametrize("target", [True, -1, 2, 0.5])
def test_invalid_target(target):
    with pytest.raises(ValueError):
        classification_metrics(_model(), [([0.0], target)])


@pytest.mark.parametrize("options", [{"min_accuracy": math.nan}, {"min_recall": 1.1},
                                      {"min_support": True}, {"min_support": 0}])
def test_invalid_gate_thresholds(options):
    with pytest.raises(ValueError):
        classification_gate({}, **options)


def test_regression_output_is_not_interpreted_as_probabilities():
    model = _model()
    model["activations"] = ["linear"]
    with pytest.raises(ValueError, match="softmax"):
        classification_metrics(model, [([0.0], 0)])


@pytest.mark.parametrize("change", [
    {"class_support": [], "class_recall": []}, {"class_recall": [1.0]},
    {"class_support": [True, 10]}, {"samples": 100},
])
def test_incomplete_evidence_cannot_pass_gate(change):
    metrics = classification_metrics(_model(), [([-2.0], 0)] * 10 + [([2.0], 1)] * 10)
    metrics.update(change)
    with pytest.raises(ValueError):
        classification_gate(metrics)
