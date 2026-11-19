"""Classification evidence with explicit support and abstention denominators."""

from __future__ import annotations

import math

from bellium.micro_nn.mlp import predict_mlp, validate_mlp


def classification_metrics(model: dict, samples: list, *, confidence_threshold: float = 0.55) -> dict:
    """Evaluate a final test split, independently of validation-based selection.

    Accuracy and recall include abstaining predictions. Selective accuracy counts
    only accepted predictions. Neither softmax scores nor these metrics certify
    calibrated probabilities or real-world task quality.
    """
    validate_mlp(model)
    if not samples:
        raise ValueError("need evaluation samples")
    if (isinstance(confidence_threshold, bool)
            or not isinstance(confidence_threshold, (int, float))
            or not math.isfinite(confidence_threshold)
            or not 0.0 <= confidence_threshold <= 1.0):
        raise ValueError("confidence_threshold must be between 0 and 1")
    if model["activations"][-1] != "softmax":
        raise ValueError("classification metrics require softmax output")
    classes = model["layers"][-1]
    matrix = [[0] * classes for _ in range(classes)]
    accepted = accepted_hits = 0
    abstentions = [0] * classes
    for vector, target in samples:
        if (isinstance(target, bool) or not isinstance(target, int)
                or not 0 <= target < classes):
            raise ValueError("target must be an integer in the class range")
        probabilities = predict_mlp(model, vector)
        predicted = max(range(classes), key=probabilities.__getitem__)
        matrix[target][predicted] += 1
        if probabilities[predicted] >= confidence_threshold:
            accepted += 1
            accepted_hits += predicted == target
        else:
            abstentions[target] += 1
    supports = [sum(row) for row in matrix]
    recalls = [matrix[index][index] / count if count else None
               for index, count in enumerate(supports)]
    defined_recalls = [value for value in recalls if value is not None]
    return {
        "samples": len(samples),
        "confusion_true_rows": matrix,
        "class_support": supports,
        "class_recall": recalls,
        "accuracy": sum(matrix[i][i] for i in range(classes)) / len(samples),
        "balanced_accuracy": sum(defined_recalls) / len(defined_recalls),
        "accepted": accepted,
        "abstained": len(samples) - accepted,
        "class_abstentions": abstentions,
        "coverage": accepted / len(samples),
        "selective_accuracy": accepted_hits / accepted if accepted else None,
        "confidence_threshold": confidence_threshold,
    }


def classification_gate(metrics: dict, *, min_accuracy: float = 0.95,
                        min_recall: float = 0.90, min_support: int = 10) -> dict:
    """Reject global-score passes that conceal an unsupported or failing class."""
    for name, value in (("min_accuracy", min_accuracy), ("min_recall", min_recall)):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError(f"{name} must be between 0 and 1")
    if isinstance(min_support, bool) or not isinstance(min_support, int) or min_support <= 0:
        raise ValueError("min_support must be a positive integer")
    supports = metrics["class_support"]
    recalls = metrics["class_recall"]
    if (not isinstance(supports, list) or not supports
            or not isinstance(recalls, list) or len(recalls) != len(supports)):
        raise ValueError("gate needs matching non-empty support and recall lists")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
           for value in supports):
        raise ValueError("class support must contain non-negative integers")
    if sum(supports) != metrics["samples"]:
        raise ValueError("class support must account for every sample")
    reasons = []
    accuracy = metrics["accuracy"]
    if not math.isfinite(accuracy) or accuracy < min_accuracy:
        reasons.append("accuracy_below_threshold")
    for index, (support, recall) in enumerate(zip(supports, recalls)):
        if support < min_support:
            reasons.append(f"class_{index}_insufficient_support")
        if recall is None or not math.isfinite(recall) or recall < min_recall:
            reasons.append(f"class_{index}_recall_below_threshold")
    return {"passed": not reasons, "reasons": reasons,
            "thresholds": {"min_accuracy": min_accuracy, "min_recall": min_recall,
                           "min_support": min_support}}
