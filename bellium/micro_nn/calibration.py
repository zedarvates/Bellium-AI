"""Explicit temperature calibration for native softmax classifiers. Stdlib only.

Calibration changes probability scores, never weights or class order. Fit on a
separate labeled calibration split; evaluate on fresh data before deployment.
No sidecar is loaded, saved or applied automatically.
"""

from __future__ import annotations

import hashlib
import json
import math

from bellium.micro_nn.mlp import predict_mlp, validate_mlp

SCHEMA = "bellium.temperature-calibration/v1"


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _temperature(value):
    value = _number(value, "temperature")
    if value <= 0.0:
        raise ValueError("temperature must be positive")
    return value


def model_fingerprint(model: dict) -> str:
    validate_mlp(model)
    if model["activations"][-1] != "softmax" or model["layers"][-1] < 2:
        raise ValueError("calibration requires a multi-class softmax model")
    payload = json.dumps(model, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def predict_logits(model: dict, vector: list[float]) -> list[float]:
    validate_mlp(model)
    if model["activations"][-1] != "softmax" or model["layers"][-1] < 2:
        raise ValueError("calibration requires a multi-class softmax model")
    # Copy only the activation list; the inference engine never mutates weights.
    linear_output = {**model, "activations": [*model["activations"][:-1], "linear"]}
    return predict_mlp(linear_output, vector)


def _rows(logits, labels):
    if not isinstance(logits, list) or not logits or not isinstance(labels, list):
        raise ValueError("need non-empty logits and labels lists")
    if len(logits) != len(labels):
        raise ValueError("logits and labels must have identical lengths")
    rows = []
    classes = None
    for values, label in zip(logits, labels):
        if not isinstance(values, (list, tuple)) or len(values) < 2:
            raise ValueError("each logit vector must contain at least two classes")
        if classes is None:
            classes = len(values)
        if len(values) != classes:
            raise ValueError("logit vectors must have the same class count")
        if isinstance(label, bool) or not isinstance(label, int) or not 0 <= label < classes:
            raise ValueError("label must be an integer in the class range")
        rows.append([_number(value, "logit") for value in values])
    return rows, classes


def _scaled(row, temperature):
    scaled = [_number(value / temperature, "scaled logit") for value in row]
    peak = max(scaled)
    differences = [_number(value - peak, "logit difference") for value in scaled]
    exps = [math.exp(value) for value in differences]
    total = sum(exps)
    return [value / total for value in exps], differences, math.log(total)


def _nll(rows, labels, temperature):
    losses = []
    for row, label in zip(rows, labels):
        _, differences, log_total = _scaled(row, temperature)
        losses.append(log_total - differences[label])
    return _number(sum(value / len(rows) for value in losses), "mean loss")


def probability_metrics(logits: list, labels: list, *, temperature: float = 1.0,
                        bins: int = 10, confidence_threshold: float = 0.55) -> dict:
    rows, classes = _rows(logits, labels)
    temperature = _temperature(temperature)
    if isinstance(bins, bool) or not isinstance(bins, int) or bins <= 0:
        raise ValueError("bins must be a positive integer")
    threshold = _number(confidence_threshold, "confidence_threshold")
    if not 0 <= threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1")
    buckets = [[] for _ in range(bins)]
    support = [0] * classes
    class_losses = [0.0] * classes
    class_briers = [0.0] * classes
    hits = accepted = accepted_hits = 0
    brier = 0.0
    for row, label in zip(rows, labels):
        probabilities, differences, log_total = _scaled(row, temperature)
        prediction = max(range(classes), key=row.__getitem__)
        confidence = probabilities[prediction]
        correct = int(prediction == label)
        support[label] += 1
        hits += correct
        if confidence >= threshold:
            accepted += 1
            accepted_hits += correct
        buckets[min(bins - 1, int(confidence * bins))].append((confidence, correct))
        row_brier = sum((p - int(index == label)) ** 2 for index, p in enumerate(probabilities))
        brier += row_brier / len(rows)
        class_losses[label] += log_total - differences[label]
        class_briers[label] += row_brier
    reliability = []
    ece = 0.0
    for index, bucket in enumerate(buckets):
        confidence = sum(value for value, _ in bucket) / len(bucket) if bucket else None
        accuracy = sum(hit for _, hit in bucket) / len(bucket) if bucket else None
        if bucket:
            ece += len(bucket) / len(rows) * abs(accuracy - confidence)
        reliability.append({"lower": index / bins, "upper": (index + 1) / bins,
                            "samples": len(bucket), "mean_confidence": confidence,
                            "accuracy": accuracy})
    return {"samples": len(rows), "class_support": support,
            "class_nll": [loss / count if count else None for loss, count in zip(class_losses, support)],
            "class_brier": [loss / count if count else None for loss, count in zip(class_briers, support)],
            "nll": _nll(rows, labels, temperature), "brier": brier, "ece": ece,
            "accuracy": hits / len(rows), "accepted": accepted,
            "abstained": len(rows) - accepted, "coverage": accepted / len(rows),
            "selective_accuracy": accepted_hits / accepted if accepted else None,
            "temperature": temperature, "confidence_threshold": threshold,
            "reliability_bins": reliability}


def fit_temperature(model: dict, samples: list, *, min_temperature: float = 0.1,
                    max_temperature: float = 10.0, iterations: int = 60) -> dict:
    fingerprint = model_fingerprint(model)
    lower = _temperature(min_temperature)
    upper = _temperature(max_temperature)
    if not lower <= 1.0 <= upper:
        raise ValueError("temperature search bounds must contain identity 1.0")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    if not isinstance(samples, list) or not samples:
        raise ValueError("need calibration samples")
    logits, labels = [], []
    for sample in samples:
        if not isinstance(sample, (list, tuple)) or len(sample) != 2:
            raise ValueError("sample must contain vector and label")
        vector, label = sample
        logits.append(predict_logits(model, vector))
        labels.append(label)
    rows, classes = _rows(logits, labels)
    support = [labels.count(index) for index in range(classes)]
    if not all(support):
        raise ValueError("calibration requires support for every model class")
    # NLL is convex in inverse temperature; a monotone log transform keeps a
    # single minimum. Boundaries and identity are always considered explicitly.
    left, right = math.log(lower), math.log(upper)
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = right - ratio * (right - left), left + ratio * (right - left)
    loss_a, loss_b = _nll(rows, labels, math.exp(a)), _nll(rows, labels, math.exp(b))
    for _ in range(iterations):
        if loss_a <= loss_b:
            right, b, loss_b = b, a, loss_a
            a = right - ratio * (right - left)
            loss_a = _nll(rows, labels, math.exp(a))
        else:
            left, a, loss_a = a, b, loss_b
            b = left + ratio * (right - left)
            loss_b = _nll(rows, labels, math.exp(b))
    candidates = (1.0, lower, upper, math.exp((left + right) / 2.0))
    best = min(candidates, key=lambda t: (_nll(rows, labels, t), abs(math.log(t))))
    return {"schema": SCHEMA, "model_fingerprint": fingerprint,
            "classes": classes, "samples": len(samples), "class_support": support,
            "temperature": best, "calibration_nll_before": _nll(rows, labels, 1.0),
            "calibration_nll_after": _nll(rows, labels, best),
            "search_bounds": [lower, upper],
            "at_search_bound": math.isclose(best, lower) or math.isclose(best, upper)}


def predict_calibrated(model: dict, vector: list[float], calibration: dict) -> list[float]:
    if not isinstance(calibration, dict) or calibration.get("schema") != SCHEMA:
        raise ValueError("temperature calibration schema mismatch")
    if calibration.get("model_fingerprint") != model_fingerprint(model):
        raise ValueError("calibration does not match model")
    if calibration.get("classes") != model["layers"][-1]:
        raise ValueError("calibration class count does not match model")
    temperature = _temperature(calibration.get("temperature"))
    logits = predict_logits(model, vector)
    probabilities = _scaled(logits, temperature)[0]
    if max(range(len(logits)), key=logits.__getitem__) != max(range(len(probabilities)), key=probabilities.__getitem__):
        raise ValueError("temperature loses class ordering at floating-point precision")
    return probabilities
