"""Six-feature k-NN and tiny neural baselines; no imported agent-task weights.

Nano means 6->1 sigmoid (7 parameters). Micro means 6->8 tanh->1 sigmoid
(65 parameters). These are project-local size names, not general model standards.
"""
from __future__ import annotations

import hashlib
import json
import math
import random

from .features import FEATURE_NAMES, FEATURE_SCHEMA, validate_vector

BUNDLE_SCHEMA = "bellium.alpha-models/v1"
THRESHOLD = 0.80
MAX_DISTANCE = 0.20  # RMS distance to training support; heuristic, not calibrated OOD.


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, value))))


def _forward(model: dict, features: list[float]) -> tuple[list[float], float]:
    if not model["hidden"]:
        return [], _sigmoid(sum(w * x for w, x in zip(model["w2"], features)) + model["b2"])
    hidden = [math.tanh(sum(w * x for w, x in zip(row, features)) + b)
              for row, b in zip(model["w1"], model["b1"])]
    return hidden, _sigmoid(sum(w * x for w, x in zip(model["w2"], hidden)) + model["b2"])


def _train(rows: list[dict], hidden: int, seed: int, epochs: int) -> dict:
    rng = random.Random(seed)
    model = {
        "hidden": hidden,
        "w1": [[rng.uniform(-0.6, 0.6) for _ in FEATURE_NAMES] for _ in range(hidden)],
        "b1": [0.0] * hidden,
        "w2": [rng.uniform(-0.6, 0.6) for _ in range(hidden or len(FEATURE_NAMES))],
        "b2": 0.0,
    }
    order = list(rows)
    counts = {label: sum(row["label"] == label for row in rows) for label in ("candidate", "review")}
    if not all(counts.values()):
        raise ValueError("Training needs candidate and review examples")
    for _ in range(epochs):
        rng.shuffle(order)
        for row in order:
            features = row["features"]
            hidden_values, probability = _forward(model, features)
            weight = len(rows) / (2 * counts[row["label"]])
            error = (probability - (row["label"] == "review")) * weight
            old_output_weights = list(model["w2"])
            for i, value in enumerate(hidden_values if hidden else features):
                model["w2"][i] -= 0.04 * (error * value + 0.001 * model["w2"][i])
            model["b2"] -= 0.04 * error
            for i, value in enumerate(hidden_values):
                delta = error * old_output_weights[i] * (1 - value * value)
                for j, feature in enumerate(features):
                    model["w1"][i][j] -= 0.04 * (delta * feature + 0.001 * model["w1"][i][j])
                model["b1"][i] -= 0.04 * delta
    return model


def train_bundle(rows: list[dict], *, seed: int = 17, epochs: int = 100) -> dict:
    """Fit only explicitly marked train rows. Validation/test data never become neighbors."""
    if not rows or any(row["split"] != "train" for row in rows):
        raise ValueError("Only training rows may be supplied to fit")
    if not 1 <= epochs <= 1000 or len(rows) > 5000:
        raise ValueError("Training exceeds bounded epoch/example limits")
    for row in rows:
        validate_vector(row["features"])
        if row["label"] not in ("candidate", "review"):
            raise ValueError("Unknown label")
    support = [{"features": row["features"], "label": row["label"], "sha256": row["sha256"]}
               for row in rows]
    bundle = {
        "schema": BUNDLE_SCHEMA, "feature_schema": FEATURE_SCHEMA,
        "feature_names": list(FEATURE_NAMES), "authority": "shadow",
        "seed": seed, "epochs": epochs, "threshold": THRESHOLD,
        "max_distance": MAX_DISTANCE, "k": 5,
        "training_digest": digest(support), "support": support,
        "networks": {"nano": _train(rows, 0, seed, epochs),
                     "micro": _train(rows, 8, seed, epochs)},
    }
    bundle["bundle_sha256"] = digest(bundle)
    return bundle


def validate_bundle(bundle: dict) -> None:
    keys = {"schema", "feature_schema", "feature_names", "authority", "seed", "epochs",
            "threshold", "max_distance", "k", "training_digest", "support", "networks", "bundle_sha256"}
    if not isinstance(bundle, dict) or set(bundle) != keys:
        raise ValueError("Invalid or unknown model bundle fields")
    if (bundle["schema"] != BUNDLE_SCHEMA or bundle["feature_schema"] != FEATURE_SCHEMA
            or bundle["feature_names"] != list(FEATURE_NAMES) or bundle["authority"] != "shadow"):
        raise ValueError("Incompatible feature contract or authority")
    if bundle["bundle_sha256"] != digest({k: v for k, v in bundle.items() if k != "bundle_sha256"}):
        raise ValueError("Model bundle digest mismatch")
    if bundle["threshold"] != THRESHOLD or bundle["max_distance"] != MAX_DISTANCE or bundle["k"] != 5:
        raise ValueError("This version uses fixed, untuned abstention parameters")
    support = bundle["support"]
    if not isinstance(support, list) or not 2 <= len(support) <= 5000:
        raise ValueError("Invalid training support")
    for row in support:
        if not isinstance(row, dict) or set(row) != {"features", "label", "sha256"}:
            raise ValueError("Invalid support row")
        validate_vector(row["features"])
        if row["label"] not in ("candidate", "review"):
            raise ValueError("Invalid support label")
        if (not isinstance(row["sha256"], str) or len(row["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in row["sha256"])):
            raise ValueError("Invalid support identity")
    if digest(support) != bundle["training_digest"]:
        raise ValueError("Training support digest mismatch")
    if not isinstance(bundle["networks"], dict) or set(bundle["networks"]) != {"nano", "micro"}:
        raise ValueError("Expected nano and micro networks")
    for name, hidden in (("nano", 0), ("micro", 8)):
        model = bundle["networks"][name]
        if not isinstance(model, dict) or set(model) != {"hidden", "w1", "b1", "w2", "b2"}:
            raise ValueError("Invalid network fields")
        if model["hidden"] != hidden or not isinstance(model["w1"], list) or len(model["w1"]) != hidden:
            raise ValueError("Invalid architecture")
        arrays = [(model["b1"], hidden), (model["w2"], hidden or len(FEATURE_NAMES))]
        arrays += [(row, len(FEATURE_NAMES)) for row in model["w1"]]
        for array, length in arrays:
            if not isinstance(array, list) or len(array) != length:
                raise ValueError("Invalid parameter shape")
        values = [model["b2"]] + [value for array, _ in arrays for value in array]
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               or abs(v) > 1e6 for v in values):
            raise ValueError("Invalid network parameter")


class Predictors:
    def __init__(self, bundle: dict):
        validate_bundle(bundle)
        # Own the validated data; callers cannot mutate support/weights after validation.
        self.bundle = json.loads(canonical_bytes(bundle))

    def predict(self, name: str, features: list[float], *, sha256: str | None = None) -> dict:
        if name not in ("knn", "nano", "micro"):
            raise ValueError("Unknown predictor")
        features = validate_vector(features)
        neighbors = []
        seen = set()
        for row in self.bundle["support"]:
            identity = row["sha256"]
            if identity == sha256 or identity in seen:
                continue
            seen.add(identity)
            distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(features, row["features"])) / len(features))
            neighbors.append((distance, row))
        neighbors.sort(key=lambda item: (item[0], item[1]["sha256"]))
        near = [row for distance, row in neighbors[:5] if distance <= MAX_DISTANCE]
        if len(near) < 3:
            return {"label": "abstain", "review_score": None, "reason": "insufficient_nearby_training_support"}
        if name == "knn":
            probability = sum(row["label"] == "review" for row in near) / len(near)
        else:
            _, probability = _forward(self.bundle["networks"][name], features)
        label = "review" if probability >= THRESHOLD else "candidate" if 1 - probability >= THRESHOLD else "abstain"
        return {"label": label, "review_score": round(probability, 6),
                "reason": "uncalibrated_score" if label != "abstain" else "score_in_abstention_band"}
