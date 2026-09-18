"""Tiny feed-forward classifier inference. Stdlib only.

JSON layout matches the Botte Secrete micro-NN export:
layers, weights (flat row-major), biases, activations.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


def _relu(values: list[float]) -> list[float]:
    return [v if v > 0.0 else 0.0 for v in values]


def _sigmoid(values: list[float]) -> list[float]:
    out: list[float] = []
    for value in values:
        clipped = max(-500.0, min(500.0, value))
        out.append(1.0 / (1.0 + math.exp(-clipped)))
    return out


def _softmax(values: list[float]) -> list[float]:
    peak = max(values)
    exps = [math.exp(value - peak) for value in values]
    total = sum(exps) or 1.0
    return [value / total for value in exps]


def _dot_row(weights: list[float], vec: list[float], out_dim: int, in_dim: int) -> list[float]:
    if len(weights) != out_dim * in_dim:
        raise ValueError("weight matrix size does not match layer dimensions")
    result = [0.0] * out_dim
    for row in range(out_dim):
        offset = row * in_dim
        acc = 0.0
        for col in range(in_dim):
            acc += weights[offset + col] * vec[col]
        result[row] = acc
    return result


def load_mlp(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_mlp(data)
    return data


def validate_mlp(data: dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise ValueError("model file must be a JSON object")
    for key in ("layers", "weights", "biases", "activations"):
        if key not in data:
            raise ValueError(f"model file missing {key}")
    layers = data["layers"]
    if not isinstance(layers, list) or len(layers) < 2:
        raise ValueError("layers must list input and output sizes")
    if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in layers):
        raise ValueError("layer sizes must be positive integers")
    if len(data["weights"]) != len(layers) - 1:
        raise ValueError("weight blocks must match layer transitions")
    if len(data["biases"]) != len(layers) - 1:
        raise ValueError("bias blocks must match layer transitions")
    if len(data["activations"]) != len(layers) - 1:
        raise ValueError("activations must match layer transitions")
    for index in range(len(layers) - 1):
        for key, size in (("weights", layers[index] * layers[index + 1]), ("biases", layers[index + 1])):
            values = data[key][index]
            if not isinstance(values, list) or len(values) != size:
                raise ValueError(f"{key} size does not match layers")
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
                raise ValueError(f"{key} must contain finite numbers")
        if data["activations"][index] not in {"relu", "sigmoid", "softmax", "linear"}:
            raise ValueError("unsupported activation")


def predict_mlp(model: dict[str, Any], input_vec: list[float]) -> list[float]:
    validate_mlp(model)
    layers = model["layers"]
    if len(input_vec) != layers[0]:
        raise ValueError(f"expected {layers[0]} features, got {len(input_vec)}")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in input_vec):
        raise ValueError("input features must be finite numbers")
    hidden = [float(value) for value in input_vec]
    for index, weights in enumerate(model["weights"]):
        in_dim = layers[index]
        out_dim = layers[index + 1]
        biases = model["biases"][index]
        if len(biases) != out_dim:
            raise ValueError("bias length does not match output dimension")
        summed = _dot_row([float(w) for w in weights], hidden, out_dim, in_dim)
        preact = [summed[i] + float(biases[i]) for i in range(out_dim)]
        if not all(math.isfinite(value) for value in preact):
            raise ValueError("model arithmetic overflowed")
        act = model["activations"][index]
        if act == "relu":
            hidden = _relu(preact)
        elif act == "sigmoid":
            hidden = _sigmoid(preact)
        elif act == "softmax":
            hidden = _softmax(preact)
        elif act == "linear":
            hidden = preact
        else:
            raise ValueError(f"unsupported activation: {act}")
    return hidden


def save_mlp(path: str | Path, model: dict[str, Any]) -> None:
    validate_mlp(model)
    payload = json.dumps(model, indent=2, allow_nan=False)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
