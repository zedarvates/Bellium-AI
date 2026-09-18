"""Tiny SGD trainer for Bellium-native micro-NNs. Stdlib only."""

from __future__ import annotations

import math
import random
from typing import Callable


def _zeros(rows: int, cols: int) -> list[float]:
    return [0.0] * (rows * cols)


def _softmax(values: list[float]) -> list[float]:
    peak = max(values)
    exps = [math.exp(value - peak) for value in values]
    total = sum(exps) or 1.0
    return [value / total for value in exps]


def _relu(values: list[float]) -> list[float]:
    return [v if v > 0.0 else 0.0 for v in values]


def init_mlp(layer_sizes: list[int], rng: random.Random) -> dict:
    weights: list[list[float]] = []
    biases: list[list[float]] = []
    activations: list[str] = []
    for index in range(len(layer_sizes) - 1):
        in_dim = layer_sizes[index]
        out_dim = layer_sizes[index + 1]
        scale = math.sqrt(2.0 / in_dim)
        block = [rng.gauss(0.0, scale) for _ in range(out_dim * in_dim)]
        weights.append(block)
        biases.append([0.0] * out_dim)
        activations.append("relu" if index < len(layer_sizes) - 2 else "softmax")
    return {
        "layers": list(layer_sizes),
        "weights": weights,
        "biases": biases,
        "activations": activations,
    }


def _forward(model: dict, vec: list[float]) -> tuple[list[list[float]], list[list[float]]]:
    hidden = [float(v) for v in vec]
    preacts: list[list[float]] = []
    acts: list[list[float]] = [hidden]
    layers = model["layers"]
    for index, weights in enumerate(model["weights"]):
        in_dim = layers[index]
        out_dim = layers[index + 1]
        z = [0.0] * out_dim
        for row in range(out_dim):
            acc = float(model["biases"][index][row])
            offset = row * in_dim
            for col in range(in_dim):
                acc += weights[offset + col] * hidden[col]
            z[row] = acc
        preacts.append(z)
        if model["activations"][index] == "relu":
            hidden = _relu(z)
        else:
            hidden = _softmax(z)
        acts.append(hidden)
    return preacts, acts


def train_classifier(
    samples: list[tuple[list[float], int]],
    *,
    layer_sizes: list[int],
    epochs: int = 400,
    lr: float = 0.08,
    seed: int = 7,
) -> dict:
    if not samples:
        raise ValueError("need training samples")
    rng = random.Random(seed)
    model = init_mlp(layer_sizes, rng)
    n_classes = layer_sizes[-1]
    for _ in range(epochs):
        rng.shuffle(samples)
        for vec, label in samples:
            preacts, acts = _forward(model, vec)
            probs = acts[-1]
            delta = [probs[i] - (1.0 if i == label else 0.0) for i in range(n_classes)]
            # backprop last layer
            for layer_index in range(len(model["weights"]) - 1, -1, -1):
                in_act = acts[layer_index]
                in_dim = model["layers"][layer_index]
                out_dim = model["layers"][layer_index + 1]
                weights = model["weights"][layer_index]
                biases = model["biases"][layer_index]
                next_delta = [0.0] * in_dim
                for row in range(out_dim):
                    grad = delta[row]
                    biases[row] -= lr * grad
                    offset = row * in_dim
                    for col in range(in_dim):
                        next_delta[col] += grad * weights[offset + col]
                        weights[offset + col] -= lr * grad * in_act[col]
                if layer_index == 0:
                    break
                relu_pre = preacts[layer_index - 1]
                delta = [next_delta[i] if relu_pre[i] > 0.0 else 0.0 for i in range(in_dim)]
    model["labels"] = list(range(n_classes))
    return model


def accuracy(model: dict, samples: list[tuple[list[float], int]], predict: Callable) -> float:
    if not samples:
        return 0.0
    hits = 0
    for vec, label in samples:
        probs = predict(model, vec)
        pred = max(range(len(probs)), key=probs.__getitem__)
        if pred == label:
            hits += 1
    return hits / len(samples)


def init_regressor(layer_sizes: list[int], rng: random.Random) -> dict:
    """Same tiny topology as the classifier, with a linear output for a scalar target."""
    model = init_mlp(layer_sizes, rng)
    model["activations"][-1] = "linear"
    return model


def _forward_regressor(model: dict, vec: list[float]) -> tuple[list[list[float]], list[list[float]]]:
    hidden = [float(v) for v in vec]
    preacts: list[list[float]] = []
    acts: list[list[float]] = [hidden]
    layers = model["layers"]
    for index, weights in enumerate(model["weights"]):
        in_dim = layers[index]
        out_dim = layers[index + 1]
        z = [0.0] * out_dim
        for row in range(out_dim):
            acc = float(model["biases"][index][row])
            offset = row * in_dim
            for col in range(in_dim):
                acc += weights[offset + col] * hidden[col]
            z[row] = acc
        preacts.append(z)
        hidden = _relu(z) if model["activations"][index] == "relu" else z
        acts.append(hidden)
    return preacts, acts


def train_regressor(
    samples: list[tuple[list[float], float]],
    *,
    layer_sizes: list[int],
    epochs: int = 400,
    lr: float = 0.05,
    seed: int = 7,
    target_scale: float = 1.0,
    momentum: float = 0.9,
) -> dict:
    """Plain SGD on mean squared error for a single scalar output.

    The target is divided by target_scale so the linear output starts in a
    reasonable range, and the caller keeps the scale to undo it at inference.
    """
    if not samples:
        raise ValueError("need training samples")
    if isinstance(target_scale, bool) or not isinstance(target_scale, (int, float)):
        raise ValueError("target_scale must be numeric")
    if not math.isfinite(float(target_scale)) or float(target_scale) == 0.0:
        raise ValueError("target_scale must be finite and non-zero")
    if isinstance(momentum, bool) or not isinstance(momentum, (int, float)):
        raise ValueError("momentum must be numeric")
    if not 0.0 <= float(momentum) < 1.0:
        raise ValueError("momentum must be between 0 and 1")
    rng = random.Random(seed)
    model = init_regressor(layer_sizes, rng)
    velocity_weights = [[0.0] * len(block) for block in model["weights"]]
    velocity_biases = [[0.0] * len(bias) for bias in model["biases"]]
    scaled = [(vec, float(target) / float(target_scale)) for vec, target in samples]
    for _ in range(epochs):
        rng.shuffle(scaled)
        for vec, target in scaled:
            preacts, acts = _forward_regressor(model, vec)
            delta = [acts[-1][0] - target]
            for layer_index in range(len(model["weights"]) - 1, -1, -1):
                in_act = acts[layer_index]
                in_dim = model["layers"][layer_index]
                out_dim = model["layers"][layer_index + 1]
                weights = model["weights"][layer_index]
                biases = model["biases"][layer_index]
                v_weights = velocity_weights[layer_index]
                v_biases = velocity_biases[layer_index]
                next_delta = [0.0] * in_dim
                for row in range(out_dim):
                    grad = delta[row]
                    v_biases[row] = momentum * v_biases[row] - lr * grad
                    biases[row] += v_biases[row]
                    offset = row * in_dim
                    for col in range(in_dim):
                        next_delta[col] += grad * weights[offset + col]
                        v_weights[offset + col] = (
                            momentum * v_weights[offset + col] - lr * grad * in_act[col]
                        )
                        weights[offset + col] += v_weights[offset + col]
                if layer_index == 0:
                    break
                relu_pre = preacts[layer_index - 1]
                delta = [next_delta[i] if relu_pre[i] > 0.0 else 0.0 for i in range(in_dim)]
    model["target_scale"] = float(target_scale)
    return model


def regression_metrics(
    model: dict,
    samples: list[tuple[list[float], float]],
    predict: Callable,
) -> dict[str, float]:
    """Root mean squared error and worst case on a sample set, in target units."""
    if not samples:
        raise ValueError("need samples")
    scale = float(model.get("target_scale", 1.0))
    errors = []
    for vec, target in samples:
        output = predict(model, vec)
        value = output[0] * scale if isinstance(output, list) else float(output) * scale
        errors.append(value - float(target))
    count = len(errors)
    rmse = math.sqrt(sum(error * error for error in errors) / count)
    return {
        "rmse": rmse,
        "max_abs": max(abs(error) for error in errors),
        "bias": sum(errors) / count,
    }
