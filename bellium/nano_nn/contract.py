"""Size contract for the nano-NN tier.

Nano-NN and micro-NN are project size labels, not standard architectures and
never language models. Every nano specialist declares its parameter count,
model bytes, numeric precision and device targets, and a model that exceeds its
declared budget is refused instead of being loaded.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class NanoBudget:
    max_parameters: int
    max_model_bytes: int
    precision: str
    target_latency_ms: float | None = None
    target_peak_ram_bytes: int | None = None
    device: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.max_parameters, bool) or not isinstance(self.max_parameters, int):
            raise ValueError("max_parameters must be an integer")
        if isinstance(self.max_model_bytes, bool) or not isinstance(self.max_model_bytes, int):
            raise ValueError("max_model_bytes must be an integer")
        if self.max_parameters <= 0 or self.max_model_bytes <= 0:
            raise ValueError("nano budgets must be positive")
        if not isinstance(self.precision, str) or not self.precision.strip():
            raise ValueError("precision must be declared")
        for name, value in (
            ("target_latency_ms", self.target_latency_ms),
            ("target_peak_ram_bytes", self.target_peak_ram_bytes),
        ):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric or null")
            if not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{name} must be positive and finite")

    def describe(self) -> dict[str, Any]:
        described = asdict(self)
        described["measured_latency_ms"] = None
        described["measured_peak_ram_bytes"] = None
        described["device_verified"] = False
        return described


def count_parameters(model: dict[str, Any]) -> int:
    """Weights and biases of every layer, counted from the declared shapes."""
    if not isinstance(model, dict):
        raise ValueError("model must be an object")
    weights = model.get("weights")
    biases = model.get("biases")
    if not isinstance(weights, list) or not isinstance(biases, list):
        raise ValueError("model needs weights and biases")
    if len(weights) != len(biases):
        raise ValueError("weight and bias blocks must match")
    total = 0
    for block, bias in zip(weights, biases):
        if not isinstance(block, list) or not isinstance(bias, list):
            raise ValueError("blocks must be lists")
        total += len(block) + len(bias)
    return total


def serialized_bytes(model: dict[str, Any]) -> int:
    """Bytes of the indented JSON that save_mlp writes, not a compact rewrite."""
    return len(json.dumps(model, indent=2, allow_nan=False).encode("utf-8"))


def inspect_model(model: dict[str, Any], *, budget: NanoBudget) -> dict[str, Any]:
    """Measured size of a nano model, refused when the declared budget is exceeded."""
    parameters = count_parameters(model)
    model_bytes = serialized_bytes(model)
    if parameters > budget.max_parameters:
        raise ValueError(
            f"nano budget exceeded: {parameters} parameters > {budget.max_parameters}"
        )
    if model_bytes > budget.max_model_bytes:
        raise ValueError(f"nano budget exceeded: {model_bytes} bytes > {budget.max_model_bytes}")
    return {
        "parameters": parameters,
        "model_bytes": model_bytes,
        "bytes_basis": "indented JSON, LF newlines",
        "precision": budget.precision,
        "budget": budget.describe(),
        "within_budget": True,
    }
