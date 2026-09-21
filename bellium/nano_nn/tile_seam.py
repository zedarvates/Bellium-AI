"""Nano-NN seam verdict for one texture axis.

The six inputs are deterministic measures. The model only replaces the decision
rule, and the deterministic threshold baseline stays published beside it so the
nano tier still has to earn its place.
"""

from __future__ import annotations

import math
from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.knn._texture import SEAM_CHANNELS, deterministic_seam_verdict
from bellium.micro_nn.mlp import load_mlp, predict_mlp
from bellium.nano_nn.contract import NanoBudget, inspect_model
from bellium.resources import model_path

SPECIALIST_ID = "bellium/nano-nn/tile-seam:v0"
MODEL_PATH = model_path("nano_nn", "tile-seam", "v0.json")
BASELINE_ID = "bellium/deterministic/seam-threshold:v0"
LABELS = ("continuous", "mismatch")
MIN_CONFIDENCE = 0.6
TILE_SEAM_BUDGET = NanoBudget(
    max_parameters=64,
    max_model_bytes=8192,
    precision="float-json-weights-v1",
)


def seam_input(features: dict[str, Any]) -> list[float]:
    """Ordered model input. Missing or out-of-range measures fail closed."""
    if not isinstance(features, dict):
        raise ValueError("seam features must be an object")
    values = []
    for name in SEAM_CHANNELS:
        raw = features.get(name)
        if raw is None:
            raise ValueError(f"feature {name} is unknown; do not coerce to 0")
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"feature {name} must be numeric")
        number = float(raw)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"feature {name} must be between 0 and 1")
        values.append(number)
    return values


def classify_seam(
    features: dict[str, Any],
    *,
    model: dict[str, Any] | None = None,
) -> SpecialistResult:
    vector = seam_input(features)
    loaded = model if model is not None else load_mlp(MODEL_PATH)
    contract = inspect_model(loaded, budget=TILE_SEAM_BUDGET)
    probs = predict_mlp(loaded, vector)
    best = max(range(len(probs)), key=probs.__getitem__)
    confidence = float(probs[best])
    baseline = deterministic_seam_verdict(features)
    output = {
        "baseline": baseline,
        "baseline_id": BASELINE_ID,
        "probabilities": {LABELS[i]: round(float(probs[i]), 4) for i in range(len(LABELS))},
        "contract": contract,
        "neural_advantage_established": False,
    }
    if confidence < MIN_CONFIDENCE:
        output["status"] = "abstain"
        output["reason"] = "low_confidence"
        output["verdict"] = None
        return SpecialistResult(
            SPECIALIST_ID, output, round(confidence, 4), True, AuthorityMode.CONSULTATIVE,
            evidence_ref=MODEL_PATH.name,
            notes=("The nano model abstains instead of guessing a seam.",),
        )
    verdict = LABELS[best]
    output["status"] = "suggest"
    output["verdict"] = verdict
    output["agrees_with_baseline"] = verdict == baseline
    return SpecialistResult(
        SPECIALIST_ID,
        output,
        round(confidence, 4),
        False,
        AuthorityMode.CONSULTATIVE,
        evidence_ref=MODEL_PATH.name,
        notes=("Tiny seam decision only. It neither repairs nor crops the texture.",),
    )
