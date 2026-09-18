"""Animation phase for one frame transition.

The seven inputs are deterministic deltas between consecutive cells. No nano
model is shipped for this decision: the measurement in
scripts/train_nano_frame_phase.py showed that the published threshold baseline is
not beaten on the generated motion set, so the rule answers and the result says
so. A model is used only when one is bundled and inside its budget.
"""

from __future__ import annotations

import math
from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.knn._sheet import deterministic_phase_verdict
from bellium.micro_nn.mlp import load_mlp, predict_mlp
from bellium.nano_nn.contract import NanoBudget, inspect_model
from bellium.resources import model_path

SPECIALIST_ID = "bellium/nano-nn/frame-phase:v0"
MODEL_PATH = model_path("nano_nn", "frame-phase", "v0.json")
BASELINE_ID = "bellium/deterministic/phase-threshold:v0"
FEATURE_NAMES = (
    "changed_ratio",
    "added_ratio",
    "removed_ratio",
    "centroid_shift",
    "area_delta",
    "change_slope",
    "position",
)
LABELS = ("hold", "build", "impact", "recover")
MIN_CONFIDENCE = 0.6
FRAME_PHASE_BUDGET = NanoBudget(
    max_parameters=64,
    max_model_bytes=8192,
    precision="float-json-weights-v1",
)


def phase_input(features: dict[str, Any]) -> list[float]:
    """Ordered model input. Missing or out-of-range measures fail closed."""
    if not isinstance(features, dict):
        raise ValueError("phase features must be an object")
    values = []
    for name in FEATURE_NAMES:
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


def classify_phase(
    features: dict[str, Any],
    *,
    model: dict[str, Any] | None = None,
) -> SpecialistResult:
    vector = phase_input(features)
    baseline = deterministic_phase_verdict(features)
    if model is None and not MODEL_PATH.exists():
        return SpecialistResult(
            SPECIALIST_ID,
            {
                "status": "baseline_only",
                "phase": baseline,
                "baseline": baseline,
                "baseline_id": BASELINE_ID,
                "model_shipped": False,
                "reason": "no_measured_benefit_over_baseline",
                "inputs": list(FEATURE_NAMES),
            },
            1.0,
            False,
            AuthorityMode.CONSULTATIVE,
            notes=(
                "No nano model is shipped for this task.",
                "The deterministic phase rule answered instead.",
            ),
        )
    loaded = model if model is not None else load_mlp(MODEL_PATH)
    contract = inspect_model(loaded, budget=FRAME_PHASE_BUDGET)
    probs = predict_mlp(loaded, vector)
    best = max(range(len(probs)), key=probs.__getitem__)
    confidence = float(probs[best])
    output = {
        "baseline": baseline,
        "baseline_id": BASELINE_ID,
        "probabilities": {LABELS[i]: round(float(probs[i]), 4) for i in range(len(LABELS))},
        "contract": contract,
        "neural_advantage_established": False,
    }
    if confidence < MIN_CONFIDENCE:
        output.update(status="abstain", reason="low_confidence", phase=None)
        return SpecialistResult(
            SPECIALIST_ID, output, round(confidence, 4), True, AuthorityMode.CONSULTATIVE,
            evidence_ref=MODEL_PATH.name,
            notes=("The nano model abstains instead of guessing an animation phase.",),
        )
    phase = LABELS[best]
    output.update(status="suggest", phase=phase, agrees_with_baseline=phase == baseline)
    return SpecialistResult(
        SPECIALIST_ID, output, round(confidence, 4), False, AuthorityMode.CONSULTATIVE,
        evidence_ref=MODEL_PATH.name,
        notes=("Motion-change phase only. It is not a semantic animation label.",),
    )
