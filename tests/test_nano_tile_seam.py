import json
import math

from bellium.knn._texture import seam_features
from bellium.nano_nn.contract import NanoBudget, count_parameters, inspect_model
from bellium.nano_nn.tile_seam import MODEL_PATH, TILE_SEAM_BUDGET, classify_seam, seam_input


def _gray(value):
    level = max(0, min(255, int(value)))
    return (level, level, level)


def _sine(size=64, period=8, amplitude=90.0):
    return [
        [_gray(128 + amplitude * math.sin(2 * math.pi * c / period)) for c in range(size)]
        for _ in range(size)
    ]


def _wrapped_step(size=64, step=140):
    return [
        [_gray(90 + (step / 2 if c < size // 2 else -step / 2)) for c in range(size)]
        for _ in range(size)
    ]


def test_model_stays_inside_its_declared_budget() -> None:
    model = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    contract = inspect_model(model, budget=TILE_SEAM_BUDGET)
    assert contract["within_budget"] is True
    assert contract["parameters"] <= 64
    assert contract["model_bytes"] <= 8192
    assert contract["budget"]["device_verified"] is False
    assert model["heldout_accuracy"] >= model["heldout_baseline_accuracy"]


def test_oversized_model_is_refused() -> None:
    small = NanoBudget(max_parameters=4, max_model_bytes=4096, precision="float-json-weights-v1")
    model = {"layers": [6, 4, 2], "weights": [[0.0] * 24, [0.0] * 8], "biases": [[0.0] * 4, [0.0] * 2],
             "activations": ["relu", "softmax"]}
    assert count_parameters(model) == 38
    blocked = False
    try:
        inspect_model(model, budget=small)
    except ValueError:
        blocked = True
    assert blocked


def test_continuous_and_broken_wraps_are_separated() -> None:
    continuous = classify_seam(seam_features(_sine())["x"])
    broken = classify_seam(seam_features(_wrapped_step())["x"])
    assert continuous.output["verdict"] in {"continuous", None}
    assert broken.output["verdict"] in {"mismatch", None}
    if not broken.abstained:
        assert broken.output["agrees_with_baseline"] is True


def test_abstention_is_available() -> None:
    neutral = {"gap": 0.2, "seam": 0.3, "interior": 0.3, "edge_step": 0.25,
               "seam_max": 0.4, "hot_rows": 0.3}
    result = classify_seam(neutral)
    assert isinstance(result.abstained, bool)
    if result.abstained:
        assert result.output["verdict"] is None


def test_missing_measure_is_not_coerced() -> None:
    blocked = False
    try:
        seam_input({"gap": 0.1, "seam": 0.1, "interior": 0.1, "edge_step": 0.1, "seam_max": 0.1})
    except ValueError:
        blocked = True
    assert blocked


def test_out_of_range_measure_is_refused() -> None:
    blocked = False
    try:
        seam_input({"gap": 1.4, "seam": 0.1, "interior": 0.1, "edge_step": 0.1,
                    "seam_max": 0.1, "hot_rows": 0.1})
    except ValueError:
        blocked = True
    assert blocked
