from bellium.nano_nn.frame_phase import (
    LABELS,
    MODEL_PATH,
    classify_phase,
    phase_input,
)

IMPACT_FEATURES = {
    "changed_ratio": 0.40,
    "added_ratio": 0.20,
    "removed_ratio": 0.20,
    "centroid_shift": 0.05,
    "area_delta": 0.10,
    "change_slope": 0.50,
    "position": 0.50,
}
HOLD_FEATURES = {
    "changed_ratio": 0.01,
    "added_ratio": 0.005,
    "removed_ratio": 0.005,
    "centroid_shift": 0.005,
    "area_delta": 0.005,
    "change_slope": 0.50,
    "position": 0.25,
}


def _hold_model() -> dict:
    return {
        "layers": [7, 4, 4],
        "weights": [[0.0] * 28, [0.0] * 16],
        "biases": [[0.0] * 4, [3.0, 0.0, 0.0, 0.0]],
        "activations": ["relu", "softmax"],
    }


def test_no_model_is_shipped_and_the_rule_answers() -> None:
    assert MODEL_PATH.exists() is False
    result = classify_phase(IMPACT_FEATURES)
    assert result.abstained is False
    assert result.output["status"] == "baseline_only"
    assert result.output["model_shipped"] is False
    assert result.output["phase"] == result.output["baseline"] == "impact"


def test_a_bundled_model_is_used_and_budgeted() -> None:
    result = classify_phase(HOLD_FEATURES, model=_hold_model())
    assert result.output["status"] == "suggest"
    assert result.output["phase"] in LABELS
    assert result.output["phase"] == "hold"
    assert result.output["agrees_with_baseline"] is True
    assert result.output["contract"]["within_budget"] is True


def test_model_over_budget_is_refused() -> None:
    big = {
        "layers": [7, 40, 4],
        "weights": [[0.0] * 280, [0.0] * 160],
        "biases": [[0.0] * 40, [0.0] * 4],
        "activations": ["relu", "softmax"],
    }
    blocked = False
    try:
        classify_phase(IMPACT_FEATURES, model=big)
    except ValueError:
        blocked = True
    assert blocked


def test_missing_measure_is_not_coerced() -> None:
    partial = dict(IMPACT_FEATURES)
    partial.pop("area_delta")
    blocked = False
    try:
        phase_input(partial)
    except ValueError:
        blocked = True
    assert blocked


def test_out_of_range_measure_is_refused() -> None:
    blocked = False
    try:
        phase_input(dict(IMPACT_FEATURES, changed_ratio=1.5))
    except ValueError:
        blocked = True
    assert blocked
