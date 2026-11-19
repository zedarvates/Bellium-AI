import copy
import math

import pytest

from scripts.benchmark_native_calibration import promotion_gate


def _metrics():
    return {"samples": 200, "class_support": [100, 100], "class_nll": [0.1, 0.1],
            "class_brier": [0.05, 0.05], "nll": 0.1, "brier": 0.05, "ece": 0.01,
            "accuracy": 0.95, "accepted": 200, "abstained": 0, "coverage": 1.0,
            "selective_accuracy": 0.95}


def _corrected():
    result = _metrics()
    result.update({"nll": 0.08, "class_nll": [0.08, 0.08], "brier": 0.04,
                   "class_brier": [0.04, 0.04], "ece": 0.005})
    return result


def test_valid_improvement_passes_without_mutating_evidence():
    raw, corrected = _metrics(), _corrected()
    before = copy.deepcopy((raw, corrected))
    assert promotion_gate(raw, corrected, {"at_search_bound": False})["passed"]
    assert (raw, corrected) == before


@pytest.mark.parametrize("name", ["nll", "brier", "ece", "coverage", "selective_accuracy"])
@pytest.mark.parametrize("value", [math.nan, math.inf, True])
def test_invalid_numeric_evidence_never_passes(name, value):
    raw, corrected = _metrics(), _corrected()
    corrected[name] = value
    result = promotion_gate(raw, corrected, {"at_search_bound": False})
    assert not result["passed"]
    assert any("invalid" in reason for reason in result["reasons"])


@pytest.mark.parametrize("change", [
    {"samples": 201}, {"accepted": 199}, {"class_support": []},
    {"class_nll": [0.08]}, {"class_brier": [math.nan, 0.04]},
    {"class_support": [0, 0], "class_nll": [None, None], "class_brier": [None, None]},
])
def test_inconsistent_denominators_and_class_metrics_fail(change):
    raw, corrected = _metrics(), _corrected()
    corrected.update(change)
    result = promotion_gate(raw, corrected, {"at_search_bound": False})
    assert not result["passed"]
    assert any("invalid" in reason for reason in result["reasons"])
