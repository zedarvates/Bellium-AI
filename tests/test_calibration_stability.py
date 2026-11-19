import copy
import math

import pytest

from bellium.micro_nn.calibration import probability_metrics
from scripts.benchmark_calibration_stability import PAIRS, pool_metrics, summarize
from scripts.benchmark_native_calibration import promotion_gate


def test_pooled_ece_recomputes_bins_instead_of_averaging_absolute_errors():
    row = [math.log(4.0), 0.0]
    first = probability_metrics([row] * 5, [0] * 5)
    second = probability_metrics([row] * 5, [0, 0, 0, 1, 1])
    pooled = pool_metrics([first, second])
    direct = probability_metrics([row] * 10, [0] * 8 + [1] * 2)
    assert (first["ece"] + second["ece"]) / 2 == pytest.approx(0.2)
    assert pooled["ece"] == pytest.approx(0.0, abs=1e-12)
    for key in ("nll", "brier", "accuracy", "coverage", "selective_accuracy", "ece"):
        assert pooled[key] == pytest.approx(direct[key], abs=1e-12)
    assert pooled["class_support"] == direct["class_support"]
    assert pooled["class_nll"] == pytest.approx(direct["class_nll"])


def test_pooling_uses_sample_and_acceptance_denominators():
    first = probability_metrics([[math.log(4.0), 0.0]] * 4, [0, 0, 0, 1])
    second = probability_metrics([[0.0, 0.0]], [1])
    pooled = pool_metrics([first, second])
    assert pooled["samples"] == 5
    assert pooled["accepted"] == 4
    assert pooled["selective_accuracy"] == 0.75
    assert pooled["coverage"] == 0.8
    assert pooled["class_support"] == [3, 2]


@pytest.mark.parametrize("change", [
    {"ece": math.nan}, {"class_support": [1, 1]}, {"samples": 100},
])
def test_malformed_metrics_cannot_be_pooled(change):
    metrics = probability_metrics([[1.0, 0.0]], [0])
    metrics.update(change)
    with pytest.raises(ValueError):
        pool_metrics([metrics])


@pytest.mark.parametrize("change", ["count", "boundary", "confidence", "ece"])
def test_reliability_bins_cannot_hide_missing_or_inconsistent_evidence(change):
    metrics = probability_metrics([[math.log(4.0), 0.0]] * 2, [0, 1])
    bucket = next(value for value in metrics["reliability_bins"] if value["samples"])
    if change == "count":
        bucket["samples"] += 1
    elif change == "boundary":
        bucket["lower"] = 0.123
    elif change == "confidence":
        bucket["mean_confidence"] = math.nan
    else:
        metrics["ece"] = 0.0
    with pytest.raises(ValueError):
        pool_metrics([metrics])


def _runs(passing):
    logits = [[math.log(4.0), 0.0]] * 100 + [[0.0, math.log(4.0)]] * 100
    labels = [0] * 95 + [1] * 5 + [1] * 95 + [0] * 5
    raw = probability_metrics(logits, labels)
    runs = []
    for index, (cal_seed, eval_seed) in enumerate(PAIRS):
        temperature = math.log(4) / math.log(19) if index < passing else 1.0
        artifact = {"temperature": temperature, "at_search_bound": False}
        corrected = probability_metrics(logits, labels, temperature=temperature)
        runs.append({"pair": index, "calibration_seed": cal_seed, "evaluation_seed": eval_seed,
                     "artifact": artifact, "raw": raw, "corrected": corrected,
                     "gate": promotion_gate(raw, corrected, artifact)})
    return runs


def test_eight_consistent_pairs_pass_without_selecting_an_artifact():
    runs = _runs(8)
    before = copy.deepcopy(runs)
    result = summarize(runs)
    assert result["passing_pairs"] == 8
    assert result["procedure_stability_gate"]["passed"]
    assert not result["procedure_stability_gate"]["individual_temperature_selected"]
    assert runs == before


def test_positive_pooled_result_does_not_hide_insufficient_stability():
    result = summarize(_runs(7))
    assert result["pooled_gate"]["passed"]
    assert not result["procedure_stability_gate"]["passed"]
    assert result["procedure_stability_gate"]["reasons"] == ["too_few_passing_pairs"]


@pytest.mark.parametrize("change", ["missing", "duplicate", "seed", "gate", "temperature"])
def test_stability_requires_all_predeclared_and_consistent_runs(change):
    runs = _runs(10)
    if change == "missing":
        runs.pop()
    elif change == "duplicate":
        runs[0]["pair"] = 1
    elif change == "seed":
        runs[0]["evaluation_seed"] = 42
    elif change == "gate":
        runs[0]["gate"]["passed"] = False
    else:
        runs[0]["artifact"]["temperature"] = math.nan
    with pytest.raises(ValueError):
        summarize(runs)
