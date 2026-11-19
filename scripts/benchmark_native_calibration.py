"""Fresh synthetic calibration/evaluation gate; never changes weights or thresholds."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bellium.micro_nn.calibration import fit_temperature, predict_calibrated, predict_logits, probability_metrics  # noqa: E402
from bellium.micro_nn.mlp import load_mlp, predict_mlp  # noqa: E402
from scripts import train_inpaint_router, train_tool_router  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CALIBRATION_SEED = 2026100901
EVALUATION_SEED = 2026100902
THRESHOLDS = {"minimum_relative_nll_improvement": 0.02,
              "maximum_brier_increase": 0.0, "maximum_ece_increase": 0.001,
              "maximum_coverage_drop": 0.02, "maximum_selective_accuracy_drop": 0.002,
              "minimum_class_support": 50, "maximum_class_nll_increase": 0.01,
              "maximum_class_brier_increase": 0.002}


def metric_evidence_errors(metrics, prefix="metrics"):
    """Malformed or non-finite evidence must never win a numeric comparison."""
    if not isinstance(metrics, dict):
        return [f"invalid_{prefix}_object"]
    errors = []

    def finite(value, lower=0.0, upper=math.inf):
        try:
            return (not isinstance(value, bool) and isinstance(value, (int, float))
                    and math.isfinite(value) and lower <= value <= upper)
        except OverflowError:
            return False

    for name, upper in (("nll", math.inf), ("brier", 2.0), ("ece", 1.0),
                        ("accuracy", 1.0), ("coverage", 1.0)):
        if not finite(metrics.get(name), upper=upper):
            errors.append(f"invalid_{prefix}_{name}")
    for name in ("samples", "accepted", "abstained"):
        value = metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < (1 if name == "samples" else 0):
            errors.append(f"invalid_{prefix}_{name}")
    support = metrics.get("class_support")
    if (not isinstance(support, list) or len(support) < 2
            or any(isinstance(count, bool) or not isinstance(count, int) or count < 0 for count in support)):
        errors.append(f"invalid_{prefix}_class_support")
    else:
        if sum(support) != metrics.get("samples"):
            errors.append(f"invalid_{prefix}_class_denominator")
        for name, upper in (("class_nll", math.inf), ("class_brier", 2.0)):
            values = metrics.get(name)
            if not isinstance(values, list) or len(values) != len(support):
                errors.append(f"invalid_{prefix}_{name}_shape")
                continue
            if any(not finite(value, upper=upper) if count else value is not None
                   for value, count in zip(values, support)):
                errors.append(f"invalid_{prefix}_{name}")
            elif sum(support) > 0 and finite(metrics.get(name.removeprefix("class_")), upper=upper):
                mean = sum((value or 0.0) * (count / sum(support)) for value, count in zip(values, support))
                if not math.isclose(mean, metrics[name.removeprefix("class_")], rel_tol=1e-7, abs_tol=1e-10):
                    errors.append(f"invalid_{prefix}_{name}_aggregate")
    if not any(f"invalid_{prefix}_{name}" in errors for name in ("samples", "accepted", "abstained")):
        if metrics["accepted"] + metrics["abstained"] != metrics["samples"]:
            errors.append(f"invalid_{prefix}_acceptance_denominator")
        if finite(metrics.get("coverage"), upper=1.0) and not math.isclose(
                metrics["coverage"], metrics["accepted"] / metrics["samples"], abs_tol=1e-10):
            errors.append(f"invalid_{prefix}_coverage_denominator")
        selective = metrics.get("selective_accuracy")
        if (metrics["accepted"] > 0 and not finite(selective, upper=1.0)) or (
                metrics["accepted"] == 0 and selective is not None):
            errors.append(f"invalid_{prefix}_selective_accuracy")
    return errors


def promotion_gate(raw, corrected, artifact):
    errors = metric_evidence_errors(raw, "raw") + metric_evidence_errors(corrected, "calibrated")
    if not isinstance(artifact, dict) or not isinstance(artifact.get("at_search_bound"), bool):
        errors.append("invalid_calibration_artifact")
    if not errors and (raw["samples"] != corrected["samples"] or raw["class_support"] != corrected["class_support"]):
        errors.append("invalid_evaluation_pair_support")
    if errors:
        return {"passed": False, "reasons": errors, "automatically_applied": False}
    reasons = []
    if corrected["nll"] > raw["nll"] * (1 - THRESHOLDS["minimum_relative_nll_improvement"]):
        reasons.append("insufficient_final_nll_improvement")
    if corrected["brier"] > raw["brier"] + THRESHOLDS["maximum_brier_increase"]:
        reasons.append("final_brier_regression")
    if corrected["ece"] > raw["ece"] + THRESHOLDS["maximum_ece_increase"]:
        reasons.append("final_ece_regression")
    if corrected["coverage"] < raw["coverage"] - THRESHOLDS["maximum_coverage_drop"]:
        reasons.append("excessive_coverage_drop")
    if (corrected["selective_accuracy"] is None or raw["selective_accuracy"] is None
            or corrected["selective_accuracy"] < raw["selective_accuracy"] - THRESHOLDS["maximum_selective_accuracy_drop"]):
        reasons.append("selective_accuracy_regression_or_unsupported")
    if corrected["accuracy"] != raw["accuracy"]:
        reasons.append("class_decisions_changed")
    for index, count in enumerate(raw["class_support"]):
        if count < THRESHOLDS["minimum_class_support"]:
            reasons.append(f"class_{index}_insufficient_support")
            continue
        if corrected["class_nll"][index] > raw["class_nll"][index] + THRESHOLDS["maximum_class_nll_increase"]:
            reasons.append(f"class_{index}_nll_regression")
        if corrected["class_brier"][index] > raw["class_brier"][index] + THRESHOLDS["maximum_class_brier_increase"]:
            reasons.append(f"class_{index}_brier_regression")
    if artifact["at_search_bound"]:
        reasons.append("optimum_at_search_bound")
    return {"passed": not reasons, "reasons": reasons, "automatically_applied": False}


def benchmark():
    report = {"schema": "bellium.native-calibration-gate/v1",
              "scope": "fresh synthetic rule-fidelity labels; no field calibration claim",
              "thresholds_declared_before_fitting": THRESHOLDS,
              "calibration_seed": CALIBRATION_SEED, "evaluation_seed": EVALUATION_SEED,
              "confidence_threshold_unchanged": 0.55, "ece_bins": 10, "models": {}}
    for name, trainer, train_size, train_seed, held_seed in (
        ("tool-router", train_tool_router, 900, 5, 21),
        ("inpaint-router", train_inpaint_router, 800, 11, 99),
    ):
        path = ROOT / "models/micro_nn" / name / "v0.json"
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        model = load_mlp(path)
        calibration = trainer.make_samples(1500, random.Random(CALIBRATION_SEED))
        evaluation = trainer.make_samples(2500, random.Random(EVALUATION_SEED))
        prior = (trainer.make_samples(train_size, random.Random(train_seed))
                 + trainer.make_samples(200, random.Random(held_seed))
                 + trainer.make_samples(1000, random.Random(20261009)))
        sets = [set(tuple(vec) for vec, _ in samples) for samples in (calibration, evaluation, prior)]
        assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
        artifact = fit_temperature(model, calibration)
        logits = [predict_logits(model, vector) for vector, _ in evaluation]
        labels = [label for _, label in evaluation]
        raw = probability_metrics(logits, labels)
        corrected = probability_metrics(logits, labels, temperature=artifact["temperature"])
        for vector, _ in evaluation:
            original = predict_mlp(model, vector)
            scaled = predict_calibrated(model, vector, artifact)
            assert max(range(len(original)), key=original.__getitem__) == max(range(len(scaled)), key=scaled.__getitem__)
        after = hashlib.sha256(path.read_bytes()).hexdigest()
        assert before == after
        report["models"][name] = {"model_sha256_before": before, "model_sha256_after": after,
                                  "split_sizes": [1500, 2500], "feature_overlap": 0,
                                  "calibration_artifact": artifact, "evaluation_raw": raw,
                                  "evaluation_calibrated": corrected,
                                  "promotion": promotion_gate(raw, corrected, artifact)}
        print(f"{name}: T={artifact['temperature']:.4f}, NLL {raw['nll']:.6f} -> "
              f"{corrected['nll']:.6f}, gate={report['models'][name]['promotion']['passed']}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = benchmark()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
