"""Predeclared repeated-split calibration study; no seed selection or model writes."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bellium.micro_nn.calibration import fit_temperature, predict_calibrated, predict_logits, probability_metrics  # noqa: E402
from bellium.micro_nn.mlp import load_mlp, predict_mlp  # noqa: E402
from scripts import train_inpaint_router, train_tool_router  # noqa: E402
from scripts.benchmark_native_calibration import THRESHOLDS, metric_evidence_errors, promotion_gate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "benchmarks/protocols/native-calibration-stability-v1.md"
PAIRS = tuple((2026101000 + 2 * index, 2026101001 + 2 * index) for index in range(10))
STABILITY_POLICY = {"required_pairs": 10, "minimum_passing_pairs": 8,
                    "pooled_quality_gate_required": True}


def pool_metrics(metrics):
    """Pool denominators and reliability-bin statistics, not the absolute ECE gaps."""
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("need metric rows")
    for row in metrics:
        errors = metric_evidence_errors(row)
        if errors:
            raise ValueError("invalid metrics: " + ",".join(errors))
    classes = len(metrics[0]["class_support"])
    bins = metrics[0].get("reliability_bins")
    if not isinstance(bins, list) or not bins:
        raise ValueError("need reliability bins")
    count = len(bins)
    for row in metrics:
        if len(row["class_support"]) != classes or len(row.get("reliability_bins", [])) != count:
            raise ValueError("pooling requires identical class and bin layouts")
        if row.get("confidence_threshold") != metrics[0].get("confidence_threshold"):
            raise ValueError("pooling requires an unchanged abstention threshold")
        total = 0
        for index, bucket in enumerate(row["reliability_bins"]):
            if bucket.get("lower") != index / count or bucket.get("upper") != (index + 1) / count:
                raise ValueError("reliability bin boundaries are inconsistent")
            n = bucket.get("samples")
            if isinstance(n, bool) or not isinstance(n, int) or n < 0:
                raise ValueError("invalid reliability-bin count")
            total += n
            for key in ("mean_confidence", "accuracy"):
                value = bucket.get(key)
                if n:
                    if (isinstance(value, bool) or not isinstance(value, (float, int))
                            or not math.isfinite(value) or not 0 <= value <= 1):
                        raise ValueError("invalid reliability-bin value")
                elif value is not None:
                    raise ValueError("empty bins must expose null values")
        if total != row["samples"]:
            raise ValueError("reliability bins do not account for every sample")
        bin_hits = sum(bucket["accuracy"] * bucket["samples"] for bucket in row["reliability_bins"] if bucket["samples"])
        if not math.isclose(bin_hits / row["samples"], row["accuracy"], abs_tol=1e-10):
            raise ValueError("reliability-bin accuracy denominator is inconsistent")
        derived_ece = sum(bucket["samples"] / row["samples"]
                          * abs(bucket["accuracy"] - bucket["mean_confidence"])
                          for bucket in row["reliability_bins"] if bucket["samples"])
        if not math.isclose(derived_ece, row["ece"], rel_tol=1e-7, abs_tol=1e-10):
            raise ValueError("reliability-bin ECE is inconsistent with reported metrics")
    samples = sum(row["samples"] for row in metrics)
    supports = [sum(row["class_support"][i] for row in metrics) for i in range(classes)]
    accepted = sum(row["accepted"] for row in metrics)
    result = {"samples": samples, "class_support": supports, "accepted": accepted,
              "abstained": samples - accepted, "coverage": accepted / samples,
              "confidence_threshold": metrics[0]["confidence_threshold"]}
    for key in ("nll", "brier", "accuracy"):
        result[key] = sum(row[key] * row["samples"] / samples for row in metrics)
    for key in ("class_nll", "class_brier"):
        result[key] = [sum((row[key][i] or 0.0) * row["class_support"][i] / supports[i]
                          for row in metrics) if supports[i] else None for i in range(classes)]
    result["selective_accuracy"] = (sum((row["selective_accuracy"] or 0.0) * row["accepted"] / accepted
                                        for row in metrics) if accepted else None)
    pooled_bins = []
    ece = 0.0
    for index in range(count):
        n = sum(row["reliability_bins"][index]["samples"] for row in metrics)
        confidence = accuracy = None
        if n:
            confidence = sum((row["reliability_bins"][index]["mean_confidence"] or 0.0)
                             * row["reliability_bins"][index]["samples"] / n for row in metrics)
            accuracy = sum((row["reliability_bins"][index]["accuracy"] or 0.0)
                           * row["reliability_bins"][index]["samples"] / n for row in metrics)
            ece += n / samples * abs(confidence - accuracy)
        pooled_bins.append({"lower": index / count, "upper": (index + 1) / count,
                            "samples": n, "mean_confidence": confidence, "accuracy": accuracy})
    result["ece"] = ece
    result["reliability_bins"] = pooled_bins
    if metric_evidence_errors(result):
        raise ValueError("pooled metrics are inconsistent")
    return result


def summarize(runs):
    if len(runs) != STABILITY_POLICY["required_pairs"]:
        raise ValueError("stability summary requires all ten pairs")
    if sorted(run["pair"] for run in runs) != list(range(len(PAIRS))):
        raise ValueError("stability summary requires ten unique pair identifiers")
    for run in runs:
        if (run["calibration_seed"], run["evaluation_seed"]) != PAIRS[run["pair"]]:
            raise ValueError("split seeds do not match the predeclared protocol")
        temperature = run["artifact"].get("temperature")
        if (isinstance(temperature, bool) or not isinstance(temperature, (float, int))
                or not math.isfinite(temperature) or temperature <= 0):
            raise ValueError("invalid temperature evidence")
        if run["gate"] != promotion_gate(run["raw"], run["corrected"], run["artifact"]):
            raise ValueError("reported pair gate does not match recomputed evidence")
    raw = pool_metrics([run["raw"] for run in runs])
    corrected = pool_metrics([run["corrected"] for run in runs])
    pooled_gate = promotion_gate(raw, corrected,
                                 {"at_search_bound": any(run["artifact"]["at_search_bound"] for run in runs)})
    passing = sum(run["gate"]["passed"] for run in runs)
    reasons = []
    if passing < STABILITY_POLICY["minimum_passing_pairs"]:
        reasons.append("too_few_passing_pairs")
    if not pooled_gate["passed"]:
        reasons.append("pooled_quality_gate_failed")
    temperatures = [run["artifact"]["temperature"] for run in runs]
    differences = {key: [run["corrected"][key] - run["raw"][key] for run in runs]
                   for key in ("nll", "brier", "ece", "coverage")}
    return {"pairs": len(runs), "passing_pairs": passing,
            "temperature": {"min": min(temperatures), "median": statistics.median(temperatures),
                            "max": max(temperatures)},
            "metric_differences": {key: {"min": min(values), "mean": statistics.mean(values),
                                         "max": max(values)} for key, values in differences.items()},
            "nll_improved_pairs": sum(value < 0 for value in differences["nll"]),
            "rejection_counts": dict(Counter(reason for run in runs for reason in run["gate"]["reasons"])),
            "pooled_raw": raw, "pooled_corrected": corrected, "pooled_gate": pooled_gate,
            "procedure_stability_gate": {"passed": not reasons, "reasons": reasons,
                                         "automatically_applied": False,
                                         "individual_temperature_selected": False}}


def benchmark():
    report = {"schema": "bellium.native-calibration-stability/v1",
              "scope": "generated rule-fidelity labels, ten fresh split pairs, no field calibration claim",
              "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "quality_thresholds": THRESHOLDS, "stability_policy": STABILITY_POLICY,
              "seed_pairs": PAIRS, "models": {}}
    for name, trainer, n_train, train_seed, held_seed in (
        ("tool-router", train_tool_router, 900, 5, 21),
        ("inpaint-router", train_inpaint_router, 800, 11, 99),
    ):
        path = ROOT / "models/micro_nn" / name / "v0.json"
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        model = load_mlp(path)
        prior = []
        for n, seed in ((n_train, train_seed), (200, held_seed), (300, 31), (1000, 20261009),
                        (1500, 2026100901), (2500, 2026100902)):
            prior.extend(trainer.make_samples(n, random.Random(seed)))
        seen = {tuple(vector) for vector, _ in prior}
        runs = []
        for index, (cal_seed, eval_seed) in enumerate(PAIRS):
            calibration = trainer.make_samples(1500, random.Random(cal_seed))
            evaluation = trainer.make_samples(2500, random.Random(eval_seed))
            for split in (calibration, evaluation):
                vectors = {tuple(vector) for vector, _ in split}
                if len(vectors) != len(split) or vectors & seen:
                    raise ValueError("new features overlap prior/current split features")
                seen.update(vectors)
            artifact = fit_temperature(model, calibration)
            logits = [predict_logits(model, vector) for vector, _ in evaluation]
            labels = [label for _, label in evaluation]
            raw = probability_metrics(logits, labels)
            corrected = probability_metrics(logits, labels, temperature=artifact["temperature"])
            run = {"pair": index, "calibration_seed": cal_seed, "evaluation_seed": eval_seed,
                   "sizes": [1500, 2500], "feature_overlap": 0,
                   "artifact": artifact, "raw": raw, "corrected": corrected,
                   "gate": promotion_gate(raw, corrected, artifact)}
            runs.append(run)
            print(f"{name} pair {index + 1}/10: T={artifact['temperature']:.4f}, "
                  f"NLL delta={corrected['nll'] - raw['nll']:+.6f}, gate={run['gate']['passed']}", flush=True)
        after = hashlib.sha256(path.read_bytes()).hexdigest()
        if before != after:
            raise ValueError("distributed model bytes changed")
        report["models"][name] = {"model_sha256_before": before, "model_sha256_after": after,
                                  "new_samples": 40000, "feature_overlap": 0,
                                  "runs": runs, "summary": summarize(runs)}
    return report


def verify_report(report):
    """Replay all saved evaluation seeds and temperatures without fitting again."""
    if report.get("schema") != "bellium.native-calibration-stability/v1":
        raise ValueError("stability report schema mismatch")
    if (report.get("quality_thresholds") != THRESHOLDS
            or report.get("stability_policy") != STABILITY_POLICY
            or [tuple(pair) for pair in report.get("seed_pairs", [])] != list(PAIRS)):
        raise ValueError("report does not match the predeclared policy")
    if report.get("protocol_sha256") != hashlib.sha256(PROTOCOL.read_bytes()).hexdigest():
        raise ValueError("protocol bytes changed")
    verified = deepcopy(report)
    if set(verified["models"]) != {"tool-router", "inpaint-router"}:
        raise ValueError("stability report needs both predeclared models")
    for name, trainer in (("tool-router", train_tool_router), ("inpaint-router", train_inpaint_router)):
        block = verified["models"][name]
        path = ROOT / "models/micro_nn" / name / "v0.json"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != block["model_sha256_before"] or digest != block["model_sha256_after"]:
            raise ValueError("model bytes do not match the report")
        if summarize(block["runs"]) != block["summary"]:
            raise ValueError("summary does not match pair evidence")
        model = load_mlp(path)
        decisions = 0
        for run in block["runs"]:
            samples = trainer.make_samples(2500, random.Random(run["evaluation_seed"]))
            logits = [predict_logits(model, vector) for vector, _ in samples]
            labels = [label for _, label in samples]
            raw = probability_metrics(logits, labels)
            corrected = probability_metrics(logits, labels, temperature=run["artifact"]["temperature"])
            if raw != run["raw"] or corrected != run["corrected"]:
                raise ValueError("replayed metrics do not match saved evaluation")
            for vector, _ in samples:
                original = predict_mlp(model, vector)
                calibrated = predict_calibrated(model, vector, run["artifact"])
                if max(range(len(original)), key=original.__getitem__) != max(range(len(calibrated)), key=calibrated.__getitem__):
                    raise ValueError("calibration changed an actual model decision")
                decisions += 1
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("model changed during replay")
        block["verification"] = {"evaluation_metrics_replayed": True, "model_decisions_checked": decisions,
                                 "decision_changes": 0, "refitting_performed": False}
        print(f"{name}: replayed ten evaluation sets, {decisions} actual decisions unchanged", flush=True)
    verified["verification_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return verified


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-report", type=Path)
    args = parser.parse_args()
    if args.verify_report:
        if args.output.resolve() == args.verify_report.resolve():
            parser.error("keep the original report and write verification to a separate path")
        report = verify_report(json.loads(args.verify_report.read_text(encoding="utf-8")))
    else:
        report = benchmark()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
