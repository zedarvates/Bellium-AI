"""Evaluate native tool-router training without replacing distributed weights."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bellium.knn.tool_router import FEATURE_NAMES, retrieve_tool  # noqa: E402
from bellium.micro_nn.evaluate import classification_gate, classification_metrics  # noqa: E402
from bellium.micro_nn.mlp import load_mlp, save_mlp  # noqa: E402
from bellium.micro_nn.train import train_classifier  # noqa: E402
from scripts.train_tool_router import DEST, _label, make_samples  # noqa: E402
from bellium.specialists.tool_router import classify_tool_need, route_tool  # noqa: E402


THRESHOLDS = {"min_accuracy": 0.95, "min_recall": 0.90, "min_support": 10}
SEEDS = (5, 17, 29)


def boundary_samples():
    """Independent authored probes around mutation=0.5 and criticality=0.85."""
    samples = []
    for code, files, error, external in itertools.product((0.0, 1.0), repeat=4):
        for mutation in (0.499, 0.5, 0.501):
            for criticality in (0.849, 0.85, 0.851):
                vector = [code, files, error, mutation, external, criticality]
                samples.append((vector, _label(*vector)))
    return samples


def evaluate(model, test, boundary):
    result = {}
    for name, samples in (("test", test), ("boundary", boundary)):
        result[name] = classification_metrics(model, samples)
        # Boundary class 'none' has only one independent probe by construction.
        thresholds = dict(THRESHOLDS, min_support=1) if name == "boundary" else THRESHOLDS
        result[f"{name}_gate"] = classification_gate(result[name], **thresholds)
    return result


def check_policy(boundary):
    checked = 0
    for vector, target in boundary:
        if target != 2:
            continue
        query = {"features": dict(zip(FEATURE_NAMES, vector))}
        result = retrieve_tool(query, memory=[])
        if result.output["status"] != "rule_escalate" or result.output["tool"] != "escalate":
            raise AssertionError("deterministic escalation failed")
        hybrid = route_tool(query)
        if hybrid.output["tool"] != "escalate" or hybrid.output["micro_nn"] is not None:
            raise AssertionError("hybrid did not preserve deterministic veto")
        checked += 1
    return {"escalation_cases": checked, "all_escalated_before_models": True}


def check_runtime_domain(boundary):
    abstained = non_veto_abstained = 0
    for vector, target in boundary:
        query = {"features": dict(zip(FEATURE_NAMES, vector))}
        result = classify_tool_need(query)
        if not result.abstained or result.output.get("reason") != "outside_training_domain":
            raise AssertionError("unsupported binary-flag interpolation was not rejected")
        abstained += 1
        if target != 2:
            hybrid = route_tool(query)
            if not hybrid.abstained:
                raise AssertionError("unsupported non-veto input received a hybrid proposal")
            non_veto_abstained += 1
    return {"unsupported_raw_inputs_abstained": abstained,
            "non_veto_hybrid_inputs_abstained": non_veto_abstained}


def benchmark():
    before = hashlib.sha256(DEST.read_bytes()).hexdigest()
    train = make_samples(900, random.Random(5))
    validation = make_samples(300, random.Random(31))
    test = make_samples(1000, random.Random(20261009))
    boundary = boundary_samples()
    split_sets = [set(tuple(vec) for vec, _ in split) for split in (train, validation, test)]
    if any(split_sets[a] & split_sets[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise AssertionError("training, validation and test contain overlapping feature vectors")
    report = {
        "schema": "bellium.tool-router-training-gate/v1",
        "scope": "synthetic rule fidelity; not field quality or calibrated confidence",
        "python": sys.version, "platform": platform.platform(),
        "thresholds_declared_before_training": THRESHOLDS,
        "split_seeds": {"train": 5, "validation": 31, "test": 20261009},
        "split_sizes": {"train": 900, "validation": 300, "test": 1000,
                        "boundary": len(boundary)},
        "split_overlap": 0,
        "labels": ["none", "use_tool", "escalate"],
        "published_sha256_before": before,
        "published": evaluate(load_mlp(DEST), test, boundary),
        "deterministic_rule_accuracy": 1.0,
        "policy_boundary": check_policy(boundary),
        "runtime_domain_boundary": check_runtime_domain(boundary),
        "runs": [],
    }
    candidate = None
    for seed in SEEDS:
        row = {"initialization_seed": seed}
        for name, options in (
            ("fixed", {}),
            ("validation", {"validation_samples": validation, "patience": 20,
                            "min_delta": 0.0001}),
        ):
            start = time.perf_counter()
            model = train_classifier(train, layer_sizes=[6, 8, 3], epochs=220,
                                     lr=0.12, seed=seed, **options)
            seconds = time.perf_counter() - start
            row[name] = evaluate(model, test, boundary)
            row[name].update({"training_seconds": seconds,
                              "epochs_run": model.get("training", {}).get("epochs_run", 220),
                              "selection": model.get("training")})
            # Predeclared seed, never select a winner using final-test results.
            if seed == 5 and name == "validation":
                candidate = model
        report["runs"].append(row)
        print(f"seed {seed}: fixed={row['fixed']['test']['accuracy']:.3f}, "
              f"validation={row['validation']['test']['accuracy']:.3f}, "
              f"epochs={row['validation']['epochs_run']}", flush=True)
    primary = report["runs"][0]["validation"]
    failures = list(primary["test_gate"]["reasons"])
    failures.extend(f"boundary_{reason}" for reason in primary["boundary_gate"]["reasons"])
    for name in ("test", "boundary"):
        if primary[name]["accuracy"] < report["published"][name]["accuracy"]:
            failures.append(f"{name}_regresses_against_published")
    report["promotion"] = {"passed": not failures, "reasons": failures,
                           "candidate_seed": 5, "shipped": False}
    after = hashlib.sha256(DEST.read_bytes()).hexdigest()
    if before != after:
        raise AssertionError("published weights changed during benchmark")
    report["published_sha256_after"] = after
    return report, candidate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path)
    args = parser.parse_args(argv)
    paths = [args.output] + ([args.candidate_output] if args.candidate_output else [])
    if any(path.resolve() == DEST.resolve() for path in paths):
        parser.error("evaluation must not overwrite distributed weights")
    if len(paths) == 2 and paths[0].resolve() == paths[1].resolve():
        parser.error("report and candidate must have distinct paths")
    report, candidate = benchmark()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    if args.candidate_output:
        if not report["promotion"]["passed"]:
            print("Candidate gate rejected; candidate output preserved.")
            return 1
        candidate.update({"name": "tool_router", "labels": report["labels"],
                          "authority_mode": "consultative",
                          "evaluation": report["promotion"]})
        save_mlp(args.candidate_output, candidate)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
