"""Synthetic calibration lot with exact ground truth for the inpainting guard.

Deterministic, license-free, and generated at run time. Measures whether the
known-context probes accept reconstructible content and refuse noise.
"""
from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bellium.knn.patch_inpaint import inpaint  # noqa: E402
from bellium.specialists.adaptive_inpaint import inpaint as adaptive_inpaint  # noqa: E402

SIZE = 96
MAX_CONTEXT_MAE = 12.0


def _gradient(rng):
    return [[(40 + int(120 * r / SIZE), 60 + int(80 * c / SIZE), 90) for c in range(SIZE)] for r in range(SIZE)]


def _ramp_noise(rng):
    return [[(30 + int(2 * r), 40 + int(2 * c), 120) for c in range(SIZE)] for r in range(SIZE)]


def _stripes(rng):
    return [[(220, 210, 200) if (c // 6) % 2 else (40, 50, 60) for c in range(SIZE)] for r in range(SIZE)]


def _checker(rng):
    return [[(210, 40, 40) if ((r // 8) + (c // 8)) % 2 else (30, 60, 200) for c in range(SIZE)] for r in range(SIZE)]


def _blurred_noise(rng):
    base = [[rng.randrange(256) for _ in range(SIZE + 2)] for _ in range(SIZE + 2)]
    out = []
    for r in range(SIZE):
        row = []
        for c in range(SIZE):
            value = sum(base[r + dr][c + dc] for dr in (0, 1, 2) for dc in (0, 1, 2)) / 9.0
            row.append((int(value), int(value), int(value)))
        out.append(row)
    return out


def _white_noise(rng):
    return [[tuple(rng.randrange(256) for _ in range(3)) for _ in range(SIZE)] for _ in range(SIZE)]


FAMILIES = {
    "gradient": (_gradient, "predictable"),
    "ramp": (_ramp_noise, "predictable"),
    "stripes": (_stripes, "predictable"),
    "checker": (_checker, "predictable"),
    "blurred_noise": (_blurred_noise, "partially_predictable"),
    "white_noise": (_white_noise, "unpredictable"),
}
CENTER = SIZE // 2


def _hole(side):
    return [[int(CENTER - side // 2 <= r < CENTER - side // 2 + side
                and CENTER - side // 2 <= c < CENTER - side // 2 + side)
             for c in range(SIZE)] for r in range(SIZE)]


def _nearest_known(corrupted, mask):
    known = [(r, c) for r in range(SIZE) for c in range(SIZE) if not mask[r][c]]
    out = [row[:] for row in corrupted]
    for r in range(SIZE):
        for c in range(SIZE):
            if mask[r][c]:
                sr, sc = min(known, key=lambda p: ((p[0]-r)**2 + (p[1]-c)**2, p))
                out[r][c] = corrupted[sr][sc]
    return out


def _ring_mean(corrupted, mask):
    missing = [(r, c) for r in range(SIZE) for c in range(SIZE) if mask[r][c]]
    r0, r1 = min(r for r, _ in missing), max(r for r, _ in missing)
    c0, c1 = min(c for _, c in missing), max(c for _, c in missing)
    ring = [corrupted[r][c] for r in range(max(0, r0-1), min(SIZE, r1+2))
            for c in range(max(0, c0-1), min(SIZE, c1+2)) if not mask[r][c]]
    mean = tuple(round(sum(p[i] for p in ring)/len(ring)) for i in range(3))
    out = [row[:] for row in corrupted]
    for r, c in missing:
        out[r][c] = mean
    return out


def _metrics(original, predicted, mask):
    errors = [abs(original[r][c][i] - predicted[r][c][i])
              for r in range(SIZE) for c in range(SIZE) if mask[r][c] for i in range(3)]
    return {"mae_255": sum(errors) / len(errors)}


def run(engine_name, engine, cases_per_family=3):
    cases = []
    for family, (builder, expectation) in FAMILIES.items():
        for trial in range(cases_per_family):
            rng = random.Random(1000 + trial)
            original = builder(rng)
            for side in (3, 7):
                mask = _hole(side)
                corrupted = [[(0, 0, 0) if mask[r][c] else original[r][c] for c in range(SIZE)] for r in range(SIZE)]
                start = time.perf_counter()
                result = engine(corrupted, mask)
                elapsed = time.perf_counter() - start
                prediction = result.output.get("image", corrupted)
                records = {"family": family, "expectation": expectation, "trial": trial, "side": side,
                           "engine": engine_name, "accepted": not result.abstained,
                           "reason": result.output.get("reason"), "seconds": elapsed}
                records.update(_metrics(original, prediction, mask))
                # Method-independent reference: can a trivial local rule do it?
                records["simple_baseline_mae_255"] = min(
                    _metrics(original, _nearest_known(corrupted, mask), mask)["mae_255"],
                    _metrics(original, _ring_mean(corrupted, mask), mask)["mae_255"],
                )
                cases.append(records)
    return cases


def summarize(cases, limits):
    rows = []
    for family in FAMILIES:
        subset = [c for c in cases if c["family"] == family]
        accepted = [c for c in subset if c["accepted"]]
        rows.append({
            "family": family, "declared_expectation": FAMILIES[family][1], "cases": len(subset),
            "accepted": len(accepted),
            "accept_rate": len(accepted) / len(subset),
            "accepted_mean_mae_255": statistics.mean(c["mae_255"] for c in accepted) if accepted else None,
            "severe_accepted": sum(c["mae_255"] > 25 for c in accepted),
            "best_simple_baseline_mae_255": statistics.mean(c["simple_baseline_mae_255"] for c in subset),
        })
    # Labels come from an independent simple interpolator, not from the filler.
    reconstructible = [c for c in cases if c["simple_baseline_mae_255"] <= limits["reconstructible_mae_255"]]
    unreconstructible = [c for c in cases if c["simple_baseline_mae_255"] > limits["unreconstructible_mae_255"]]
    coverage = (sum(c["accepted"] for c in reconstructible) / len(reconstructible)) if reconstructible else None
    leakage = (sum(c["accepted"] for c in unreconstructible) / len(unreconstructible)) if unreconstructible else None
    gates = {
        "reconstructible_coverage": coverage is not None and coverage >= limits["minimum_predictable_coverage"],
        "unreconstructible_leakage": leakage is not None and leakage <= limits["maximum_unpredictable_acceptance"],
        "accepted_error": all(c["mae_255"] <= limits["maximum_accepted_mae_255"] for c in cases if c["accepted"]),
    }
    return {"rows": rows, "reconstructible_cases": len(reconstructible),
            "unreconstructible_cases": len(unreconstructible),
            "reconstructible_coverage": coverage, "unreconstructible_acceptance": leakage,
            "gates": gates, "passed": all(gates.values())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", choices=["patch", "adaptive"], default="adaptive")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("choose a new output path")
    limits = {"minimum_predictable_coverage": 0.80, "maximum_unpredictable_acceptance": 0.10,
              "maximum_accepted_mae_255": MAX_CONTEXT_MAE,
              "reconstructible_mae_255": 12.0, "unreconstructible_mae_255": 25.0}
    engine = inpaint if args.engine == "patch" else adaptive_inpaint
    cases = run(args.engine, engine)
    report = {"schema": "bellium.inpaint-guard-calibration/v1", "engine": args.engine,
              "synthetic_only": True, "size": SIZE, "families": {k: v[1] for k, v in FAMILIES.items()},
              "python": sys.version, "platform": platform.platform(),
              "summary": summarize(cases, limits), "cases": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0 if report["summary"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
