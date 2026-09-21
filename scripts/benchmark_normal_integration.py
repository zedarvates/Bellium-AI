#!/usr/bin/env python3
"""Score the height integrators against controlled geometry with known heights.

Three published integrators are compared on the same normal fields: a
row-aligned cumulative integral, its row/column average twin, and a least-squares
Poisson solve. The world scale of a pixel is a declared input, because a normal
field carries orientation and not size, and every error is reported with the
unrecoverable additive constant removed.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bellium.material.integration import (  # noqa: E402
    DEFAULT_METHOD,
    METHODS,
    analytic_height,
    height_error,
    integrate_height,
)
from bellium.knn.integration_method import (  # noqa: E402
    deterministic_method,
    load_memory,
    measured_label,
    recommend_method,
)
from bellium.material.controlled import (  # noqa: E402
    perturbed_normals,
    procedural_albedo,
    to_image,
)
from bellium.material.photometric import (  # noqa: E402
    light_directions,
    multilight_capture,
    synthetic_geometry,
)
from bellium.specialists.photometric import recover_normals  # noqa: E402

REPORT = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "manifests"
    / "normal-integration-v1.json"
)
GEOMETRIES = ("sphere", "cone", "waves", "tilted-plane")
SIZE = 32
ITERATIONS = 400
RADIUS = SIZE * 0.45
LIGHTS = light_directions(6)
# The floor is a declared prior: every value here is measured, not chosen.
FLOORS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)
# The memory holds seed 1; the selector is scored on these seeds, never on its own
# exemplars.
SELECTOR_SEEDS = (2, 3)
NOISES = (0.0, 0.02, 0.05, 0.1)


def measured_fields(seed: int) -> list[dict]:
    """One perturbed field per geometry and noise level, with all four errors."""
    out = []
    for geometry in GEOMETRIES:
        normals, mask = synthetic_geometry(geometry, size=SIZE)
        truth, truth_mask = analytic_height(geometry, size=SIZE)
        scale = 1.0 if geometry == "waves" else 1.0 / RADIUS
        for level in NOISES:
            field = perturbed_normals(normals, level, seed=seed)
            errors = {}
            for method in METHODS:
                result = integrate_height(
                    field, mask, method=method, iterations=ITERATIONS, pixel_scale=scale
                )
                errors[method] = height_error(
                    result["height"], truth, truth_mask
                )["relative_rmse"]
            out.append({
                "geometry": geometry,
                "noise": level,
                "normals": field,
                "mask": mask,
                "errors": errors,
                "best": measured_label(errors),
            })
    return out


def selector_section() -> dict:
    """Score the k-NN method recommendation against the rule and the fixed default."""
    memory = load_memory()
    cases = []
    knn_hits = rule_hits = 0
    knn_cost = rule_cost = default_cost = oracle_cost = 0.0
    for seed in SELECTOR_SEEDS:
        for case in measured_fields(seed):
            answer = recommend_method(case["normals"], case["mask"], memory=memory)
            rule = deterministic_method(answer["features"])
            knn_hits += answer["method"] == case["best"]
            rule_hits += rule == case["best"]
            knn_cost += case["errors"][answer["method"]]
            rule_cost += case["errors"][rule]
            default_cost += case["errors"][DEFAULT_METHOD]
            oracle_cost += case["errors"][case["best"]]
            cases.append({
                "seed": seed,
                "geometry": case["geometry"],
                "noise": case["noise"],
                "best": case["best"],
                "recommended": answer["method"],
                "recommended_by": answer["status"],
                "baseline": rule,
                "recommended_error": case["errors"][answer["method"]],
                "baseline_error": case["errors"][rule],
                "best_error": case["errors"][case["best"]],
                "default_error": case["errors"][DEFAULT_METHOD],
            })
    total = len(cases)
    return {
        "note": (
            "The memory is built from seed 1 and scored here on seeds 2 and 3, so these "
            "numbers are held out. Errors are relative and offset-removed."
        ),
        "exemplars": len(memory["items"]),
        "cases": cases,
        "summary": {
            "runs": total,
            "knn_optimal": round(knn_hits / total, 4),
            "rule_optimal": round(rule_hits / total, 4),
            "knn_mean_error": round(knn_cost / total, 6),
            "rule_mean_error": round(rule_cost / total, 6),
            "default_mean_error": round(default_cost / total, 6),
            "oracle_mean_error": round(oracle_cost / total, 6),
        },
    }


def chain_section() -> dict:
    """Captures, then normals, then height, at several declared grazing floors.

    This is where the integration meets a field it did not receive as truth: the
    pixels near the silhouette carry a vertical component close to zero, and the
    slope they imply is noise.
    """
    normals, mask = synthetic_geometry("sphere", size=SIZE)
    truth, truth_mask = analytic_height("sphere", size=SIZE)
    albedo = [
        [sum(pixel) / 3.0 for pixel in row]
        for row in procedural_albedo("checker", size=SIZE, seed=4)
    ]
    captures = multilight_capture(albedo, normals, LIGHTS, ambient=0.05)
    images = [
        to_image([[(value, value, value) for value in row] for row in capture])
        for capture in captures
    ]
    fitted = recover_normals({"captures": images, "lights": LIGHTS})
    if fitted.abstained:
        raise SystemExit("the controlled captures abstained; fix the fixture first")
    cases = []
    for floor in FLOORS:
        start = time.perf_counter()
        result = integrate_height(
            fitted.output["normals"],
            mask,
            method=DEFAULT_METHOD,
            pixel_scale=1.0 / RADIUS,
            min_cosine=floor,
        )
        elapsed = (time.perf_counter() - start) * 1000.0
        kept = [
            [
                1.0 if (result["valid"][r][c] and truth_mask[r][c]) else 0.0
                for c in range(SIZE)
            ]
            for r in range(SIZE)
        ]
        offered = sum(
            1
            for r in range(SIZE)
            for c in range(SIZE)
            if fitted.output["normals"][r][c] is not None and mask[r][c] > 0.0
        )
        error = height_error(result["height"], truth, truth_mask)
        kept_error = height_error(result["height"], truth, kept)
        cases.append({
            "grazing_floor": floor,
            "pixels": result["pixels"],
            "offered": offered,
            "coverage": round(result["pixels"] / offered, 4),
            "relative_rmse_all": error["relative_rmse"],
            "relative_rmse_kept": kept_error["relative_rmse"],
            "latency_ms": round(elapsed, 3),
        })
    reference = integrate_height(
        normals, mask, method=DEFAULT_METHOD, pixel_scale=1.0 / RADIUS
    )
    reference_error = height_error(reference["height"], truth, truth_mask)
    return {
        "note": (
            "Controlled captures of a sphere with a known albedo, solved into normals "
            "by the published specialist, then integrated. relative_rmse_all counts "
            "refused pixels as error; relative_rmse_kept scores only the pixels the "
            "floor kept."
        ),
        "offered": cases[0]["offered"],
        "deterministic_normals_relative_rmse": reference_error["relative_rmse"],
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(REPORT), help="report path")
    options = parser.parse_args()
    report: dict = {
        "schema": "bellium.normal-integration-benchmark/v1",
        "note": (
            "Controlled synthetic geometry with known heights. The pixel scale is a "
            "declared input and every error removes the unrecoverable constant."
        ),
        "size": SIZE,
        "iterations": ITERATIONS,
        "methods": list(METHODS),
        "cases": [],
        "summary": {},
    }
    for geometry in GEOMETRIES:
        normals, mask = synthetic_geometry(geometry, size=SIZE)
        truth, truth_mask = analytic_height(geometry, size=SIZE)
        pixel_scale = 1.0 if geometry == "waves" else 1.0 / RADIUS
        for method in METHODS:
            start = time.perf_counter()
            result = integrate_height(
                normals, mask, method=method, iterations=ITERATIONS,
                pixel_scale=pixel_scale,
            )
            elapsed = (time.perf_counter() - start) * 1000.0
            error = height_error(result["height"], truth, truth_mask)
            record = {
                "geometry": geometry,
                "method": method,
                "pixel_scale": pixel_scale,
                "relative_rmse": error["relative_rmse"],
                "rmse": error["rmse"],
                "max_abs": error["max_abs"],
                "pixels": error["pixels"],
                "latency_ms": round(elapsed, 3),
            }
            if result["convergence"] is not None:
                record["last_change"] = result["convergence"]["last_change"]
            report["cases"].append(record)
            level = report["summary"].setdefault(method, {"relative_rmse": [],
                                                           "latencies": []})
            level["relative_rmse"].append(error["relative_rmse"] or 0.0)
            level["latencies"].append(elapsed)
    for level in report["summary"].values():
        latencies = level.pop("latencies")
        values = level.pop("relative_rmse")
        level["mean_relative_rmse"] = round(statistics.mean(values), 6)
        level["worst_relative_rmse"] = round(max(values), 6)
        level["p50_ms"] = round(statistics.median(latencies), 3)
    best = min(report["summary"], key=lambda name: report["summary"][name]
               ["mean_relative_rmse"])
    report["best_method"] = best
    report["chain"] = chain_section()
    report["selector"] = selector_section()
    destination = Path(options.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, level in report["summary"].items():
        print(
            f"{name:20s} mean_relative={level['mean_relative_rmse']:.4f} "
            f"worst={level['worst_relative_rmse']:.4f} p50={level['p50_ms']}ms"
        )
    for case in report["cases"]:
        convergence = case.get("last_change")
        print(
            f"  {case['geometry']:13s} {case['method']:20s} "
            f"relative={case['relative_rmse']} "
            f"{'converged ' + str(convergence) if convergence is not None else ''}"
        )
    print(f"best method on this set: {best}")
    chain = report["chain"]
    print(
        f"chain: {chain['offered']} pixels offered, deterministic normals at "
        f"{chain['deterministic_normals_relative_rmse']}"
    )
    for case in chain["cases"]:
        print(
            f"  floor={case['grazing_floor']:<5} kept={case['pixels']:4d} "
            f"coverage={case['coverage']:.3f} all={case['relative_rmse_all']} "
            f"kept={case['relative_rmse_kept']}"
        )
    selector = report["selector"]["summary"]
    print(
        f"selector: {selector['runs']} held-out fields, k-NN optimal "
        f"{selector['knn_optimal']:.2%} mean {selector['knn_mean_error']}, rule optimal "
        f"{selector['rule_optimal']:.2%} mean {selector['rule_mean_error']}, default "
        f"{selector['default_mean_error']}, oracle {selector['oracle_mean_error']}"
    )
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
