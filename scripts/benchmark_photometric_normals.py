#!/usr/bin/env python3
"""Score photometric normals on controlled multi-light captures.

Every case carries its known normals, so the angular error can be split three
ways: all valid pixels, the patches the residual rule trusts, and the patches it
rejects. The last column is the point of the gate.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bellium.material.controlled import procedural_albedo  # noqa: E402
from bellium.material.photometric import (  # noqa: E402
    light_directions,
    multilight_capture,
    normal_error,
    photometric_normals,
    reliable_patches,
    reliability_features,
    synthetic_geometry,
)

REPORT = Path(__file__).resolve().parents[1] / "benchmarks" / "manifests" / "photometric-normals-v1.json"
GEOMETRIES = ("sphere", "cone", "waves", "tilted-plane")
ALBEDOS = ("flat", "checker", "stripes")
SPECULARS = (0.0, 0.1, 0.25)
SIZE = 32
PATCH = 8
MODEL_MISMATCH = 0.1
LOW_COVERAGE = 0.1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(REPORT), help="report path")
    options = parser.parse_args()
    lights = light_directions(6)
    summary: dict = {
        "schema": "bellium.photometric-normals-benchmark/v1",
        "note": (
            "Controlled synthetic multi-light captures with known normals. The light "
            "directions are declared inputs, and no photograph was used."
        ),
        "size": SIZE,
        "lights": [list(direction) for direction in lights],
        "levels": {},
        "cases": [],
    }
    for geometry in GEOMETRIES:
        for albedo_kind in ALBEDOS:
            for specular in SPECULARS:
                normals, mask = synthetic_geometry(geometry, size=SIZE)
                albedo = [
                    [sum(pixel) / 3.0 for pixel in row]
                    for row in procedural_albedo(albedo_kind, size=SIZE, seed=4)
                ]
                start = time.perf_counter()
                captures = multilight_capture(
                    albedo, normals, lights, ambient=0.05, specular=specular
                )
                fit = photometric_normals(captures, lights)
                patches = reliable_patches(
                    reliability_features(fit["residual"], captures, patch=PATCH)
                )
                trusted = [
                    index for patch in patches if patch["trusted"] for index in patch["indices"]
                ]
                rejected = [
                    index for patch in patches if not patch["trusted"] for index in patch["indices"]
                ]
                overall = normal_error(fit["normals"], normals, mask)
                gated = normal_error(fit["normals"], normals, mask, indices=trusted)
                outside = normal_error(fit["normals"], normals, mask, indices=rejected)
                residuals = [
                    fit["residual"][r][c]
                    for r in range(len(fit["residual"]))
                    for c in range(len(fit["residual"][0]))
                    if fit["normals"][r][c] is not None
                ]
                mean_residual = (
                    sum(residuals) / len(residuals) if residuals else 1.0
                )
                elapsed = (time.perf_counter() - start) * 1000.0
                record = {
                    "case": f"{geometry}/{albedo_kind}/spec={specular}",
                    "geometry": geometry,
                    "albedo_kind": albedo_kind,
                    "specular": specular,
                    "pixels": overall["pixels"],
                    "mean_deg": overall["mean_deg"],
                    "p95_deg": overall["p95_deg"],
                    "trusted_pixels": gated["pixels"],
                    "trusted_mean_deg": gated["mean_deg"],
                    "trusted_p95_deg": gated["p95_deg"],
                    "rejected_pixels": outside["pixels"],
                    "rejected_mean_deg": outside["mean_deg"],
                    "mean_residual": round(mean_residual, 6),
                    "specialist_abstains": bool(
                        mean_residual > MODEL_MISMATCH
                        or (gated["pixels"] / overall["pixels"] < LOW_COVERAGE
                            if overall["pixels"] else True)
                    ),
                    "latency_ms": elapsed,
                }
                summary["cases"].append(record)
                level = summary["levels"].setdefault(f"specular={specular:.2f}", {
                    "cases": 0, "mean_deg": [], "trusted_mean_deg": [], "rejected_mean_deg": [],
                    "trusted_pixels": 0, "pixels": 0, "latencies": [],
                    "abstentions": 0,
                })
                level["cases"] += 1
                level["mean_deg"].append(overall["mean_deg"] or 0.0)
                if gated["mean_deg"] is not None:
                    level["trusted_mean_deg"].append(gated["mean_deg"])
                if outside["mean_deg"] is not None:
                    level["rejected_mean_deg"].append(outside["mean_deg"])
                level["trusted_pixels"] += gated["pixels"]
                level["pixels"] += overall["pixels"]
                level["latencies"].append(elapsed)
                level["abstentions"] += int(record["specialist_abstains"])
    for level in summary["levels"].values():
        latencies = level.pop("latencies")
        for key in ("mean_deg", "trusted_mean_deg", "rejected_mean_deg"):
            values = level.pop(key)
            level[f"mean_{key}"] = round(statistics.mean(values), 4) if values else None
        level["trusted_fraction"] = round(
            level["trusted_pixels"] / level["pixels"], 4
        ) if level["pixels"] else None
        level["p50_ms"] = round(statistics.median(latencies), 4)
    destination = Path(options.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for name, level in summary["levels"].items():
        print(
            f"{name} cases={level['cases']} all={level['mean_mean_deg']}deg "
            f"trusted={level['mean_trusted_mean_deg']}deg "
            f"rejected={level['mean_rejected_mean_deg']}deg "
            f"trusted_fraction={level['trusted_fraction']} "
            f"abstentions={level['abstentions']}/{level['cases']} p50={level['p50_ms']}ms"
        )
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
