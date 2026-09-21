#!/usr/bin/env python3
"""Benchmark horizon-based ambient occlusion on controlled synthetic topographies.

Scores geometric accessibility [0.0, 1.0] against analytic and controlled relief cases
(crevice, hemisphere, cone, sinusoidal waves, and planar ramp).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bellium.material.ambient_occlusion import horizon_ambient_occlusion  # noqa: E402
from bellium.material.controlled import sampled_geometry, sampled_height  # noqa: E402
from bellium.specialists.ambient_occlusion import estimate_ambient_occlusion  # noqa: E402

REPORT = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "manifests"
    / "ambient-occlusion-v1.json"
)
SIZE = 32
GEOMETRIES = ("sphere", "cone", "waves", "tilted-plane")


def v_groove_fixture(size: int = SIZE, slope: float = 1.0) -> tuple[list[list[float]], list[list[float]]]:
    """Symmetric V-groove with known horizon angle atan(slope) along x axis."""
    center = (size - 1) / 2.0
    height = [[slope * abs(c - center) for c in range(size)] for r in range(size)]
    valid = [[1.0] * size for _ in range(size)]
    return height, valid


def hemispherical_pit_fixture(size: int = SIZE, radius: float = 12.0) -> tuple[list[list[float]], list[list[float]]]:
    """Inverted dome (pit) where occlusion increases quadratically towards the center."""
    center = (size - 1) / 2.0
    height = [[0.0] * size for _ in range(size)]
    valid = [[1.0] * size for _ in range(size)]
    for r in range(size):
        for c in range(size):
            d2 = (r - center) ** 2 + (c - center) ** 2
            if d2 <= radius * radius:
                height[r][c] = -math.sqrt(max(0.0, radius * radius - d2))
    return height, valid


def run_benchmark() -> dict:
    cases = []

    # 1. Analytic V-groove at different slopes
    for slope in (0.5, 1.0, 2.0):
        h, v = v_groove_fixture(size=SIZE, slope=slope)
        center_col = int((SIZE - 1) / 2.0)
        start = time.perf_counter()
        res = horizon_ambient_occlusion(h, v, radius=6, directions=8)
        elapsed = (time.perf_counter() - start) * 1000.0
        center_ao = res["ao"][center_col][center_col]
        cases.append({
            "fixture": "v-groove",
            "parameter": f"slope={slope}",
            "mean_ao": res["mean_ao"],
            "center_ao": center_ao,
            "latency_ms": round(elapsed, 3),
            "pixels": res["pixels"],
        })

    # 2. Pit fixture
    h_pit, v_pit = hemispherical_pit_fixture(size=SIZE, radius=10.0)
    center_col = int((SIZE - 1) / 2.0)
    start = time.perf_counter()
    res_pit = horizon_ambient_occlusion(h_pit, v_pit, radius=6, directions=8)
    elapsed_pit = (time.perf_counter() - start) * 1000.0
    cases.append({
        "fixture": "hemispherical-pit",
        "parameter": "radius=10.0",
        "mean_ao": res_pit["mean_ao"],
        "center_ao": res_pit["ao"][center_col][center_col],
        "latency_ms": round(elapsed_pit, 3),
        "pixels": res_pit["pixels"],
    })

    # 3. Standard synthetic geometries
    for geom in GEOMETRIES:
        h, v = sampled_height(geom, size=SIZE)
        start = time.perf_counter()
        res_geom = horizon_ambient_occlusion(h, v, radius=6, directions=8)
        elapsed_geom = (time.perf_counter() - start) * 1000.0
        
        # Also test specialist path integrated from normals
        normals, mask = sampled_geometry(geom, size=SIZE)
        start_spec = time.perf_counter()
        spec_res = estimate_ambient_occlusion({
            "normals": normals,
            "mask": mask,
            "radius": 6,
            "directions": 8,
        })
        elapsed_spec = (time.perf_counter() - start_spec) * 1000.0

        cases.append({
            "fixture": geom,
            "parameter": "analytic_height",
            "mean_ao": res_geom["mean_ao"],
            "latency_ms": round(elapsed_geom, 3),
            "pixels": res_geom["pixels"],
        })
        cases.append({
            "fixture": geom,
            "parameter": "from_normals_integrated",
            "mean_ao": spec_res.output["mean_ao"],
            "latency_ms": round(elapsed_spec, 3),
            "pixels": spec_res.output["pixels"],
        })

    return {
        "schema": "bellium.ambient-occlusion-benchmark/v1",
        "note": "Horizon-based ambient occlusion benchmarking on controlled height fields and normals.",
        "size": SIZE,
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(REPORT), help="report path")
    options = parser.parse_args()

    report = run_benchmark()
    destination = Path(options.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for c in report["cases"]:
        print(
            f"{c['fixture']:18s} {c['parameter']:24s} mean_ao={c['mean_ao']:.4f} "
            f"latency={c['latency_ms']:.3f}ms pixels={c['pixels']}"
        )

    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
