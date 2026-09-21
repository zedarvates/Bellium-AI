#!/usr/bin/env python3
"""Measure 2D drafting round-trips on hand-authored fixtures.

Every case is authored geometry. The report is reproducible without assets.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bellium.drafting.document import check_drawing, drawing_to_dict, drawings_match
from bellium.drafting.dxf import drawing_from_dxf, drawing_to_dxf
from bellium.drafting.fixtures import FIXTURE_NAMES, fixture
from bellium.drafting.svg import drawing_from_svg, drawing_to_svg
from bellium.specialists.drafting import inspect_drawing

REPORT = Path(__file__).resolve().parents[1] / "benchmarks" / "manifests" / "drafting-roundtrip-v1.json"
REPEATS = 20


def _ms(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
    return {
        "p50": statistics.median(ordered) * 1000.0,
        "p95": ordered[p95_index] * 1000.0,
    }


def main() -> int:
    cases = []
    violations = 0
    svg_mismatches = 0
    dxf_mismatches = 0
    specialist_failures = 0
    latencies: list[float] = []
    for name in FIXTURE_NAMES:
        drawing = fixture(name)
        local_violations = check_drawing(drawing)
        violations += len(local_violations)
        svg_times = []
        dxf_times = []
        for _ in range(REPEATS):
            started = time.perf_counter()
            svg = drawing_to_svg(drawing)
            svg_back = drawing_from_svg(svg)
            svg_times.append(time.perf_counter() - started)
            started = time.perf_counter()
            dxf = drawing_to_dxf(drawing)
            dxf_back = drawing_from_dxf(dxf)
            dxf_times.append(time.perf_counter() - started)
        svg_ok = drawings_match(drawing, svg_back)
        dxf_ok = drawings_match(drawing, dxf_back)
        svg_mismatches += 0 if svg_ok else 1
        dxf_mismatches += 0 if dxf_ok else 1
        result = inspect_drawing({"drawing": drawing_to_dict(drawing)})
        if result.output.get("status") != "ready":
            specialist_failures += 1
        latencies.extend(svg_times)
        latencies.extend(dxf_times)
        cases.append(
            {
                "name": name,
                "unit": drawing.unit,
                "entities": len(drawing.entities),
                "layers": len(drawing.layers),
                "svg_roundtrip": svg_ok,
                "dxf_roundtrip": dxf_ok,
                "violations": local_violations,
                "specialist_status": result.output.get("status"),
                "svg_ms": _ms(svg_times),
                "dxf_ms": _ms(dxf_times),
                "svg_bytes": len(svg.encode("utf-8")),
                "dxf_bytes": len(dxf.encode("utf-8")),
            }
        )
    payload = {
        "gate": "drafting-roundtrip-v1",
        "specialist_id": "bellium/deterministic/drafting-document:v0",
        "authority": "consultative",
        "fixtures": list(FIXTURE_NAMES),
        "repeats": REPEATS,
        "invariant_violations": violations,
        "svg_mismatches": svg_mismatches,
        "dxf_mismatches": dxf_mismatches,
        "specialist_failures": specialist_failures,
        "latency_ms": _ms(latencies),
        "cases": cases,
    }
    REPORT.write_text(json.dumps(payload, indent=2) + chr(10), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "invariant_violations", "svg_mismatches", "dxf_mismatches",
        "specialist_failures", "latency_ms",
    )}, indent=2))
    print(f"wrote {REPORT}")
    if violations or svg_mismatches or dxf_mismatches or specialist_failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
