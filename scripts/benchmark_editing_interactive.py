#!/usr/bin/env python3
"""Measure whether the editing path can be called interactive, per resolution.

The budget is declared before the measurement: a preview pipeline must stay at or
below 33 ms at 95th percentile to count as interactive, which is two frames at 60
Hz. Peak memory is measured with tracemalloc for the preview and for the full
resolution export, and the two are reported separately, never conflated.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bellium.editing import (  # noqa: E402
    EditStack,
    EditStep,
    apply_filter,
    apply_filter_fast,
    backend_for,
    numpy_available,
)
from bellium.editing.preview import PreviewSession, preview_stack  # noqa: E402

REPORT = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "manifests"
    / "editing-interactive-v1.json"
)
INTERACTIVE_BUDGET_MS = 33.0
PREVIEW_SIDE = 128
SIZES = (64, 192, 512)
OPS = (("sepia", {}), ("grayscale", {"strength": 0.6}))
REPEATS = 3


def _image(size: int, seed: int = 17):
    rng = random.Random(seed)
    return [
        [tuple(rng.randrange(256) for _ in range(3)) for _ in range(size)]
        for _ in range(size)
    ]


def _measure(function, repeats: int = REPEATS) -> tuple[float, float]:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1000.0)
    ordered = sorted(samples)
    return statistics.median(samples), ordered[int(0.95 * (len(ordered) - 1))]


def _pipeline(size: int, *, backend: str) -> float:
    image = _image(size)
    stack = EditStack(image)
    stack.add(EditStep("grayscale", {"strength": 0.7}))
    stack.add(EditStep("sepia", {"strength": 0.6}))
    stack.add(EditStep("brightness_contrast", {"contrast": 0.2}))
    _, p95 = _measure(
        lambda: preview_stack(stack, max_side=PREVIEW_SIDE, backend=backend),
        repeats=5,
    )
    return p95


def _peak_bytes(function) -> int:
    tracemalloc.start()
    function()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(REPORT), help="report path")
    options = parser.parse_args()
    report: dict = {
        "schema": "bellium.editing-interactive-benchmark/v1",
        "note": (
            "Synthetic images on this workstation. The budget is declared before the "
            "measurement and preview latency is never presented as export latency."
        ),
        "budget_ms": INTERACTIVE_BUDGET_MS,
        "preview_side": PREVIEW_SIDE,
        "numpy_available": numpy_available(),
        "backends": {name: backend_for(name, **parameters) for name, parameters in OPS},
        "resolutions": {},
    }
    for size in SIZES:
        image = _image(size)
        entry = {"size": size, "operations": {}}
        for name, parameters in OPS:
            plain_p50, plain_p95 = _measure(
                lambda name=name, parameters=parameters: apply_filter(image, name, **parameters)
            )
            fast_p50, fast_p95 = _measure(
                lambda name=name, parameters=parameters: apply_filter_fast(
                    image, name, **parameters
                )
            )
            entry["operations"][name] = {
                "plain_p50_ms": round(plain_p50, 3),
                "plain_p95_ms": round(plain_p95, 3),
                "fast_p50_ms": round(fast_p50, 3),
                "fast_p95_ms": round(fast_p95, 3),
                "speedup": round(plain_p50 / fast_p50, 3) if fast_p50 else None,
            }
        report["resolutions"][size] = entry
    report["preview_pipeline"] = {
        "plain_p95_ms": round(_pipeline(512, backend="plain"), 3),
        "fast_p95_ms": round(_pipeline(512, backend="fast"), 3),
    }
    session_source = _image(512)
    session_steps = (
        EditStep("grayscale", {"strength": 0.7}),
        EditStep("sepia", {"strength": 0.6}),
        EditStep("brightness_contrast", {"contrast": 0.2}),
    )
    start = time.perf_counter()
    session = PreviewSession(session_source, backend="fast")
    open_ms = (time.perf_counter() - start) * 1000.0
    _, session_p95 = _measure(
        lambda: session.preview(session_steps, max_side=PREVIEW_SIDE), repeats=5
    )
    report["preview_session"] = {
        "backend": session.backend,
        "open_ms": round(open_ms, 3),
        "repeat_p95_ms": round(session_p95, 3),
        "interactive": session_p95 <= INTERACTIVE_BUDGET_MS,
    }
    report["preview_pipeline"]["plain_interactive"] = (
        report["preview_pipeline"]["plain_p95_ms"] <= INTERACTIVE_BUDGET_MS
    )
    report["preview_pipeline"]["fast_interactive"] = (
        report["preview_pipeline"]["fast_p95_ms"] <= INTERACTIVE_BUDGET_MS
    )
    full = _image(512)
    stack = EditStack(full)
    stack.add(EditStep("grayscale", {"strength": 0.7}))
    report["peak_bytes"] = {
        "stack_apply_512": _peak_bytes(stack.apply),
        "preview_128": _peak_bytes(
            lambda: preview_stack(stack, max_side=PREVIEW_SIDE, backend="fast")
        ),
    }
    destination = Path(options.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"numpy available: {report['numpy_available']} backends={report['backends']}")
    for size, entry in report["resolutions"].items():
        for name, metrics in entry["operations"].items():
            print(
                f"{size:4d}px {name:18s} plain={metrics['plain_p50_ms']:8.3f}ms "
                f"fast={metrics['fast_p50_ms']:8.3f}ms speedup={metrics['speedup']}"
            )
    pipeline = report["preview_pipeline"]
    print(
        f"preview pipeline p95 plain={pipeline['plain_p95_ms']}ms "
        f"fast={pipeline['fast_p95_ms']}ms budget={INTERACTIVE_BUDGET_MS}ms "
        f"interactive plain={pipeline['plain_interactive']} fast={pipeline['fast_interactive']}"
    )
    session_report = report["preview_session"]
    print(
        f"preview session open={session_report['open_ms']}ms "
        f"repeat p95={session_report['repeat_p95_ms']}ms "
        f"interactive={session_report['interactive']}"
    )
    print(f"peak bytes {report['peak_bytes']}")
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
