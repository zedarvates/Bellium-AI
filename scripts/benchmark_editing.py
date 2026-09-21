#!/usr/bin/env python3
"""Measure the editing tools: latency per operation and the preservation invariants.

Preview and full-resolution latencies are reported separately, because the
roadmap asks for that split before anything is called interactive. The invariants
are checked on generated cases: the source never changes, uncovered pixels never
move, undo/redo is deterministic and a recipe replays to the same pixels.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bellium.editing import (  # noqa: E402
    EditStack,
    EditStep,
    apply_filter,
    apply_mask,
    feather,
    rect_mask,
)
from bellium.specialists.editing import edit_image, erase_region  # noqa: E402

REPORT = Path(__file__).resolve().parents[1] / "benchmarks" / "manifests" / "editing-v1.json"
RESOLUTIONS = {"preview": 64, "full": 192}
OPS = (
    ("grayscale", {"strength": 0.8}),
    ("sepia", {"strength": 0.6}),
    ("invert", {}),
    ("brightness_contrast", {"brightness": 0.1, "contrast": 0.2}),
    ("saturation", {"factor": 1.4}),
    ("tint", {"factors": (1.2, 1.0, 0.8)}),
    ("threshold", {"level": 0.45}),
)
REPEATS = 5


def _image(size: int, seed: int = 5):
    rng = random.Random(seed)
    return [
        [tuple(rng.randrange(256) for _ in range(3)) for _ in range(size)]
        for _ in range(size)
    ]


def _time(function, repeats: int = REPEATS) -> tuple[float, float]:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1000.0)
    ordered = sorted(samples)
    return statistics.median(samples), ordered[int(0.95 * (len(ordered) - 1))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(REPORT), help="report path")
    options = parser.parse_args()
    report: dict = {
        "schema": "bellium.editing-benchmark/v1",
        "note": (
            "Synthetic images on this workstation. Preview and full-resolution "
            "latencies are separate; no real-time claim is made."
        ),
        "repeats": REPEATS,
        "resolutions": {},
        "invariants": {},
    }
    for label, size in RESOLUTIONS.items():
        image = _image(size)
        per_op = {}
        for name, parameters in OPS:
            p50, p95 = _time(lambda name=name, parameters=parameters: apply_filter(
                image, name, **parameters
            ))
            per_op[name] = {"p50_ms": round(p50, 4), "p95_ms": round(p95, 4)}
        mask = feather(rect_mask((size, size), (size // 4, size // 4, size // 2, size // 2)),
                       radius=3)
        p50, p95 = _time(lambda: apply_mask(image, apply_filter(image, "sepia"), mask))
        per_op["masked_sepia"] = {"p50_ms": round(p50, 4), "p95_ms": round(p95, 4)}
        stack = EditStack(image)
        for name, parameters in OPS[:3]:
            stack.add(EditStep(name, parameters, mask=None))
        p50, p95 = _time(stack.apply, repeats=3)
        per_op["stack_replay"] = {"p50_ms": round(p50, 4), "p95_ms": round(p95, 4)}
        small = rect_mask((size, size), (size // 3, size // 3, max(4, size // 8),
                                         max(4, size // 8)))
        p50, p95 = _time(lambda: erase_region({"image": image, "mask": small}), repeats=3)
        per_op["magic_eraser_preview"] = {"p50_ms": round(p50, 4), "p95_ms": round(p95, 4)}
        report["resolutions"][label] = {"size": size, "operations": per_op}
    size = RESOLUTIONS["preview"]
    source = _image(size)
    before = [row[:] for row in source]
    mask = feather(rect_mask((size, size), (8, 8, 24, 24)), radius=2)
    edited = apply_mask(source, apply_filter(source, "invert"), mask)
    outside_preserved = all(
        edited[r][c] == source[r][c]
        for r in range(size)
        for c in range(size)
        if mask[r][c] == 0.0
    )
    report["invariants"]["source_unchanged_by_filter"] = source == before
    report["invariants"]["uncovered_pixels_preserved"] = outside_preserved
    stack = EditStack(source)
    stack.add(EditStep("grayscale", {"strength": 1.0}))
    stack.add(EditStep("sepia", {"strength": 0.5}))
    first = stack.apply()
    stack.undo()
    stack.redo()
    report["invariants"]["undo_redo_is_deterministic"] = stack.apply() == first
    replayed = EditStack.from_recipe(source, stack.to_recipe()).apply()
    report["invariants"]["recipe_replays_to_the_same_pixels"] = replayed == first
    preview = edit_image({
        "image": source,
        "recipe": [{"name": "grayscale", "parameters": {"strength": 0.5}}],
    })
    report["invariants"]["specialist_keeps_the_source"] = (
        preview.output["source_preserved"] is True and source == before
    )
    destination = Path(options.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for label, data in report["resolutions"].items():
        print(f"{label} ({data['size']}x{data['size']})")
        for name, metrics in data["operations"].items():
            print(f"  {name:22s} p50={metrics['p50_ms']:8.3f}ms p95={metrics['p95_ms']:8.3f}ms")
    for name, value in report["invariants"].items():
        print(f"invariant {name}: {value}")
    print(f"wrote {destination}")
    return 0 if all(report["invariants"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
