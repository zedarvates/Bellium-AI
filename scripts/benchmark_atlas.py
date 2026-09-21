#!/usr/bin/env python3
"""Measure atlas packing: invariants, page occupancy and latency per case.

Every case is authored geometry, so the report is reproducible without assets.
Three methods run on the same frames: the shipped skyline packer and the two
published baselines (insertion-order shelf and height-sorted shelf). Ties and
losses are reported per case; the summary never claims a uniform win.
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

from bellium.atlas.packing import METHODS, check_plan, pack, parse_frames  # noqa: E402
from bellium.specialists.atlas import pack_atlas  # noqa: E402

REPORT = Path(__file__).resolve().parents[1] / "benchmarks" / "manifests" / "atlas-packing-v1.json"
REPEATS = 5
MEASURED_CASE = (
    ("f_0", 16, 12), ("f_1", 32, 24), ("f_2", 8, 40), ("f_3", 24, 16), ("f_4", 32, 40),
    ("f_5", 8, 32), ("f_6", 40, 16), ("f_7", 24, 16), ("f_8", 12, 16), ("f_9", 40, 12),
    ("f_10", 24, 24), ("f_11", 40, 24), ("f_12", 16, 32), ("f_13", 12, 24),
)


def _seeded(prefix: str, count: int, sizes: tuple[int, ...], seed: int) -> list[tuple[str, int, int]]:
    rng = random.Random(seed)
    return [
        (f"{prefix}_{index}", rng.choice(sizes), rng.choice(sizes))
        for index in range(count)
    ]


def cases() -> list[dict]:
    uniform_32 = [(f"u32_{index}", 32, 32) for index in range(64)]
    uniform_16 = [(f"u16_{index}", 16, 16) for index in range(256)]
    many_small = [(f"small_{index}", 8, 8) for index in range(400)]
    tall_wide = [
        (f"tw_{index}", 16, 96) if index % 2 else (f"tw_{index}", 96, 16)
        for index in range(40)
    ]
    near_full = [(f"nf_{index}", 24, 24) for index in range(100)]
    mixed = _seeded("mix", 60, (12, 16, 20, 24, 32, 40, 48, 64), seed=11)
    return [
        {"name": "uniform_32", "frames": uniform_32, "width": 256, "height": 256, "max_pages": 1},
        {"name": "uniform_16", "frames": uniform_16, "width": 256, "height": 256, "max_pages": 1},
        {"name": "many_small", "frames": many_small, "width": 128, "height": 128, "max_pages": 1},
        {"name": "mixed_sprite", "frames": mixed, "width": 256, "height": 256, "max_pages": 1},
        {"name": "mixed_sprite_padded", "frames": mixed, "width": 256, "height": 256,
         "max_pages": 1, "padding": 2},
        {"name": "tall_and_wide", "frames": tall_wide, "width": 256, "height": 256,
         "max_pages": 1},
        {"name": "near_full_24", "frames": near_full, "width": 256, "height": 256,
         "max_pages": 1},
        {"name": "measured_96", "frames": list(MEASURED_CASE), "width": 96, "height": 96,
         "max_pages": 1},
        {"name": "three_pages", "frames": [(f"p_{index}", 32, 32) for index in range(200)],
         "width": 128, "height": 128, "max_pages": 3},
        {"name": "one_oversized", "frames": [("huge", 512, 512), ("ok", 16, 16)],
         "width": 256, "height": 256, "max_pages": 1},
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(REPORT), help="report path")
    parser.add_argument("--repeats", type=int, default=REPEATS, help="timed repetitions per method")
    options = parser.parse_args()
    report: dict = {
        "schema": "bellium.atlas-packing-benchmark/v1",
        "note": (
            "Authored geometry, no assets. The three methods share the same frames; "
            "an incomplete plan is a valid measured outcome and is reported as such."
        ),
        "repeats": options.repeats,
        "cases": [],
    }
    verdicts: dict[str, dict] = {method: {"wins": 0, "ties": 0, "losses": 0} for method in METHODS}
    violations_total = 0
    print(f"{'case':<22}{'method':<10}{'pages':>6}{'occ':>8}{'ok':>4}{'unp':>5}{'ms':>9}")
    for case in cases():
        frames = parse_frames([
            {"name": name, "width": width, "height": height}
            for name, width, height in case["frames"]
        ])
        entry = {"name": case["name"], "frame_count": len(frames),
                 "page_width": case["width"], "page_height": case["height"],
                 "max_pages": case["max_pages"], "padding": case.get("padding", 0), "methods": {}}
        for method in METHODS:
            durations = []
            plan = None
            for _ in range(max(1, options.repeats)):
                start = time.perf_counter()
                plan = pack(
                    frames, width=case["width"], height=case["height"], method=method,
                    padding=case.get("padding", 0), max_pages=case["max_pages"],
                )
                durations.append((time.perf_counter() - start) * 1000.0)
            assert plan is not None
            violations = check_plan(frames, plan)
            violations_total += len(violations)
            entry["methods"][method] = {
                "complete": plan.complete,
                "pages": plan.page_count,
                "occupancy": round(plan.occupancy, 4),
                "unplaced": len(plan.unplaced),
                "violations": violations,
                "ms_p50": round(statistics.median(durations), 4),
            }
            print(f"{case['name']:<22}{method:<10}{plan.page_count:>6}"
                  f"{plan.occupancy:>8.4f}{'yes' if plan.complete else 'no':>4}"
                  f"{len(plan.unplaced):>5}{statistics.median(durations):>9.3f}")
        report["cases"].append(entry)
    for case in report["cases"]:
        shipped = case["methods"]["skyline"]
        for method in ("shelf", "next-fit"):
            entry = case["methods"][method]
            if entry["complete"] != shipped["complete"]:
                counted = "wins" if entry["complete"] else "losses"
                verdicts[method][counted] += 1
            elif abs(entry["occupancy"] - shipped["occupancy"]) < 1e-9:
                verdicts[method]["ties"] += 1
            elif entry["occupancy"] > shipped["occupancy"]:
                verdicts[method]["wins"] += 1
            else:
                verdicts[method]["losses"] += 1
    baseline = {
        "shelf": verdicts["shelf"],
        "next-fit": verdicts["next-fit"],
        "violations": violations_total,
        "note": (
            "Each baseline is counted against the shipped skyline packer: wins, ties and "
            "losses on occupancy when both place everything, otherwise the differing "
            "completeness verdict decides."
        ),
    }
    report["baselines"] = baseline
    specialist = pack_atlas({
        "frames": [{"name": name, "width": width, "height": height}
                   for name, width, height in MEASURED_CASE],
        "width": 96, "height": 96, "power_of_two": True,
    })
    report["specialist_check"] = {
        "status": specialist.output["status"],
        "page_width": specialist.output["page_width"],
        "occupancy": specialist.output["occupancy"],
        "certified": specialist.output["certified"],
        "pixels_written": specialist.output["pixels_written"],
    }
    target = Path(options.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"violations across every case: {violations_total}")
    print("baselines:", json.dumps(baseline))
    print("report:", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
