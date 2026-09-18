#!/usr/bin/env python3
"""Measure the animation timing tiers on held-out clips.

Clips are generated from the published curves with seeds and hold patterns that
the tests do not use. The report separates clean from noisy clips, because the
published rule degrades with noise and the k-NN tier must abstain instead of
guessing when it cannot confirm the rule.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bellium.knn._motion import REFERENCE_CURVES  # noqa: E402
from bellium.knn.easing_profile import classify_easing  # noqa: E402
from bellium.specialists.animation_timing import report_timing  # noqa: E402

REPORT = Path(__file__).resolve().parents[1] / "benchmarks" / "manifests" / "animation-timing-v1.json"
NOISE_LEVELS = (0.0, 0.02, 0.05)
HOLDS = (0, 1, 2)
RUNS_PER_CURVE = 4


def transitions(curve: list[float], holds_end: int, noise: float, rng: random.Random) -> list[float]:
    series = [
        max(0.0, curve[index] - curve[index - 1]) * 0.1 for index in range(1, len(curve))
    ]
    if noise > 0.0:
        series = [max(0.0, value + rng.gauss(0.0, noise * 0.1)) for value in series]
    return series + [0.0] * holds_end


def clips() -> list[dict]:
    rows = []
    for noise in NOISE_LEVELS:
        for name, curve in sorted(REFERENCE_CURVES.items()):
            for run in range(RUNS_PER_CURVE):
                rng = random.Random(hash((name, noise, run)) & 0xFFFF)
                holds = HOLDS[run % len(HOLDS)]
                series = transitions(curve, holds, noise, rng)
                duplicates = [0.0 if index % 3 == 1 else 0.25 for index in range(len(series))]
                rows.append({
                    "name": f"{name}/noise={noise:.2f}/holds={holds}/run={run}",
                    "curve": name,
                    "noise": noise,
                    "holds": holds,
                    "series": series,
                    "frame_mismatches": duplicates,
                })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(REPORT), help="report path")
    options = parser.parse_args()
    summary: dict = {
        "schema": "bellium.animation-timing-benchmark/v1",
        "note": (
            "Held-out clips built from the published curves with unseen seeds and hold "
            "patterns. Abstention is a valid outcome and is reported per noise level."
        ),
        "levels": {},
        "clips": [],
    }
    for clip in clips():
        easing, easing_ms = _timed(lambda: classify_easing(
            {"family": "ui", "series": clip["series"]}
        ))
        timing, timing_ms = _timed(lambda: report_timing({
            "family": "ui",
            "series": clip["series"],
            "frame_mismatches": clip["frame_mismatches"],
        }))
        expected_duplicates = sum(1 for value in clip["frame_mismatches"] if value <= 0.02)
        found_duplicates = len(timing.output.get("duplicates", []))
        record = {
            "name": clip["name"],
            "curve": clip["curve"],
            "noise": clip["noise"],
            "holds": clip["holds"],
            "easing_status": easing.output["status"],
            "easing": easing.output.get("easing"),
            "easing_hit": easing.output.get("easing") == clip["curve"],
            "baseline": easing.output.get("baseline"),
            "rule_hit": easing.output.get("baseline") == clip["curve"],
            "expected_duplicates": expected_duplicates,
            "found_duplicates": found_duplicates,
            "holds_found": [run["length"] for run in timing.output.get("holds", [])],
            "expected_holds": clip["holds"],
            "easing_ms": easing_ms,
            "timing_ms": timing_ms,
        }
        summary["clips"].append(record)
        level = summary["levels"].setdefault(f"{clip['noise']:.2f}", {
            "clips": 0, "easing_hits": 0, "abstentions": 0, "rule_hits": 0,
            "duplicates_found": 0, "duplicates_expected": 0,
            "holds_matched": 0, "latencies": [],
        })
        level["clips"] += 1
        level["easing_hits"] += int(record["easing_hit"])
        level["abstentions"] += int(record["easing_status"] == "abstain")
        level["rule_hits"] += int(record["rule_hit"])
        level["duplicates_found"] += found_duplicates
        level["duplicates_expected"] += expected_duplicates
        level["holds_matched"] += int(sum(record["holds_found"]) == clip["holds"])
        level["latencies"].append(timing_ms)
    for level in summary["levels"].values():
        latencies = level.pop("latencies")
        level["p50_ms_timing"] = round(statistics.median(latencies), 4)
    destination = Path(options.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for name, level in summary["levels"].items():
        print(
            f"noise={name} clips={level['clips']} easing_hits={level['easing_hits']} "
            f"rule_hits={level['rule_hits']} abstentions={level['abstentions']} "
            f"duplicates={level['duplicates_found']}/{level['duplicates_expected']} "
            f"holds_matched={level['holds_matched']} p50={level['p50_ms_timing']}ms"
        )
    print(f"wrote {destination}")
    return 0


def _timed(function):
    import time

    start = time.perf_counter()
    result = function()
    return result, (time.perf_counter() - start) * 1000.0


if __name__ == "__main__":
    raise SystemExit(main())
