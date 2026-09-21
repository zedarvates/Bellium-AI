#!/usr/bin/env python3
"""P8 first gate: repeat, tileability and grain on held-out distorted textures.

Every case here uses parameters that appear neither in the shipped memories nor
in their builders. The measurement reports, per distortion class, whether a
period was found where one exists, whether one was invented where none exists,
and whether a perspective distortion was correctly refused instead of being
reported as one global period. Nothing is promoted by the result.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bellium.knn.texture_orientation import classify_orientation  # noqa: E402
from bellium.knn.texture_repeat import classify_repeat  # noqa: E402
from bellium.knn.tileability import classify_tileability  # noqa: E402

REPORT = Path(__file__).resolve().parents[1] / "benchmarks" / "manifests" / "texture-repeat-v1.json"
SIZE = 64
FAMILY = "texture"
ANGLE_TOLERANCE_DEG = 7.0

Image = list[list[tuple[int, int, int]]]


def gray(value: float) -> tuple[int, int, int]:
    level = max(0, min(255, int(value)))
    return level, level, level


def sine(period: float, angle_deg: float = 0.0, amplitude: float = 90.0) -> Image:
    radians = math.radians(angle_deg)
    return [
        [
            gray(128 + amplitude * math.sin(2 * math.pi * (c * math.cos(radians)
                                                          + r * math.sin(radians)) / period))
            for c in range(SIZE)
        ]
        for r in range(SIZE)
    ]


def tiled_noise(tile: int, seed: int) -> Image:
    rng = random.Random(seed)
    block = [[gray(rng.randrange(256)) for _ in range(tile)] for _ in range(tile)]
    return [[block[r % tile][c % tile] for c in range(SIZE)] for r in range(SIZE)]


def noise(seed: int) -> Image:
    rng = random.Random(seed)
    return [[gray(rng.randrange(256)) for _ in range(SIZE)] for _ in range(SIZE)]


def ramp(seed: int) -> Image:
    rng = random.Random(seed)
    shift = rng.randrange(20, 60)
    return [[gray(shift + (200 - shift) * (c / SIZE)) for c in range(SIZE)] for _ in range(SIZE)]


def wrapped_step(step: float) -> Image:
    return [
        [gray(90 + (step / 2 if c < SIZE // 2 else -step / 2)
              + 6 * math.sin(2 * math.pi * c / 16))
         for c in range(SIZE)]
        for _ in range(SIZE)
    ]


def warp(image: Image, strength: float, *, vertical: bool = False) -> Image:
    """Progressive scale across one axis: the period changes from row to row."""
    out = []
    for r in range(SIZE):
        progress = (r / SIZE) if vertical else 1.0
        row = []
        for c in range(SIZE):
            horizontal = 1.0 + strength * (r / SIZE)
            if vertical:
                source_r = min(max(int(round(r / (1.0 + strength * (c / SIZE)))), 0), SIZE - 1)
                row.append(image[source_r][c])
                continue
            source_c = min(max(int(round(c / horizontal)), 0), SIZE - 1)
            row.append(image[r][source_c])
        out.append(row)
        _ = progress
    return out


def cases() -> list[dict]:
    rows: list[dict] = []
    for period in (11, 13, 17):
        rows.append({
            "name": f"periodic sine p={period}",
            "class": "periodic",
            "image": sine(period),
            "period": period,
            "grain": "repeat-x",
            "angle": 0.0,
            "tileable": SIZE % period == 0,
        })
    for tile in (12, 20, 24):
        rows.append({
            "name": f"periodic tile t={tile}",
            "class": "periodic",
            "image": tiled_noise(tile, seed=200 + tile),
            "period": tile if tile * (SIZE // tile) == SIZE else None,
            "grain": "repeat-both",
            "angle": None,
            "tileable": SIZE % tile == 0,
        })
    for seed in (101, 102, 103):
        rows.append({
            "name": f"nonperiodic noise {seed}",
            "class": "nonperiodic",
            "image": noise(seed),
            "period": None,
            "grain": "no-repeat",
            "angle": None,
            "tileable": None,
        })
    rows.append({
        "name": "nonperiodic ramp",
        "class": "nonperiodic",
        "image": ramp(104),
        "period": None,
        "grain": "no-repeat",
        "angle": None,
        "tileable": None,
    })
    rows.append({
        "name": "nonperiodic wrapped step",
        "class": "nonperiodic",
        "image": wrapped_step(130.0),
        "period": None,
        "grain": "no-repeat",
        "angle": None,
        "tileable": False,
    })
    for angle in (20.0, 35.0, 50.0, 70.0):
        rows.append({
            "name": f"rotated {angle:.0f} deg",
            "class": "rotated",
            "image": sine(12, angle_deg=angle),
            "period": 12,
            "grain": "repeat-diagonal",
            "angle": angle,
            "tileable": None,
        })
    for strength in (0.20, 0.35, 0.50):
        rows.append({
            "name": f"perspective {strength:.2f}",
            "class": "perspective",
            "image": warp(sine(12), strength),
            "period": None,
            "grain": None,
            "angle": None,
            "tileable": None,
        })
    return rows


def _timed(function):
    start = time.perf_counter()
    result = function()
    return result, (time.perf_counter() - start) * 1000.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(REPORT), help="report path")
    options = parser.parse_args()
    summary = {
        "schema": "bellium.texture-repeat-benchmark/v1",
        "note": (
            "Held-out synthetic cases: parameters differ from the shipped memories and "
            "their builders. Distortion classes are reported separately and nothing is "
            "promoted by this measurement."
        ),
        "size": SIZE,
        "angle_tolerance_deg": ANGLE_TOLERANCE_DEG,
        "classes": {},
        "cases": [],
    }
    for case in cases():
        image = case["image"]
        repeat, repeat_ms = _timed(lambda: classify_repeat({"family": FAMILY, "image": image}))
        tile, tile_ms = _timed(lambda: classify_tileability({"family": FAMILY, "image": image}))
        orientation, orientation_ms = _timed(
            lambda: classify_orientation({"family": FAMILY, "image": image})
        )
        record = {
            "name": case["name"],
            "class": case["class"],
            "expected_period": case["period"],
            "expected_grain": case["grain"],
            "expected_angle": case["angle"],
            "expected_tileable": case["tileable"],
            "repeat_status": repeat.output["status"],
            "repeat_verdict": repeat.output.get("verdict"),
            "repeat_period_x": repeat.output.get("period_x"),
            "repeat_period_y": repeat.output.get("period_y"),
            "tileability_status": tile.output["status"],
            "tileability_verdict": tile.output.get("verdict"),
            "orientation_status": orientation.output["status"],
            "orientation_grain": orientation.output.get("grain"),
            "orientation_angle": orientation.output.get("angle_deg"),
            "orientation_reason": orientation.output.get("reason"),
            "latency_ms": {
                "repeat": repeat_ms,
                "tileability": tile_ms,
                "orientation": orientation_ms,
            },
        }
        record["period_found"] = record["repeat_verdict"] == "periodic"
        record["period_invented"] = (
            case["period"] is None and record["repeat_verdict"] == "periodic"
        )
        record["period_exact"] = bool(
            case["period"] is not None
            and record["repeat_period_x"] == case["period"]
        )
        record["grain_hit"] = record["orientation_grain"] == case["grain"]
        record["angle_error"] = (
            None if case["angle"] is None or record["orientation_angle"] is None
            else abs(record["orientation_angle"] - case["angle"])
        )
        record["perspective_refused"] = (
            record["orientation_status"] == "abstain"
            and record["orientation_reason"] == "repeat_varies_across_image"
        )
        summary["cases"].append(record)
        bucket = summary["classes"].setdefault(case["class"], {
            "cases": 0, "period_found": 0, "period_exact": 0, "period_invented": 0,
            "grain_hits": 0, "angle_errors": [], "perspective_refused": 0,
            "latencies": {"repeat": [], "tileability": [], "orientation": []},
        })
        bucket["cases"] += 1
        bucket["period_found"] += int(record["period_found"])
        bucket["period_exact"] += int(record["period_exact"])
        bucket["period_invented"] += int(record["period_invented"])
        bucket["grain_hits"] += int(record["grain_hit"])
        bucket["perspective_refused"] += int(record["perspective_refused"])
        if record["angle_error"] is not None:
            bucket["angle_errors"].append(record["angle_error"])
        for key, value in record["latency_ms"].items():
            bucket["latencies"][key].append(value)
    for bucket in summary["classes"].values():
        latencies = bucket.pop("latencies")
        errors = bucket.pop("angle_errors")
        bucket["mean_angle_error_deg"] = (
            round(statistics.mean(errors), 3) if errors else None
        )
        bucket["max_angle_error_deg"] = round(max(errors), 3) if errors else None
        for key, values in latencies.items():
            bucket[f"p50_ms_{key}"] = round(statistics.median(values), 4)
    destination = Path(options.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for name, bucket in summary["classes"].items():
        print(
            f"{name:12s} cases={bucket['cases']} period_found={bucket['period_found']} "
            f"exact={bucket['period_exact']} invented={bucket['period_invented']} "
            f"grain_hits={bucket['grain_hits']} angle_err_max={bucket['max_angle_error_deg']} "
            f"perspective_refused={bucket['perspective_refused']}"
        )
    print(f"wrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
