#!/usr/bin/env python3
"""Measure exact duplicate removal: bytes saved, pages saved, pixels redrawn.

Cases are generated with a known duplicate structure, including a control with no
duplicates and a set of near misses that must not merge. Every declared frame is
redrawn from the stored atlas through its alias and compared with its source; the
report counts mismatches instead of assuming none.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bellium.atlas.dedup import (  # noqa: E402
    analyze_frames,
    apply_transform,
    check_aliases,
    pixel_delta,
)
from bellium.atlas.packing import pack, parse_frames  # noqa: E402
from bellium.atlas.png import decode_png, inspect_png  # noqa: E402
from bellium.specialists.atlas_dedup import dedup_atlas  # noqa: E402

REPORT = Path(__file__).resolve().parents[1] / "benchmarks" / "manifests" / "atlas-dedup-v1.json"
REPEATS = 5


def sprite(width: int, height: int, seed: int) -> list[list[tuple[int, int, int, int]]]:
    return [
        [
            ((row * 9 + column * 5 + seed * 31) % 200,
             (row * 3 + column * 11 + seed * 17) % 200,
             (seed * 53 + row + column) % 256,
             255 if (row + column + seed) % 4 else 0)
            for column in range(width)
        ]
        for row in range(height)
    ]


def cases() -> list[dict]:
    """Each case declares its frames as (name, width, height, pose)."""
    walk = [(f"walk_{i}", 16, 16, i % 4) for i in range(12)]
    idle = [(f"idle_{i}", 12, 12, 0 if i < 6 else 1 + (i % 2)) for i in range(8)]
    distinct = [(f"run_{i}", 16, 16, i) for i in range(12)]
    near = [(f"near_{i}", 16, 16, i) for i in range(12)]
    mixed = [(f"small_{i}", 12, 12, i % 2) for i in range(6)]
    mixed += [(f"big_{i}", 20, 24, i % 2) for i in range(6)]
    pressure = [(f"pose_{i}", 32, 32, i % 4) for i in range(16)]
    jittered = [(f"draw_{i}", 16, 16, i % 4) for i in range(12)]
    mirrored = [(f"mirror_{i}", 16, 16, i % 4) for i in range(12)]
    return [
        {"name": "walk_duplicates", "frames": walk, "width": 64, "height": 64, "max_pages": 2},
        {"name": "idle_repeats", "frames": idle, "width": 48, "height": 48, "max_pages": 2},
        {"name": "no_duplicates", "frames": distinct, "width": 128, "height": 128, "max_pages": 2},
        {"name": "near_misses", "frames": near, "width": 128, "height": 128, "max_pages": 2,
         "nudge": True},
        {"name": "mixed_sizes", "frames": mixed, "width": 96, "height": 96, "max_pages": 3},
        {"name": "page_pressure", "frames": pressure, "width": 64, "height": 64,
         "max_pages": 2},
        {"name": "jitter_within_bound", "frames": jittered, "width": 64, "height": 64,
         "max_pages": 2, "jitter": 2, "max_delta": 3},
        {"name": "jitter_beyond_bound", "frames": jittered, "width": 64, "height": 64,
         "max_pages": 2, "jitter": 2, "max_delta": 1},
        {"name": "mirrored_walk", "frames": mirrored, "width": 64, "height": 64,
         "max_pages": 2, "mirror": True},
    ]


def sources_for(case: dict) -> dict:
    sources = {}
    for name, width, height, pose in case["frames"]:
        image = sprite(width, height, pose)
        if case.get("nudge"):
            index = int(name.split("_")[1])
            image = [row[:] for row in image]
            image[0][0] = (image[0][0][0], image[0][0][1], image[0][0][2], index % 256)
        if case.get("jitter"):
            amount = case["jitter"]
            index = int(name.split("_")[1])
            image = [[tuple(pixel) for pixel in row] for row in image]
            for step in range(2 + index % 3):
                row = (index + step) % height
                column = (index * 3 + step) % width
                pixel = list(image[row][column])
                pixel[1] = max(0, min(255, pixel[1] + (amount if step % 2 else -amount)))
                pixel[2] = max(0, min(255, pixel[2] + amount))
                image[row][column] = tuple(pixel)
        if case.get("mirror"):
            index = int(name.split("_")[1])
            image = apply_transform(image, ("none", "flip-x", "flip-y")[index % 3])
        sources[name] = image
    return sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(REPORT), help="report path")
    parser.add_argument("--repeats", type=int, default=REPEATS, help="timed repetitions")
    options = parser.parse_args()
    report: dict = {
        "schema": "bellium.atlas-dedup-benchmark/v1",
        "note": (
            "Generated sprites with a known duplicate structure. Near misses differ by one "
            "channel value and must not merge. Nothing is written to disk and no engine runs."
        ),
        "repeats": options.repeats,
        "cases": [],
    }
    mismatches_total = 0
    violations_total = 0
    analyse_ms: list[float] = []
    pipeline_ms: list[float] = []
    print(f"{'case':<18}{'declared':>9}{'stored':>7}{'saved':>8}{'ratio':>7}"
          f"{'pages':>11}{'png':>9}{'worst':>7}{'differ':>8}{'exact':>7}")
    for case in cases():
        sources = sources_for(case)
        frames = parse_frames([{"name": name, "width": width, "height": height}
                               for name, width, height, _ in case["frames"]])
        starts = []
        for _ in range(max(1, options.repeats)):
            start = time.perf_counter()
            analyze_frames(frames, sources)
            starts.append((time.perf_counter() - start) * 1000.0)
        entry_analyse = statistics.median(starts)
        analyse_ms.append(entry_analyse)
        query = {
            "frames": [{"name": name, "width": width, "height": height}
                       for name, width, height, _ in case["frames"]],
            "sources": sources, "width": case["width"], "height": case["height"],
            "max_pages": case["max_pages"], "encode": "png",
            "max_delta": case.get("max_delta", 0),
        }
        durations = []
        result = None
        for _ in range(max(1, options.repeats)):
            start = time.perf_counter()
            result = dedup_atlas(query)
            durations.append((time.perf_counter() - start) * 1000.0)
        assert result is not None
        pipeline_ms.append(statistics.median(durations))
        output = result.output
        exact_saved = dedup_atlas({**query, "max_delta": 0}).output.get("saved_bytes")
        report_obj = analyze_frames(frames, sources)
        violations_total += len(check_aliases(frames, sources, report_obj))
        pages_all = pack(frames, width=case["width"], height=case["height"],
                         max_pages=case["max_pages"])
        worst_drawn = 0
        differing = 0
        mismatches = 0
        bound_violations = 0
        if output["status"] == "ready":
            regions = {item["name"]: item for item in output["placements"]}
            pages = output["pages"] if "pages" in output else None
            decoded = [decode_png(data) for data in output["page_bytes"]]
            aliases = {item["name"]: item["canonical"] for item in output["aliases"]}
            transforms = {item["name"]: item["transform"] for item in output["aliases"]}
            for name, source in sources.items():
                region = regions[aliases[name]]
                page = decoded[region["page"]]
                stored = [
                    [tuple(page.pixels[region["y"] + row][region["x"] + column])
                     for column in range(region["width"])]
                    for row in range(region["height"])
                ]
                drawn = apply_transform(stored, transforms[name])
                for row in range(len(source)):
                    for column in range(len(source[0])):
                        difference = pixel_delta(drawn[row][column], source[row][column])
                        if difference:
                            mismatches += 1
                            differing += 1
                        worst_drawn = max(worst_drawn, difference)
            if worst_drawn > output["max_delta"]:
                bound_violations += 1
            assert pages is None
            for index, data in enumerate(output["page_bytes"]):
                info = inspect_png(data)
                if not info.ok or decode_png(data).digest != output["page_digests"][index]:
                    violations_total += 1
        mismatches_total += bound_violations
        entry = {
            "name": case["name"],
            "status": output["status"],
            "reason": output["reason"],
            "frames_declared": len(frames),
            "frames_stored": output.get("frames_stored"),
            "duplicate_groups": len(output.get("groups") or []),
            "original_bytes": output.get("original_bytes"),
            "stored_bytes": output.get("stored_bytes"),
            "saved_bytes": output.get("saved_bytes"),
            "saved_ratio": output.get("saved_ratio"),
            "pages_before": pages_all.page_count,
            "complete_before": pages_all.complete,
            "pages_stored": output.get("page_count"),
            "raw_bytes_stored": output.get("raw_bytes"),
            "png_bytes": sum(item["bytes"] for item in output.get("png") or []),
            "max_delta": output.get("max_delta"),
            "worst_delta": output.get("worst_delta"),
            "saved_bytes_exact": exact_saved,
            "worst_drawn_delta": worst_drawn,
            "differing_pixels": differing,
            "drawn_mismatches": mismatches,
            "transforms": output.get("transforms"),
            "bound_violations": bound_violations,
            "ms_analyse": round(entry_analyse, 4),
            "ms_pipeline": round(statistics.median(durations), 4),
        }
        report["cases"].append(entry)
        print(f"{case['name']:<18}{len(frames):>9}{entry['frames_stored']:>7}"
              f"{entry['saved_bytes']:>8}{entry['saved_ratio']:>7.3f}"
              f"{str(entry['pages_before']) + '->' + str(entry['pages_stored']):>11}"
              f"{entry['png_bytes']:>9}{entry['worst_drawn_delta']:>7}{differing:>8}"
              f"{mismatches:>7}")
    report["summary"] = {
        "cases": len(report["cases"]),
        "declared_frames": sum(case["frames_declared"] for case in report["cases"]),
        "stored_frames": sum(case["frames_stored"] or 0 for case in report["cases"]),
        "saved_bytes": sum(case["saved_bytes"] or 0 for case in report["cases"]),
        "bound_violations": mismatches_total,
        "worst_drawn_delta": max(case["worst_drawn_delta"] for case in report["cases"]),
        "differing_pixels": sum(case["differing_pixels"] for case in report["cases"]),
        "drawn_mismatches": sum(case["drawn_mismatches"] for case in report["cases"]),
        "violations": violations_total,
        "ms_p50_analyse": round(statistics.median(analyse_ms), 4),
        "ms_p50_pipeline": round(statistics.median(pipeline_ms), 4),
    }
    target = Path(options.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("summary:", json.dumps(report["summary"]))
    print("report:", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
