#!/usr/bin/env python3
"""Measure atlas rasterization: exact pixels, PNG bytes and verification cost.

Every case is authored geometry with generated sprites, so the run is reproducible.
The report separates raster bytes from encoded bytes, re-checks every page, and
cross-checks the encoded files with Pillow when it is importable.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bellium.atlas.packing import pack, parse_frames  # noqa: E402
from bellium.atlas.png import decode_png, encode_pages, inspect_png  # noqa: E402
from bellium.atlas.raster import check_raster, rasterize  # noqa: E402

REPORT = Path(__file__).resolve().parents[1] / "benchmarks" / "manifests" / "atlas-export-v1.json"
REPEATS = 5


def sprite(width: int, height: int, seed: int, *, alpha: bool) -> list[list[tuple[int, ...]]]:
    """A generated blob: an ellipse of colour over a transparent or black field."""
    rows = []
    radius_x = width / 2.0
    radius_y = height / 2.0
    for row in range(height):
        line: list[tuple[int, ...]] = []
        for column in range(width):
            dx = (column + 0.5 - radius_x) / radius_x
            dy = (row + 0.5 - radius_y) / radius_y
            inside = dx * dx + dy * dy <= 1.0
            noise = (row * 31 + column * 17 + seed * 13) % 24
            red = min(255, 40 + int(160 * (column / max(1, width - 1))) + noise)
            green = min(255, 30 + int(140 * (row / max(1, height - 1))) + noise)
            blue = min(255, 90 + (seed * 37 + row * 5 + column * 3) % 120)
            if alpha:
                line.append((red, green, blue, 255 if inside else 0))
            else:
                line.append((red if inside else 0, green if inside else 0, blue if inside else 0))
        rows.append(line)
    return rows


def case_frames(spec: list[tuple[str, int, int]], *, alpha: bool):
    frames = parse_frames([{"name": name, "width": width, "height": height}
                           for name, width, height in spec])
    sources = {frame.name: sprite(frame.width, frame.height, index, alpha=alpha)
               for index, frame in enumerate(frames)}
    return frames, sources


def cases() -> list[dict]:
    measured = [
        ("f_0", 16, 12), ("f_1", 32, 24), ("f_2", 8, 40), ("f_3", 24, 16), ("f_4", 32, 40),
        ("f_5", 8, 32), ("f_6", 40, 16), ("f_7", 24, 16), ("f_8", 12, 16), ("f_9", 40, 12),
        ("f_10", 24, 24), ("f_11", 40, 24), ("f_12", 16, 32), ("f_13", 12, 24),
    ]
    mixed = [
        ("idle_0", 24, 32), ("idle_1", 24, 32), ("run_0", 28, 24), ("run_1", 28, 24),
        ("jump_0", 20, 40), ("jump_1", 20, 40), ("hit_0", 36, 36), ("fall_0", 16, 48),
        ("fall_1", 16, 48), ("fx_0", 48, 12), ("fx_1", 48, 12), ("icon_0", 12, 12),
    ]
    return [
        {"name": "uniform_16_rgba", "frames": [(f"u_{i}", 16, 16) for i in range(64)],
         "alpha": True, "width": 128, "height": 128, "max_pages": 1},
        {"name": "mixed_rgb", "frames": mixed, "alpha": False, "width": 128, "height": 128,
         "max_pages": 1},
        {"name": "mixed_padded_rgba", "frames": mixed, "alpha": True, "padding": 2,
         "width": 192, "height": 192, "max_pages": 1},
        {"name": "measured_96_rgba", "frames": measured, "alpha": True, "width": 96, "height": 96,
         "max_pages": 1},
        {"name": "two_pages_rgb", "frames": [(f"p_{i}", 32, 32) for i in range(24)],
         "alpha": False, "width": 128, "height": 128, "max_pages": 3},
    ]


def pillow_mismatches(pages, encoded) -> tuple[int, bool]:
    if importlib.util.find_spec("PIL") is None:
        return 0, False
    from PIL import Image
    import io

    mismatches = 0
    for page, data in zip(pages, encoded):
        image = Image.open(io.BytesIO(data))
        for row in range(page.height):
            for column in range(page.width):
                if tuple(image.getpixel((column, row)))[:page.channels] != page.pixels[row][column]:
                    mismatches += 1
    return mismatches, True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default=str(REPORT), help="report path")
    parser.add_argument("--repeats", type=int, default=REPEATS, help="timed repetitions")
    options = parser.parse_args()
    report: dict = {
        "schema": "bellium.atlas-export-benchmark/v1",
        "note": (
            "Generated sprites, no external assets. Encoded pages use filter 0, one fixed zlib "
            "level and no interlacing: a tuned encoder would produce smaller files. No engine "
            "was run and nothing was written to disk."
        ),
        "repeats": options.repeats,
        "cases": [],
    }
    raster_ms: list[float] = []
    encode_ms: list[float] = []
    verify_ms: list[float] = []
    violations_total = 0
    mismatch_total = 0
    print(f"{'case':<20}{'pages':>6}{'px':>10}{'raw':>10}{'png':>10}{'ratio':>7}"
          f"{'raster':>9}{'encode':>9}{'verify':>9}")
    for case in cases():
        frames, sources = case_frames(case["frames"], alpha=case["alpha"])
        plan = pack(frames, width=case["width"], height=case["height"],
                    padding=case.get("padding", 0), max_pages=case["max_pages"])
        durations = {"raster": [], "encode": [], "verify": []}
        raster = None
        encoded: tuple[bytes, ...] = ()
        for _ in range(max(1, options.repeats)):
            start = time.perf_counter()
            raster = rasterize(frames, plan, sources)
            durations["raster"].append((time.perf_counter() - start) * 1000.0)
            start = time.perf_counter()
            encoded = encode_pages(raster)
            durations["encode"].append((time.perf_counter() - start) * 1000.0)
            start = time.perf_counter()
            for index, data in enumerate(encoded):
                info = inspect_png(data)
                if not info.ok or decode_png(data).digest != raster.pages[index].digest:
                    violations_total += 1
            durations["verify"].append((time.perf_counter() - start) * 1000.0)
        assert raster is not None
        violations = check_raster(frames, plan, raster, sources)
        violations_total += len(violations)
        mismatches, pillow_used = pillow_mismatches(raster.pages, encoded)
        mismatch_total += mismatches
        raw = raster.bytes_written
        png_bytes = sum(len(data) for data in encoded)
        pixels = sum(page.width * page.height for page in raster.pages)
        for key in durations:
            (raster_ms if key == "raster" else encode_ms if key == "encode" else verify_ms).append(
                statistics.median(durations[key])
            )
        report["cases"].append({
            "name": case["name"],
            "frames": len(frames),
            "channels": raster.channels,
            "pages": raster.page_count,
            "pixels": pixels,
            "raw_bytes": raw,
            "png_bytes": png_bytes,
            "png_ratio": round(png_bytes / raw, 4) if raw else 0.0,
            "digests": [page.digest for page in raster.pages],
            "page_sha256": [hashlib.sha256(data).hexdigest() for data in encoded],
            "violations": violations,
            "pillow_mismatches": mismatches,
            "pillow_used": pillow_used,
            "ms_p50": {key: round(statistics.median(value), 4) for key, value in durations.items()},
        })
        entry = report["cases"][-1]
        print(f"{case['name']:<20}{raster.page_count:>6}{pixels:>10}{raw:>10}{png_bytes:>10}"
              f"{entry['png_ratio']:>7.3f}{entry['ms_p50']['raster']:>9.3f}"
              f"{entry['ms_p50']['encode']:>9.3f}{entry['ms_p50']['verify']:>9.3f}")
    report["summary"] = {
        "cases": len(report["cases"]),
        "raw_bytes": sum(case["raw_bytes"] for case in report["cases"]),
        "png_bytes": sum(case["png_bytes"] for case in report["cases"]),
        "pillow_used": any(case["pillow_used"] for case in report["cases"]),
        "violations": violations_total,
        "pillow_mismatches": mismatch_total,
        "ms_p50_raster": round(statistics.median(raster_ms), 4),
        "ms_p50_encode": round(statistics.median(encode_ms), 4),
        "ms_p50_verify": round(statistics.median(verify_ms), 4),
    }
    report["summary"]["png_ratio"] = (
        round(report["summary"]["png_bytes"] / report["summary"]["raw_bytes"], 4)
        if report["summary"]["raw_bytes"] else 0.0
    )
    target = Path(options.report)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("summary:", json.dumps(report["summary"]))
    print("report:", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

