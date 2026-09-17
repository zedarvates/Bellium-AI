"""Compare a pinned spatial-copy baseline with bounded context patch matching.

Original procedural RGB fixtures are evaluation cases, not real asset evidence.
Run: python -m benchmarks.inpaint_context --output new-report.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import time
import tracemalloc
from pathlib import Path
from unittest.mock import patch

import PIL
from PIL import Image, ImageDraw

from bellium.inpaint import inpaint_patch_knn
from benchmarks.inpaint_v0.patch_knn import inpaint_patch_knn as spatial_baseline

PINNED_BASELINE = "52e16b05aeec85d52730320b2a5755fd23407533"
PARAMETERS = {"patch_size": 5, "search_radius": 8, "k_neighbors": 3}
FAMILIES = ("constant", "x_stripes", "y_stripes", "checker", "gradient", "noise")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fixtures(seed: int = 23) -> list[dict]:
    rng = random.Random(seed)
    cases = []
    for family in FAMILIES:
        for variant in range(8):
            image = Image.new("RGB", (32, 32))
            first = tuple(rng.randrange(15, 100) for _ in range(3))
            second = tuple(rng.randrange(155, 245) for _ in range(3))
            period = variant % 3 + 1
            for y in range(32):
                for x in range(32):
                    if family == "constant":
                        color = first
                    elif family == "x_stripes":
                        color = first if (x // period) % 2 else second
                    elif family == "y_stripes":
                        color = first if (y // period) % 2 else second
                    elif family == "checker":
                        color = first if ((x // period) + (y // period)) % 2 else second
                    elif family == "gradient":
                        color = (5 * x + 2 * y + variant, 2 * x + 5 * y + variant, 3 * x + 3 * y + variant)
                    else:
                        color = tuple(rng.randrange(256) for _ in range(3))
                    image.putpixel((x, y), color)
            side = 2 + variant % 2
            x, y = rng.randrange(10, 20), rng.randrange(10, 20)
            mask = Image.new("L", image.size)
            ImageDraw.Draw(mask).rectangle((x, y, x + side - 1, y + side - 1), fill=255)
            damaged = image.copy()
            damaged.paste((255, 0, 255), (x, y, x + side, y + side))
            cases.append({"id": f"{family}-{variant}", "family": family, "reference": image,
                          "damaged": damaged, "mask": mask,
                          "reference_sha256": sha(image.tobytes()), "mask_sha256": sha(mask.tobytes())})
    return cases


def quality(reference: Image.Image, output: Image.Image, mask: Image.Image) -> dict:
    selected = [(x, y) for y in range(reference.height) for x in range(reference.width)
                if mask.getpixel((x, y)) > 128]
    boundary = [(x, y) for x, y in selected if any(
        0 <= x + dx < reference.width and 0 <= y + dy < reference.height and mask.getpixel((x + dx, y + dy)) <= 128
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)))]

    def mae(coords):
        return sum(abs(output.getpixel(xy)[c] - reference.getpixel(xy)[c]) for xy in coords for c in range(3)) / (255 * 3 * len(coords))

    return {"masked_rgb_mae": mae(selected), "boundary_rgb_mae": mae(boundary)}


def evaluate(seed: int = 23) -> dict:
    dataset = fixtures(seed)
    identities = [{key: case[key] for key in ("id", "family", "reference_sha256", "mask_sha256")} for case in dataset]
    rows = []
    methods = {"spatial_v0": spatial_baseline, "context_v1": inpaint_patch_knn}
    for case in dataset:
        original_bytes = case["damaged"].tobytes()
        for name, operation in methods.items():
            # The frozen baseline's random global fallback is not exercised by these fixtures.
            # Fail explicitly if future fixture changes would make comparison stochastic.
            with patch("benchmarks.inpaint_v0.patch_knn.random.sample", side_effect=AssertionError("Uncontrolled baseline fallback")):
                started = time.perf_counter()
                result = operation(case["damaged"], case["mask"], **PARAMETERS)
                elapsed = (time.perf_counter() - started) * 1000
            selected = sum(case["mask"].histogram()[129:])
            completed = result.metrics.verdict.method == "patch_knn" and result.metrics.filled_pixels == selected
            outside_unchanged = all(result.image.getpixel((x, y)) == case["damaged"].getpixel((x, y))
                                    for y in range(32) for x in range(32) if case["mask"].getpixel((x, y)) <= 128)
            source_unchanged = case["damaged"].tobytes() == original_bytes
            if not outside_unchanged or not source_unchanged:
                raise AssertionError("Image preservation gate failed")
            rows.append({"id": case["id"], "family": case["family"], "method": name,
                         "completed": completed, "filled_pixels": result.metrics.filled_pixels,
                         "reason": result.metrics.verdict.reason, "outside_unchanged": outside_unchanged,
                         "source_unchanged": source_unchanged, "latency_ms": elapsed,
                         "output_sha256": sha(result.image.tobytes()),
                         "quality": quality(case["reference"], result.image, case["mask"]) if completed else None})
    summaries = {}
    for name, operation in methods.items():
        mine = [row for row in rows if row["method"] == name]
        answered = [row for row in mine if row["completed"]]
        latencies = sorted(row["latency_ms"] for row in mine)
        tracemalloc.start()
        try:
            operation(dataset[24]["damaged"], dataset[24]["mask"], **PARAMETERS)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        summaries[name] = {
            "completed": len(answered), "abstained": len(mine) - len(answered),
            "mean_masked_mae_completed": statistics.mean(row["quality"]["masked_rgb_mae"] for row in answered) if answered else None,
            "mean_boundary_mae_completed": statistics.mean(row["quality"]["boundary_rgb_mae"] for row in answered) if answered else None,
            "latency_p50_ms": statistics.median(latencies), "latency_p95_ms": latencies[45],
            "python_allocation_peak_probe_bytes": peak,
        }
    paired = []
    for case in dataset:
        by_method = {row["method"]: row for row in rows if row["id"] == case["id"]}
        if all(row["completed"] for row in by_method.values()):
            paired.append({"family": case["family"], "id": case["id"],
                           "baseline": by_method["spatial_v0"]["quality"]["masked_rgb_mae"],
                           "context": by_method["context_v1"]["quality"]["masked_rgb_mae"]})
    by_family = {}
    for family in FAMILIES:
        family_pairs = [pair for pair in paired if pair["family"] == family]
        by_family[family] = {
            "paired_completed": len(family_pairs),
            "baseline_mean_masked_mae": statistics.mean(p["baseline"] for p in family_pairs) if family_pairs else None,
            "context_mean_masked_mae": statistics.mean(p["context"] for p in family_pairs) if family_pairs else None,
            "improved": sum(p["context"] < p["baseline"] for p in family_pairs),
            "worse": sum(p["context"] > p["baseline"] for p in family_pairs),
            "equal": sum(p["context"] == p["baseline"] for p in family_pairs),
        }
    root = Path(__file__).resolve().parents[1]
    tracked_sources = ("bellium/inpaint/patch_knn.py", "bellium/inpaint/router.py", "benchmarks/inpaint_context.py",
                       "benchmarks/inpaint_v0/patch_knn.py", "benchmarks/inpaint_v0/router.py")
    return {
        "schema": "bellium.inpaint-comparison/v1", "authority": "shadow", "seed": seed,
        "baseline_revision": PINNED_BASELINE, "parameters": PARAMETERS,
        "context_only_parameters": {"max_context_error": 0.05, "max_comparisons": 200_000},
        "dataset_sha256": sha(json.dumps(identities, sort_keys=True, separators=(",", ":")).encode()),
        "fixture_identities": identities,
        "source_sha256_lf": {name: sha((root / name).read_text(encoding="utf-8").encode()) for name in tracked_sources},
        "hardware": {"system": platform.system(), "machine": platform.machine(),
                     "python": platform.python_version(), "pillow": PIL.__version__, "device": "cpu"},
        "summary": summaries, "paired_by_family": by_family, "cases": rows,
        "limits": [
            "48 procedural 32x32 RGB fixtures; no real sprites, photographs or GPU measurements.",
            "Baseline files are pinned verbatim; original random fallback is forbidden during comparison.",
            "Abstentions are excluded from quality means and reported separately; compare common completed pairs.",
            "Unchanged damaged inputs returned on abstention are not scored as reconstructed images.",
            "Fixed parameters; no training, tuning, model activation or production authorization.",
            "Noise cannot in general be reconstructed from neighboring patches; unique details can be invented or lost.",
            "Latency is one complete inference per case, excluding decoding; allocation is a separate one-case Python probe.",
            "Color averaging can smooth detail. Correct execution and low context error do not imply a correct fill.",
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Choose a new output path")
    report = evaluate()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"dataset_sha256": report["dataset_sha256"], "summary": report["summary"],
                      "paired_by_family": report["paired_by_family"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
