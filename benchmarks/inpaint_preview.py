"""Replay old development cases and compare fixed previews on new identities."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import platform
import random
import statistics
import time
import tracemalloc

import PIL
from PIL import Image

from bellium.inpaint import PREVIEW_METHODS, inpaint_patch_knn, inpaint_preview
from benchmarks.inpaint_context import fixtures as development_fixtures, quality, sha

BASELINE_REVISION = "018f0c10c7cd7bb8458a237f2957d30dadb73cb4"
PRE_EVALUATION_PREVIEW_SHA256 = "89f32baa31ed60d4576ade795feb0016886c17f57af97438350ddc9342ec7044"
PARAMETERS = {"patch_size": 5, "search_radius": 8, "k_neighbors": 3,
              "max_comparisons": 200_000, "max_context_error": 0.05}
SEED = 104729
FAMILIES = ("affine_integer", "affine_quantized", "bilinear", "constant", "stripes", "checker",
            "curved", "step", "noise", "hidden_detail", "edge_gradient", "alpha_edge")


def fixtures() -> list[dict]:
    rng = random.Random(SEED)
    cases = []
    for family in FAMILIES:
        for variant in range(8):
            w, h = rng.randrange(36, 45), rng.randrange(36, 45)
            side = 1 + variant % 4
            mx, my = rng.randrange(10, w - 9 - side), rng.randrange(10, h - 9 - side)
            if family == "edge_gradient":
                mx = variant % 2
            selected = {(mx + i, my + j) for j in range(side) for i in range(side)
                        if variant % 3 != 2 or i == j}
            base = tuple(rng.randrange(96, 160) for _ in range(3))
            second = tuple(rng.randrange(200, 240) for _ in range(3))
            a = tuple(rng.choice((-2, -1, 1, 2)) for _ in range(3))
            b = tuple(rng.choice((-2, -1, 1, 2)) for _ in range(3))
            qa = tuple(rng.choice((-7, -5, -3, 1, 3, 5, 7)) for _ in range(3))
            qb = tuple(rng.choice((-7, -5, -3, 1, 3, 5, 7)) for _ in range(3))
            cross = tuple(rng.choice((-2, -1, 1, 2)) for _ in range(3))
            period = 1 + variant % 3
            phase = rng.randrange(2 * period)
            mode = "RGBA" if variant % 2 or family == "alpha_edge" else "RGB"
            image = Image.new(mode, (w, h))
            for y in range(h):
                for x in range(w):
                    dx, dy = x - w // 2, y - h // 2
                    color = tuple(base[c] + a[c] * dx + b[c] * dy for c in range(3))
                    if family == "affine_quantized":
                        color = tuple((2 * (4 * base[c] + qa[c] * dx + qb[c] * dy) + 4) // 8 for c in range(3))
                    elif family == "bilinear":
                        color = tuple((2 * (16 * base[c] + 8 * a[c] * dx + 8 * b[c] * dy
                                            + cross[c] * dx * dy) + 16) // 32 for c in range(3))
                    elif family == "constant":
                        color = base
                    elif family == "stripes":
                        color = base if ((x + phase) // period) % 2 else second
                    elif family == "checker":
                        color = base if (((x + phase) // period) + (y // period)) % 2 else second
                    elif family == "curved":
                        bend = 3 * (x - mx - side // 2) ** 2 + 2 * (y - my - side // 2) ** 2
                        color = tuple(base[c] + bend for c in range(3))
                    elif family == "step":
                        color = base if x + y < mx + my + side else second
                    elif family == "noise":
                        color = tuple(rng.randrange(256) for _ in range(3))
                    elif family == "hidden_detail" and (x, y) in selected:
                        color = (240, 20 + variant, 210)
                    rgb = tuple(max(0, min(255, v)) for v in color)
                    alpha = (96 if x < mx + side // 2 else 224) if family == "alpha_edge" else (64, 128, 192, 255)[variant % 4]
                    image.putpixel((x, y), rgb if mode == "RGB" else (*rgb, alpha))
            mask = Image.new("L", image.size)
            damaged = image.copy()
            for xy in sorted(selected):
                mask.putpixel(xy, 255)
                damaged.putpixel(xy, (255, 0, 255) if mode == "RGB" else (255, 0, 255, image.getpixel(xy)[3]))
            cases.append({"id": f"new-{family}-{variant}", "family": family, "reference": image,
                          "damaged": damaged, "mask": mask})
    return cases


def identities(dataset):
    return [{"id": case["id"], "family": case["family"], "size": list(case["reference"].size),
             "mode": case["reference"].mode, "reference_sha256": sha(case["reference"].tobytes()),
             "mask_sha256": sha(case["mask"].tobytes()), "damaged_sha256": sha(case["damaged"].tobytes())}
            for case in dataset]


def evaluate_cases(dataset):
    rows = []
    methods = {"context_v1": inpaint_patch_knn, "preview_v1": inpaint_preview}
    for case in dataset:
        reference, damaged, mask = (case[key] for key in ("reference", "damaged", "mask"))
        original_bytes, mask_bytes = damaged.tobytes(), mask.tobytes()
        selected = [(x, y) for y in range(mask.height) for x in range(mask.width) if mask.getpixel((x, y)) > 128]
        baseline = None
        for name, operation in methods.items():
            started = time.perf_counter()
            result = operation(damaged, mask, **PARAMETERS)
            elapsed = (time.perf_counter() - started) * 1000
            metrics = result.metrics
            completed = metrics.verdict.method in PREVIEW_METHODS and metrics.filled_pixels == len(selected)
            gate = getattr(metrics, "interpolation", None)
            preservation = {
                "source_unchanged": damaged.tobytes() == original_bytes,
                "mask_unchanged": mask.tobytes() == mask_bytes,
                "mode_unchanged": result.image.mode == damaged.mode,
                "outside_unchanged": all(result.image.getpixel((x, y)) == damaged.getpixel((x, y))
                                         for y in range(mask.height) for x in range(mask.width)
                                         if mask.getpixel((x, y)) <= 128),
                "alpha_unchanged": damaged.mode != "RGBA" or result.image.getchannel("A").tobytes() == damaged.getchannel("A").tobytes(),
            }
            if not all(preservation.values()):
                raise AssertionError("Pixel preservation gate failed")
            if not completed and (metrics.filled_pixels or result.image.tobytes() != original_bytes):
                raise AssertionError("Abstention must return the entire original")
            fallback_matches = None
            if gate is not None and not gate.accepted:
                fallback_matches = (result.image.tobytes() == baseline.image.tobytes()
                                    and metrics.verdict == baseline.metrics.verdict
                                    and metrics.filled_pixels == baseline.metrics.filled_pixels
                                    and metrics.comparisons == baseline.metrics.comparisons
                                    and metrics.discarded_pixels == baseline.metrics.discarded_pixels)
                if not fallback_matches:
                    raise AssertionError("Fallback changed from context v1")
            max_error = max(abs(result.image.getpixel(xy)[c] - reference.getpixel(xy)[c])
                            for xy in selected for c in range(3)) if completed else None
            rows.append({"id": case["id"], "family": case["family"], "method": name,
                         "synthesis_method": metrics.method, "completed": completed,
                         "filled_pixels": metrics.filled_pixels, "reason": metrics.verdict.reason,
                         "interpolation_gate": asdict(gate) if gate is not None else None,
                         "fallback_matches_baseline": fallback_matches, **preservation,
                         "latency_ms": elapsed, "output_sha256": sha(result.image.tobytes()),
                         "quality": quality(reference, result.image, mask) if completed else None,
                         "max_masked_rgb_error": max_error})
            if name == "context_v1":
                baseline = result
    summaries = {}
    for name, operation in methods.items():
        mine = [row for row in rows if row["method"] == name]
        completed = [row for row in mine if row["completed"]]
        interpolated = [row for row in completed if row["synthesis_method"] == "bilinear_rgb_v1"]
        latencies = sorted(row["latency_ms"] for row in mine)
        tracemalloc.start()
        try:
            operation(dataset[0]["damaged"], dataset[0]["mask"], **PARAMETERS)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        summaries[name] = {
            "completed": len(completed), "abstained": len(mine) - len(completed),
            "interpolated": len(interpolated),
            "nonexact_interpolation_candidates": sum(row["max_masked_rgb_error"] > 0 for row in interpolated),
            "interpolation_candidates_error_above_one": sum(row["max_masked_rgb_error"] > 1 for row in interpolated),
            "latency_p50_ms": statistics.median(latencies),
            "latency_p95_ms": latencies[math.ceil(0.95 * len(latencies)) - 1],
            "python_allocation_peak_probe_bytes": peak,
            "allocation_probe_id": dataset[0]["id"],
        }
    by_family = {}
    for family in dict.fromkeys(case["family"] for case in dataset):
        context = [row for row in rows if row["method"] == "context_v1" and row["family"] == family]
        preview = [row for row in rows if row["method"] == "preview_v1" and row["family"] == family]
        pairs = [(a["quality"]["masked_rgb_mae"], b["quality"]["masked_rgb_mae"])
                 for a, b in zip(context, preview) if a["completed"] and b["completed"]]
        by_family[family] = {
            "cases": len(context), "context_completed": sum(row["completed"] for row in context),
            "preview_completed": sum(row["completed"] for row in preview),
            "interpolated": sum(row["synthesis_method"] == "bilinear_rgb_v1" for row in preview),
            "paired_completed": len(pairs),
            "context_mean_masked_mae": statistics.mean(a for a, _ in pairs) if pairs else None,
            "preview_mean_masked_mae": statistics.mean(b for _, b in pairs) if pairs else None,
            "improved": sum(b < a for a, b in pairs), "equal": sum(b == a for a, b in pairs),
            "worse": sum(b > a for a, b in pairs),
            "interpolation_candidates_error_above_one": sum(row["synthesis_method"] == "bilinear_rgb_v1"
                                                            and row["max_masked_rgb_error"] > 1 for row in preview),
        }
    ids = identities(dataset)
    return {"dataset_sha256": sha(json.dumps(ids, sort_keys=True, separators=(",", ":")).encode()),
            "fixture_identities": ids, "summary": summaries, "paired_by_family": by_family, "cases": rows}


def evaluate():
    root = Path(__file__).resolve().parents[1]
    sources = ("bellium/inpaint/preview.py", "bellium/inpaint/patch_knn.py", "bellium/inpaint/router.py",
               "benchmarks/inpaint_preview.py", "benchmarks/inpaint_context.py", "benchmarks/preview_protocol.md")
    hashes = {name: sha((root / name).read_text(encoding="utf-8").encode()) for name in sources}
    if hashes["bellium/inpaint/preview.py"] != PRE_EVALUATION_PREVIEW_SHA256:
        raise AssertionError("Mechanism changed after protocol freeze; version the evaluation")
    development, fresh = development_fixtures(), fixtures()
    old_hashes = {item["reference_sha256"] for item in identities(development)}
    new_hashes = {item["reference_sha256"] for item in identities(fresh)}
    if old_hashes & new_hashes or len(new_hashes) != len(fresh):
        raise AssertionError("Duplicated or overlapping evaluation identities")
    return {
        "schema": "bellium.inpaint-preview-comparison/v1", "authority": "review_candidates_only",
        "baseline_revision": BASELINE_REVISION, "seed": SEED, "parameters": PARAMETERS,
        "pre_evaluation_preview_sha256": PRE_EVALUATION_PREVIEW_SHA256,
        "source_sha256_lf": hashes, "reference_hashes_disjoint": True,
        "hardware": {"system": platform.system(), "machine": platform.machine(),
                     "python": platform.python_version(), "pillow": PIL.__version__, "device": "cpu"},
        "development_48": evaluate_cases(development), "new_identities_96": evaluate_cases(fresh),
        "limits": [
            "Fixed mechanism and protocol before the first new-identity evaluation; no post-result tuning.",
            "Procedural RGB/RGBA fixtures, not independent human-annotated assets or production authorization.",
            "The earlier 48 cases are development evidence; the earlier frozen report remains unchanged.",
            "Quality pairs include only cases completed by both methods; abstention quality is null.",
            "Nonexact interpolation counts include one-code-value quantization differences; errors above one are separate.",
            "Entirely hidden details cannot be inferred from matching context; retain all adverse candidates and human review.",
            "Interpolation works in stored RGB, not linear light, and requires uniform positive local alpha.",
            "CPU latency includes both interpolation gates and fallback, excludes file decoding, and is not CLI timing.",
            "Allocation is a separate first-case Python probe, not full process RAM or GPU VRAM.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Choose a new output path")
    report = evaluate()
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({name: {key: report[name][key] for key in ("dataset_sha256", "summary", "paired_by_family")}
                      for name in ("development_48", "new_identities_96")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
