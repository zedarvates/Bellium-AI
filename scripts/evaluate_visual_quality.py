"""Evaluate a frozen visual protocol offline; --fetch explicitly downloads public fixtures."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
import statistics
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bellium.knn.color_cutout import cutout  # noqa: E402
from bellium.knn.patch_inpaint import inpaint as patch_inpaint  # noqa: E402
from bellium.specialists.inpaint_router import route_inpaint  # noqa: E402

ENGINES = {
    "patch": ("bellium.knn.patch_inpaint", patch_inpaint),
}


def load_engine(name: str):
    """Resolve a named development candidate without changing the frozen default."""
    if name == "adaptive":
        from bellium.specialists.adaptive_inpaint import inpaint as adaptive_inpaint
        return "bellium.specialists.adaptive_inpaint", adaptive_inpaint
    if name not in ENGINES:
        raise ValueError(f"unknown engine '{name}'")
    return ENGINES[name]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_assets(protocol: dict, cache: Path, *, fetch: bool) -> list[dict]:
    cache.mkdir(parents=True, exist_ok=True)
    records = []
    for asset in protocol["assets"]:
        filename = asset["file"]
        if Path(filename).name != filename or Path(filename).suffix.lower() not in {".png", ".jpg"}:
            raise ValueError("asset must name one PNG or JPEG file")
        path = cache / filename
        # Some scikit-image assets live in the external data repository and are not
        # reachable through the package path; a manifest may pin the exact URL.
        url = asset.get("url") or (
            f'https://raw.githubusercontent.com/scikit-image/scikit-image/'
            f'{protocol["upstream_revision"]}/skimage/data/{filename}'
        )
        if not path.exists():
            if not fetch:
                raise FileNotFoundError(f"missing {filename}; use --fetch once")
            with urllib.request.urlopen(url, timeout=30) as response:
                data = response.read(4 * 1024 * 1024 + 1)
            if len(data) > 4 * 1024 * 1024:
                raise ValueError("asset exceeds the download budget")
            if digest(data) != asset["sha256"]:
                raise ValueError(f"download hash mismatch: {filename}")
            path.write_bytes(data)
        if digest(path.read_bytes()) != asset["sha256"]:
            raise ValueError(f"cached asset hash mismatch: {filename}")
        records.append({**asset, "url": url, "bytes": path.stat().st_size})
    return records


def rows(image: Image.Image) -> list:
    pixels = image.load()
    return [[pixels[c, r] for c in range(image.width)] for r in range(image.height)]


def image_from_rows(values: list) -> Image.Image:
    image = Image.new("RGB", (len(values[0]), len(values)))
    image.putdata([pixel for row in values for pixel in row])
    return image


def mask_metrics(reference: list, predicted: list, mask: list) -> dict:
    if not reference or len(predicted) != len(reference) or len(mask) != len(reference):
        raise ValueError("images and mask must have matching, nonempty shapes")
    errors = []
    outside_unchanged = True
    for r, row in enumerate(reference):
        if len(predicted[r]) != len(row) or len(mask[r]) != len(row):
            raise ValueError("images and mask must have matching shapes")
        for c, pixel in enumerate(row):
            if mask[r][c]:
                errors.extend(abs(a-b) for a, b in zip(pixel, predicted[r][c]))
            elif tuple(pixel) != tuple(predicted[r][c]):
                outside_unchanged = False
    if not errors:
        raise ValueError("quality scoring needs at least one masked pixel")
    mse = sum(e*e for e in errors) / len(errors)
    return {"mae_255": sum(errors) / len(errors),
            "psnr_db": None if mse == 0 else 10 * math.log10(255**2 / mse),
            "exact_reconstruction": mse == 0, "known_pixels_preserved": outside_unchanged}


def nearest_known(corrupted: list, mask: list) -> list:
    known = [(r, c) for r in range(len(mask)) for c in range(len(mask[0])) if not mask[r][c]]
    if not known:
        raise ValueError("baseline needs known pixels")
    result = [row[:] for row in corrupted]
    for r, row in enumerate(mask):
        for c, missing in enumerate(row):
            if missing:
                sr, sc = min(known, key=lambda point: ((point[0]-r)**2 + (point[1]-c)**2, point))
                result[r][c] = corrupted[sr][sc]
    return result


def ring_mean(corrupted: list, mask: list) -> list:
    missing = [(r, c) for r in range(len(mask)) for c in range(len(mask[0])) if mask[r][c]]
    r0, r1 = min(r for r, _ in missing), max(r for r, _ in missing)
    c0, c1 = min(c for _, c in missing), max(c for _, c in missing)
    ring = [corrupted[r][c] for r in range(max(0, r0-1), min(len(mask), r1+2))
            for c in range(max(0, c0-1), min(len(mask[0]), c1+2)) if not mask[r][c]]
    mean = tuple(round(sum(p[i] for p in ring) / len(ring)) for i in range(3))
    result = [row[:] for row in corrupted]
    for r, c in missing:
        result[r][c] = mean
    return result


def inpaint_summary(cases: list[dict], limits: dict) -> dict:
    accepted = [case for case in cases if not case["abstained"]]
    coverage = len(accepted) / len(cases)
    mean_mae = statistics.mean(case["patch"]["mae_255"] for case in accepted) if accepted else None
    baseline_mae = statistics.mean(case["nearest"]["mae_255"] for case in accepted) if accepted else None
    improvement = (1 - mean_mae / baseline_mae) if baseline_mae else (0.0 if mean_mae == 0 else None)
    bad = sum(case["patch"]["mae_255"] > case["severe_error_mae_255"] for case in accepted)
    bad_rate = bad / len(accepted) if accepted else None
    latencies = sorted(case["elapsed_seconds"] for case in cases)
    p95 = latencies[math.ceil(0.95 * len(latencies)) - 1]
    gates = {
        "coverage": coverage >= limits["minimum_coverage"],
        "mean_error": mean_mae is not None and mean_mae <= limits["maximum_accepted_mean_mae_255"],
        "beats_primary_baseline": improvement is not None and improvement >= limits["minimum_primary_baseline_improvement"],
        "bad_accept_rate": bad_rate is not None and bad_rate <= limits["maximum_bad_accept_rate"],
        "latency": p95 <= limits["maximum_p95_seconds"],
        "known_pixels_preserved": all(case["patch"]["known_pixels_preserved"] for case in cases),
    }
    return {"cases": len(cases), "accepted": len(accepted), "coverage": coverage,
            "accepted_mean_mae_255": mean_mae, "nearest_mean_on_same_cases": baseline_mae,
            "primary_baseline_improvement": improvement, "severe_accepted_cases": bad,
            "bad_accept_rate": bad_rate, "p95_seconds": p95, "gates": gates, "passed": all(gates.values())}


def run_photos(protocol: dict, cache: Path, output: Path, baseline=None, engine=None,
               engine_name: str = "patch") -> list[dict]:
    config = protocol["inpaint"]
    engine = engine or patch_inpaint
    cases = []
    for asset in protocol["assets"]:
        if asset["kind"] != "photograph":
            continue
        reference = Image.open(cache / asset["file"]).convert("RGB")
        reference.thumbnail((config["long_side"], config["long_side"]), Image.Resampling.LANCZOS)
        original = rows(reference)
        reference.save(output / f'{asset["id"]}-reference.png')
        for point_index, (fx, fy) in enumerate(config["centers_xy"]):
            for side in config["hole_sides"]:
                cx, cy = round(fx*(reference.width-1)), round(fy*(reference.height-1))
                mask = [[int(abs(r-cy) <= side//2 and abs(c-cx) <= side//2)
                         for c in range(reference.width)] for r in range(reference.height)]
                corrupted = [[(0, 0, 0) if mask[r][c] else original[r][c]
                              for c in range(reference.width)] for r in range(reference.height)]
                start = time.perf_counter()
                result = engine(corrupted, mask)
                elapsed = time.perf_counter() - start
                route = route_inpaint(corrupted, mask)
                prediction = result.output.get("image", corrupted)
                # A compliant consumer applies nothing when the specialist abstains.
                delivered = corrupted if result.abstained else prediction
                baseline_record = None
                if baseline is not None:
                    baseline_start = time.perf_counter()
                    baseline_result = baseline(corrupted, mask)
                    baseline_elapsed = time.perf_counter() - baseline_start
                    baseline_record = {
                        "abstained": baseline_result.abstained,
                        "confidence": baseline_result.confidence,
                        "elapsed_seconds": baseline_elapsed,
                        "patch": mask_metrics(original, baseline_result.output.get("image", corrupted), mask),
                    }
                nearest = nearest_known(corrupted, mask)
                ring = ring_mean(corrupted, mask)
                case_id = f'{asset["id"]}-p{point_index}-s{side}'
                filename = case_id + "-comparison.png"
                panels = [reference, image_from_rows(corrupted), image_from_rows(prediction), image_from_rows(nearest)]
                sheet = Image.new("RGB", (reference.width*4, reference.height+22), "white")
                drawing = ImageDraw.Draw(sheet)
                for i, (panel, label) in enumerate(zip(panels, ("Reference", "Masked", "Bellium", "Nearest"))):
                    sheet.paste(panel, (i*reference.width, 22))
                    drawing.text((i*reference.width+4, 4), label, fill="black")
                sheet.save(output / filename)
                case = {"id": case_id, "asset": asset["id"], "size": list(reference.size),
                        "center_xy": [cx, cy], "hole_side": side, "abstained": result.abstained,
                        "confidence": result.confidence, "filled": result.output.get("filled", 0),
                        "quality": result.output.get("quality"), "reason": result.output.get("reason"),
                        "baseline": baseline_record,
                        "elapsed_seconds": elapsed, "severe_error_mae_255": config["severe_error_mae_255"],
                        "route_label": route.output["label"], "route_confidence": route.confidence,
                        "engine": engine_name,
                        "patch": mask_metrics(original, prediction, mask),
                        "delivered": mask_metrics(original, delivered, mask),
                        "nearest": mask_metrics(original, nearest, mask),
                        "ring_mean": mask_metrics(original, ring, mask), "comparison": filename}
                cases.append(case)
                print(case_id, f'MAE={case["patch"]["mae_255"]:.2f}', flush=True)
    return cases


def run_cutout(protocol: dict, cache: Path, output: Path) -> dict:
    config = protocol["cutout"]
    silhouette = Image.open(cache / "horse.png").convert("L")
    # The reference is a black horse on white; prepare a known alpha on a canvas.
    foreground = ImageOps.invert(silhouette).point(lambda p: 255 if p >= 128 else 0)
    bbox = foreground.getbbox()
    foreground = foreground.crop(bbox)
    foreground.thumbnail((96, 96), Image.Resampling.NEAREST)
    alpha = Image.new("L", tuple(config["silhouette_canvas"]), 0)
    alpha.paste(foreground, ((alpha.width-foreground.width)//2, (alpha.height-foreground.height)//2))
    truth = [[alpha.getpixel((c, r)) > 0 for c in range(alpha.width)] for r in range(alpha.height)]
    cases = []
    backgrounds = [*config["backgrounds"], config["low_contrast_background"]]
    for index, background in enumerate(backgrounds):
        image = Image.new("RGB", alpha.size, tuple(background))
        image.paste(tuple(config["foreground_color"]), (0, 0, alpha.width, alpha.height), alpha)
        start = time.perf_counter()
        result = cutout(rows(image))
        elapsed = time.perf_counter() - start
        iou = None
        if not result.abstained:
            proposed = result.output["alpha"]
            intersection = sum(bool(proposed[r][c]) and truth[r][c] for r in range(alpha.height) for c in range(alpha.width))
            union = sum(bool(proposed[r][c]) or truth[r][c] for r in range(alpha.height) for c in range(alpha.width))
            iou = intersection / union if union else 1.0
        image.save(output / f"cutout-{index}-input.png")
        cases.append({"id": f"cutout-{index}", "role": "low_contrast" if index == len(backgrounds)-1 else "in_scope_composite",
                      "abstained": result.abstained, "confidence": result.confidence, "mask_iou": iou,
                      "seconds": elapsed, "reason": result.output.get("reason")})
    in_scope = [case for case in cases if case["role"] == "in_scope_composite"]
    gates = {"in_scope_coverage": sum(not c["abstained"] for c in in_scope)/len(in_scope) >= config["minimum_coverage"],
             "mask_iou": all(c["mask_iou"] is not None and c["mask_iou"] >= config["minimum_mask_iou"] for c in in_scope),
             "low_contrast_abstention": cases[-1]["abstained"]}
    return {"cases": cases, "gates": gates, "passed": all(gates.values()),
            "scope": "controlled composites; not photographic segmentation"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "benchmarks/manifests/visual-quality-v1.json")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--baseline-module", type=Path)
    parser.add_argument("--engine", default="patch", choices=["patch", "adaptive"])
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("choose a new output directory to preserve previous evidence")
    protocol_bytes = args.manifest.read_bytes()
    protocol = json.loads(protocol_bytes)
    baseline = None
    baseline_sha = None
    if protocol.get("baseline"):
        if args.baseline_module is None:
            raise ValueError("this protocol requires its pinned baseline module")
        baseline_sha = digest(args.baseline_module.read_bytes())
        if baseline_sha != protocol["baseline"]["sha256"]:
            raise ValueError("baseline code hash does not match protocol")
        spec = importlib.util.spec_from_file_location("bellium_evaluation_baseline", args.baseline_module)
        baseline_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline_module)
        baseline = baseline_module.inpaint
    assets = fetch_assets(protocol, args.cache, fetch=args.fetch)
    args.output.mkdir(parents=True)
    started = datetime.now(timezone.utc).isoformat()
    engine_paths = [*ROOT.glob("bellium/**/*.py"), *ROOT.glob("models/**/*.json")]
    engine_hashes = {path.relative_to(ROOT).as_posix(): digest(path.read_bytes()) for path in engine_paths}
    (args.output / "protocol.json").write_bytes(protocol_bytes)
    (args.output / "engine-manifest.json").write_text(json.dumps(engine_hashes, indent=2), encoding="utf-8")
    engine_module, engine = load_engine(args.engine)
    engine_path = Path(sys.modules[engine_module].__file__).resolve()
    engine_sha = digest(engine_path.read_bytes())
    cases = run_photos(protocol, args.cache, args.output, baseline=baseline,
                       engine=engine, engine_name=args.engine)
    summary = inpaint_summary(cases, protocol["inpaint"]["gates"])
    result = {"schema": "bellium.visual-quality-report/v1", "started_at": started,
              "protocol_sha256": digest(protocol_bytes), "runner_sha256": digest(Path(__file__).read_bytes()),
              "python": sys.version, "platform": platform.platform(), "assets": assets,
              "inpaint": {"summary": summary, "cases": cases},
              "cutout": run_cutout(protocol, args.cache, args.output),
              "model_or_threshold_changes_during_run": False, "authority_promotion": False,
              "baseline_sha256": baseline_sha}
    result["engine"] = {"name": args.engine, "module": engine_module,
                        "sha256": engine_sha,
                        "default_engine": args.engine == "patch"}
    if baseline is not None:
        baseline_cases = [{**case, **case["baseline"]} for case in cases]
        result["inpaint"]["baseline_summary"] = inpaint_summary(baseline_cases, protocol["inpaint"]["gates"])
    result["engine_changed_during_run"] = [path for path, expected in engine_hashes.items() if digest((ROOT/path).read_bytes()) != expected]
    result["passed"] = summary["passed"] and result["cutout"]["passed"] and not result["engine_changed_during_run"]
    (args.output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"inpaint": summary, "cutout": result["cutout"]["gates"], "passed": result["passed"]}, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
