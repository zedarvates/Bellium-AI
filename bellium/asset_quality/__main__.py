"""Offline CLI: inspect an alpha asset or benchmark candidate specialists."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .benchmark import run_benchmark
from .dataset import load_dataset, synthetic_rows
from .features import FEATURE_NAMES, FEATURE_SCHEMA, extract_features, read_asset, rule_advice
from .models import Predictors


def inspect_asset(path: str | Path, bundle: dict | None = None, expected_sha256: str | None = None) -> dict:
    image, file_digest, size = read_asset(path)
    if expected_sha256 is not None and file_digest != expected_sha256:
        raise ValueError("Asset SHA-256 does not match expected bytes")
    features = extract_features(image)
    rules = rule_advice(features)
    learned = {}
    if bundle is not None:
        predictors = Predictors(bundle)
        rgba = image.convert("RGBA")
        pixel_digest = hashlib.sha256(str(rgba.size).encode() + rgba.tobytes()).hexdigest()
        learned = {name: predictors.predict(name, features, sha256=pixel_digest)
                   for name in ("knn", "nano", "micro")}
    return {"schema": "bellium.alpha-advice/v1", "authority": "consultative",
            "acted": False, "production_authorized": False,
            "source": {"sha256": file_digest, "bytes": size, "width": image.width, "height": image.height},
            "feature_schema": FEATURE_SCHEMA, "features": dict(zip(FEATURE_NAMES, features)),
            "rules": rules, "advice": rules["label"], "shadow_predictions": learned,
            "bundle_sha256": bundle["bundle_sha256"] if bundle else None,
            "scope": "alpha matte geometry; human review and existing asset gates still required"}


def _write_new(path: Path, data: dict) -> None:
    # Refuse overwrite, including an accidentally selected source image or manifest.
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="Read an RGBA asset and emit consultative JSON")
    inspect.add_argument("image", type=Path)
    inspect.add_argument("--models", type=Path)
    inspect.add_argument("--expected-sha256")
    inspect.add_argument("--output", type=Path)
    bench = sub.add_parser("benchmark", help="Fit and compare rules, k-NN, nano and micro")
    bench.add_argument("--dataset", type=Path, help="Local annotated manifest; omit for synthetic fixtures")
    bench.add_argument("--seed", type=int, default=17)
    bench.add_argument("--epochs", type=int, choices=range(1, 1001), metavar="1..1000", default=100)
    bench.add_argument("--output", type=Path, required=True)
    bench.add_argument("--models-output", type=Path, required=True)
    args = parser.parse_args(argv)
    written_outputs = []
    try:
        if args.command == "inspect":
            bundle = None
            if args.models:
                if args.models.stat().st_size > 8 * 1024 * 1024:
                    raise ValueError("Model bundle exceeds 8 MiB")
                bundle = json.loads(args.models.read_text(encoding="utf-8"))
            report = inspect_asset(args.image, bundle, args.expected_sha256)
            if args.output:
                _write_new(args.output, report)
                written_outputs.append(str(args.output))
            else:
                print(json.dumps(report, indent=2, allow_nan=False))
        else:
            if args.output.resolve() == args.models_output.resolve():
                raise ValueError("Report and model output paths must differ")
            if args.output.exists() or args.models_output.exists():
                raise ValueError("Output exists; choose new report/model filenames")
            if args.dataset:
                rows, provenance = load_dataset(args.dataset)
            else:
                rows = synthetic_rows(args.seed)
                provenance = "synthetic-alpha-fixtures/v1; labels from transformations; no real assets"
            report, bundle = run_benchmark(rows, provenance=provenance, seed=args.seed, epochs=args.epochs)
            _write_new(args.models_output, bundle)
            written_outputs.append(str(args.models_output))
            _write_new(args.output, report)
            written_outputs.append(str(args.output))
            print(json.dumps({"dataset_sha256": report["dataset_sha256"],
                              "bundle_sha256": report["bundle_sha256"], "split_counts": report["split_counts"]}))
        return 0
    except (OSError, ValueError, TypeError, KeyError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        print(json.dumps({"schema": "bellium.alpha-error/v1", "status": "invalid_input",
                          "error": str(exc), "written_outputs": written_outputs}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
