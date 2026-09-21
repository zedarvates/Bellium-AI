"""Offline, manifest-verified measurements and a development-first spatial gate."""

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import statistics
import sys
import time
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, __version__ as pillow_version  # noqa: E402
from bellium.compression import (  # noqa: E402
    decode_image, decode_splats, decode_stream, encode_image, encode_splats,
    encode_stream, inspect_packet,
)
from bellium.compression.container import METHODS  # noqa: E402
from bellium.compression.spatial_experiment import (  # noqa: E402
    PREDICTORS, decode_spatial_image, encode_spatial_image,
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_corpus(directory):
    directory = directory.resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "bellium-independent-compression-corpus/1":
        raise ValueError("unknown corpus schema")
    sources = {source["id"]: source for source in manifest["sources"]}
    if len(sources) != len(manifest["sources"]):
        raise ValueError("duplicate source id")
    source_groups = {}
    for source in sources.values():
        previous = source_groups.setdefault(source["sha256"], source["split"])
        if previous != source["split"]:
            raise ValueError("identical source bytes cross corpus split")
    ids = set()
    for entry in (*manifest["sources"], *manifest["samples"]):
        path = (directory / entry["path"]).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError("corpus path must be a contained file")
        if digest(path.read_bytes()) != entry["sha256"]:
            raise ValueError("corpus fingerprint mismatch")
    for sample in manifest["samples"]:
        if sample["id"] in ids:
            raise ValueError("duplicate sample id")
        ids.add(sample["id"])
        source = sources.get(sample["source_id"])
        if (source is None or source.get("split") != sample.get("split")
                or sample["split"] not in ("development", "holdout")):
            raise ValueError("source crosses corpus split")
        if not source.get("license") or not source.get("license_evidence"):
            raise ValueError("missing source license provenance")
        raw = (directory / sample["path"]).read_bytes()
        if len(raw) != sample["bytes"] or len(raw) > 65536:
            raise ValueError("sample size mismatch or work budget exceeded")
    return manifest


def measure(raw, encode, decode, repeats):
    encoding, decoding = [], []
    for _ in range(repeats):
        start = time.perf_counter()
        packet = encode()
        encoding.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        reconstructed = decode(packet)
        decoding.append((time.perf_counter() - start) * 1000)
        if reconstructed != raw:
            raise RuntimeError("reconstruction changed source bytes")
    return {"packet_bytes": len(packet), "exact": True,
            "method": inspect_packet(packet)["method"],
            "encode_ms_median": statistics.median(encoding),
            "decode_ms_median": statistics.median(decoding),
            "envelope_sha256": digest(packet)}


def spatial_rows(directory, samples, repeats):
    rows = []
    for sample in samples:
        raw = (directory / sample["path"]).read_bytes()
        options = {key: sample[key] for key in ("width", "height", "mode")}
        methods = {}
        for predictor in PREDICTORS:
            methods[predictor] = measure(raw,
                lambda: encode_spatial_image(raw, predictor=predictor, **options),
                lambda packet: decode_spatial_image(packet).pixels, repeats)
        best = min(PREDICTORS[:-1], key=lambda method: methods[method]["packet_bytes"])
        rows.append({"id": sample["id"], "best_fixed": best, "methods": methods})
    return rows


def spatial_gate(rows):
    if not rows:
        raise ValueError("spatial evaluation requires at least one image")
    fixed_bytes = sum(row["methods"][row["best_fixed"]]["packet_bytes"] for row in rows)
    learned_bytes = sum(row["methods"]["micro-nn"]["packet_bytes"] for row in rows)
    result = {"fixed_bytes": fixed_bytes, "learned_bytes": learned_bytes,
              "saved_fraction": 1 - learned_bytes / fixed_bytes,
              "worst_size_ratio": max(row["methods"]["micro-nn"]["packet_bytes"] /
                                      row["methods"][row["best_fixed"]]["packet_bytes"]
                                      for row in rows)}
    for phase in ("encode", "decode"):
        metric = f"{phase}_ms_median"
        result[f"{phase}_slowdown"] = (sum(row["methods"]["micro-nn"][metric] for row in rows)
            / max(1e-9, sum(row["methods"][row["best_fixed"]][metric] for row in rows)))
    result["development_passed"] = result["saved_fraction"] >= .01
    result["promotion_passed"] = (result["development_passed"] and result["worst_size_ratio"] <= 1.05
                                   and result["encode_slowdown"] <= 10 and result["decode_slowdown"] <= 10)
    return result


def evaluate(directory, output, repeats):
    manifest = load_corpus(directory)
    samples = manifest["samples"]
    root = Path(__file__).resolve().parents[1]
    tracked = [*sorted((root / "bellium/compression").glob("*.py")), Path(__file__).resolve(),
               root / "scripts/prepare_compression_corpus.py",
               root / "benchmarks/protocols/compression-independent-v1.md"]
    sources = {path.relative_to(root).as_posix(): digest(path.read_bytes()) for path in tracked}
    development = [s for s in samples if s["kind"] == "image" and s["split"] == "development"]
    dev_rows = spatial_rows(directory, development, repeats)
    gate = spatial_gate(dev_rows)
    candidate = {"development": dev_rows, "development_gate": gate,
                 "holdout_status": "not_run_development_gate_failed"}
    if gate["development_passed"]:
        holdout = [s for s in samples if s["kind"] == "image" and s["split"] == "holdout"]
        candidate["holdout"] = spatial_rows(directory, holdout, repeats)
        candidate["holdout_gate"] = spatial_gate(candidate["holdout"])
        candidate["holdout_status"] = "evaluated_once"
    print(json.dumps({"spatial_development_gate": gate,
                      "candidate_holdout": candidate["holdout_status"]}), flush=True)

    results = []
    for sample in samples:
        raw = (directory / sample["path"]).read_bytes()
        kind = sample["kind"]
        encoder = {"image": encode_image, "stream": encode_stream, "splats": encode_splats}[kind]
        decoder = {"image": lambda packet: decode_image(packet).pixels,
                   "stream": decode_stream, "splats": decode_splats}[kind]
        options = {key: sample[key] for key in ("width", "height", "mode")} if kind == "image" else {}
        row = {"id": sample["id"], "kind": kind, "split": sample["split"],
               "raw_bytes": len(raw), "methods": {}}
        for method in (*METHODS, "auto"):
            row["methods"][method] = measure(raw,
                lambda: encoder(raw, method=method, **options), decoder, repeats)
        selected = row["methods"]["auto"]["method"]
        row["selected"] = selected
        if kind == "image":
            buffer = io.BytesIO()
            Image.frombytes(sample["mode"], (sample["width"], sample["height"]), raw).save(buffer, "PNG")
            row["png_bytes"] = len(buffer.getvalue())
            with Image.open(io.BytesIO(buffer.getvalue())) as image:
                if image.tobytes() != raw:
                    raise RuntimeError("PNG reference changed pixels")
            row["png_exact"] = True
        results.append(row)
        print(f"measured {sample['id']}: {selected}", flush=True)
    report = {"schema": "bellium-independent-compression-results/1",
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "python": platform.python_version(), "pillow": pillow_version,
              "platform": platform.platform(), "zlib": zlib.ZLIB_VERSION, "repeats": repeats,
              "manifest_sha256": digest((directory / "manifest.json").read_bytes()),
              "source_sha256": sources, "all_exact": True,
              "sample_count": len(results), "spatial_candidate": candidate,
              "auto_selections": {method: sum(row["selected"] == method for row in results)
                                  for method in METHODS},
              "results": results, "limitations": manifest["limitations"],
              "runtime_integration": False, "fovea_comparison": "not_run"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output), "samples": len(results),
                      "all_exact": True, "auto_selections": report["auto_selections"]}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 5:
        parser.error("repeats must be in 1..5")
    evaluate(args.corpus, args.output, args.repeats)


if __name__ == "__main__":
    main()
