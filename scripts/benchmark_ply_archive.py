"""Full-file PLY archival proof: bytes, timing, and separate Python allocation peaks."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time
import tracemalloc
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bellium.compression.container import MAX_RAW_BYTES, inspect_packet  # noqa: E402
from bellium.compression.ply_archive import decode_ply, encode_ply, inspect_ply  # noqa: E402


def digest(data):
    return hashlib.sha256(data).hexdigest()


def benchmark(data, repeats=3):
    info = inspect_ply(data)
    cases = (("bare-zlib", None, None), ("original-raw", "original", "raw"),
             ("original-zlib", "original", "zlib"), ("original-delta", "original", "delta"),
             ("planes-zlib", "byte-planes", "zlib"), ("planes-delta", "byte-planes", "delta"),
             ("archive-auto", "auto", "auto"))
    rows = []
    for name, layout, method in cases:
        enc_times, dec_times = [], []
        for _ in range(repeats):
            start = time.perf_counter()
            packet = (zlib.compress(data, level=6) if name == "bare-zlib" else
                      encode_ply(data, layout=layout, method=method))
            enc_times.append((time.perf_counter() - start) * 1000)
            start = time.perf_counter()
            decoded = zlib.decompress(packet) if name == "bare-zlib" else decode_ply(packet)
            dec_times.append((time.perf_counter() - start) * 1000)
            if decoded != data:
                raise RuntimeError("whole-file reconstruction differs")
        report = None if name == "bare-zlib" else inspect_packet(packet)
        rows.append({"case": name, "packet_bytes": len(packet), "exact": True,
                     "encode_ms_median": statistics.median(enc_times),
                     "decode_ms_median": statistics.median(dec_times),
                     "selected_layout": report["metadata"]["layout"] if report else None,
                     "selected_method": report["method"] if report else "zlib",
                     "packet_sha256": digest(packet)})
        print(f"{name}: {len(packet)} bytes, exact", flush=True)

    selected = rows[-1]
    parameters = {"layout": selected["selected_layout"], "method": selected["selected_method"]}
    tracemalloc.start()
    measured_packet = encode_ply(data, **parameters)
    _, encode_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    tracemalloc.start()
    reconstructed = decode_ply(measured_packet)
    _, decode_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    if reconstructed != data:
        raise RuntimeError("profiled reconstruction differs")
    root = Path(__file__).resolve().parents[1]
    tracked = [Path(__file__).resolve(), root / "bellium/compression/ply_archive.py",
               root / "bellium/compression/container.py", root / "bellium/compression/codecs.py",
               root / "bellium/compression/predictors.py",
               root / "benchmarks/protocols/full-ply-archive-v1.md"]
    return {"schema": "bellium-full-ply-archive-benchmark/1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "platform": platform.platform(),
            "zlib": zlib.ZLIB_VERSION, "source_bytes": len(data), "source_sha256": digest(data),
            "layout": asdict(info), "repeats": repeats, "all_exact": True,
            "results": rows,
            "selected_allocation_profile": {**parameters, "encode_peak_python_bytes": encode_peak,
                "decode_peak_python_bytes": decode_peak,
                "scope": "tracemalloc; excludes caller-owned input, not RSS/native memory; forced winner only"},
            "code_sha256": {path.relative_to(root).as_posix(): digest(path.read_bytes()) for path in tracked},
            "limitations": ["One known synthetic-scene-derived integration asset, not a new holdout",
                            "No neural gain claim; predictive methods capped at 64 KiB",
                            "No .fovea, runtime, GPU, VR or natural motion comparison"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 5:
        parser.error("repeats must be in 1..5")
    with args.source.open("rb") as handle:
        data = handle.read(MAX_RAW_BYTES + 1)
    if len(data) > MAX_RAW_BYTES or digest(data) != args.sha256:
        parser.error("source size or fingerprint differs from the reviewed asset")
    report = benchmark(data, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    packet = encode_ply(data)
    archive_path = args.output.with_suffix(".blcp")
    if decode_ply(packet) != data:
        raise RuntimeError("saved archive roundtrip failed")
    archive_path.write_bytes(packet)
    print(json.dumps({"report": str(args.output), "archive": str(archive_path),
                      "raw_bytes": len(data), "archive_bytes": len(packet), "exact": True,
                      "allocation_profile": report["selected_allocation_profile"]}))


if __name__ == "__main__":
    main()
