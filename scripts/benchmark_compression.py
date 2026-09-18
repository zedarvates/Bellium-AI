"""Reproducible synthetic codec comparisons; no private assets or network access."""

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import platform
import random
import statistics
import struct
import sys
import time
import zlib

# This also supports running the source script from another working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bellium.compression import (  # noqa: E402
    decode_image, decode_splats, decode_stream, encode_image, encode_splats,
    encode_stream, inspect_packet,
)
from bellium.compression.container import METHODS  # noqa: E402


def fixtures():
    rng = random.Random(20260915)
    yield "stream/repeating-telemetry", "stream", (b"tick=12;xyz=1,2,3;hp=100\n" * 160), {}
    yield "stream/ramp", "stream", bytes(range(256)) * 16, {}
    yield "stream/noise", "stream", bytes(rng.randrange(256) for _ in range(4096)), {}
    for name in ("gradient", "transparent", "noise"):
        pixels = bytearray()
        for y in range(32):
            for x in range(32):
                color = ((x * 8, y * 8, (x + y) * 4, 255) if name == "gradient" else
                         (80, 120, 240, 0 if x < 16 else 255) if name == "transparent" else
                         tuple(rng.randrange(256) for _ in range(4)))
                pixels.extend(color)
        yield f"image/{name}", "image", bytes(pixels), {"width": 32, "height": 32, "mode": "RGBA"}
    for count in (64, 128):
        def frame(tick):
            return b"".join(struct.pack("<14f", i / 8 + tick / 32, i % 8, -0.0,
                                        0, 0, 0, 1, 0.1, 0.2, 0.3, 0.8, 0.1, 0.5, 1)
                            for i in range(count))
        base, current = frame(0), frame(1)
        yield f"splats/static-{count}", "splats", base, {}
        yield f"splats/motion-{count}", "splats", current, {"base": base}


def benchmark(*, repeats=3):
    results = []
    for name, kind, raw, options in fixtures():
        encoder = {"stream": encode_stream, "image": encode_image, "splats": encode_splats}[kind]
        record = {"fixture": name, "kind": kind, "raw_bytes": len(raw),
                  "source_sha256": hashlib.sha256(raw).hexdigest(),
                  "plain_zlib_bytes": len(zlib.compress(raw, level=6)), "methods": []}
        if kind == "image":
            try:
                from PIL import Image
                target = io.BytesIO()
                Image.frombytes("RGBA", (32, 32), raw).save(target, format="PNG")
                record["png_bytes"] = len(target.getvalue())
                with Image.open(io.BytesIO(target.getvalue())) as reconstructed:
                    if reconstructed.tobytes() != raw:
                        raise RuntimeError("PNG reference did not preserve the source pixels")
                record["png_exact"] = True
            except ImportError:
                record["png_bytes"] = None
        base = options.get("base")
        if base is not None:
            record["base_packet_bytes"] = len(encode_splats(base))
            record["two_static_packets_bytes"] = (
                record["base_packet_bytes"] + len(encode_splats(raw)))
        for method in (*METHODS, "auto"):
            encode_times, decode_times = [], []
            for _ in range(repeats):
                start = time.perf_counter()
                packet = encoder(raw, method=method, **options)
                encode_times.append((time.perf_counter() - start) * 1000)
                start = time.perf_counter()
                decoded = (decode_image(packet).pixels if kind == "image" else
                           decode_splats(packet, base=base) if kind == "splats" else
                           decode_stream(packet))
                decode_times.append((time.perf_counter() - start) * 1000)
                if decoded != raw:
                    raise RuntimeError(f"non-exact reconstruction: {name}/{method}")
            stats = inspect_packet(packet)
            entry = {"requested": method, "selected": stats["method"],
                     "packet_bytes": len(packet), "exact": True,
                     "encode_ms_median": statistics.median(encode_times),
                     "decode_ms_median": statistics.median(decode_times)}
            if base is not None:
                entry["total_with_base_bytes"] = len(packet) + record["base_packet_bytes"]
            record["methods"].append(entry)
        results.append(record)
    selected = [next(row for row in record["methods"] if row["requested"] == "auto")
                for record in results]
    root = Path(__file__).resolve().parents[1]
    source_files = sorted((root / "bellium" / "compression").glob("*.py"))
    source_files.append(Path(__file__).resolve())
    source_hashes = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in source_files}
    return {"schema": "bellium-compression-benchmark/1", "seed": 20260915,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": source_hashes,
            "python": platform.python_version(), "platform": platform.platform(),
            "repeats": repeats, "fixture_count": len(results), "all_exact": True,
            "automatic_selections": {method: sum(row["selected"] == method for row in selected)
                                     for method in METHODS},
            "evidence": "Synthetic fixtures only. No pretrained weights, private assets or runtime integration.",
            "fovea_comparison": "not_run", "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10:
        parser.error("repeats must be in 1..10")
    report = benchmark(repeats=args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}))


if __name__ == "__main__":
    main()
