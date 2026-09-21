"""Measure the archive on an external, hash-pinned polygonal PLY corpus.

The corpus and its provenance live outside the repository; this script only
verifies the recorded fingerprints and reports what the codec does.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bellium.compression.container import METHODS, inspect_packet  # noqa: E402
from bellium.compression.ply_archive import (  # noqa: E402
    ASCII_LAYOUTS, BINARY_LAYOUTS, decode_ply, encode_ply, inspect_ply,
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def layouts_for(info):
    return ASCII_LAYOUTS if info.ascii else BINARY_LAYOUTS


def measure(data, info, layout, method, repeats):
    encode_ms, decode_ms = [], []
    for _ in range(repeats):
        start = time.perf_counter()
        packet = encode_ply(data, layout=layout, method=method)
        encode_ms.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        if decode_ply(packet) != data:
            raise RuntimeError("reconstruction differs")
        decode_ms.append((time.perf_counter() - start) * 1000)
    return {"packet_bytes": len(packet), "exact": True,
            "selected_layout": inspect_packet(packet)["metadata"]["layout"],
            "selected_method": inspect_packet(packet)["method"],
            "encode_ms_median": statistics.median(encode_ms),
            "decode_ms_median": statistics.median(decode_ms),
            "packet_sha256": digest(packet)}


def run(corpus, provenance, output, repeats):
    rows = []
    for entry in provenance["files"]:
        path = corpus / entry["name"]
        data = path.read_bytes()
        if digest(data) != entry["sha256"]:
            raise ValueError(f"corpus fingerprint mismatch: {entry['name']}")
        row = {"name": entry["name"], "source_bytes": len(data),
               "sha256": entry["sha256"], "url": entry["url"],
               "bare_zlib_bytes": len(zlib.compress(data, level=6)), "methods": {}}
        try:
            info = inspect_ply(data)
        except ValueError as exc:
            row["refused"] = str(exc)
            rows.append(row)
            print(f"{entry['name']}: refused", flush=True)
            continue
        row["encoding"] = info.encoding
        row["elements"] = [{"name": element.name, "count": element.count,
                            "stride": element.stride, "planar": element.planar}
                           for element in info.elements]
        choices = (*layouts_for(info), "auto")
        for layout in choices:
            for method in (*METHODS, "auto"):
                key = f"{layout}/{method}"
                try:
                    row["methods"][key] = measure(data, info, layout, method, repeats)
                except ValueError as exc:
                    row["methods"][key] = {"skipped": str(exc)}
        row["auto"] = row["methods"]["auto/auto"]
        row["gain_vs_bare_zlib"] = round(1 - row["auto"]["packet_bytes"]
                                         / row["bare_zlib_bytes"], 5)
        rows.append(row)
        print(f"{entry['name']}: {row['auto']['packet_bytes']} bytes "
              f"({row['auto']['selected_layout']}/{row['auto']['selected_method']})", flush=True)

    root = Path(__file__).resolve().parents[1]
    tracked = [Path(__file__).resolve(), root / "bellium/compression/ply_archive.py",
               root / "benchmarks/protocols/polygonal-ply-v3.md"]
    report = {"schema": "bellium-polygonal-corpus/1",
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "python": platform.python_version(), "platform": platform.platform(),
              "zlib": zlib.ZLIB_VERSION, "repeats": repeats,
              "repository": provenance["repository"], "commit": provenance["commit"],
              "license": provenance["license"], "provenance_note": provenance["provenance_note"],
              "all_exact": all(row["auto"]["exact"] for row in rows if "refused" not in row),
              "accepted": sum("refused" not in row for row in rows),
              "refused": sum("refused" in row for row in rows),
              "code_sha256": {path.relative_to(root).as_posix(): digest(path.read_bytes())
                              for path in tracked},
              "files": rows,
              "limitations": ["Local measurement of a third-party test corpus; not redistributed",
                              "Ultimate provenance of individual meshes not independently verified",
                              "No Godot, Zig, GPU or VR measurement"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output), "accepted": report["accepted"],
                      "refused": report["refused"], "all_exact": report["all_exact"],
                      "zlib": report["zlib"]}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--provenance", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 5:
        parser.error("repeats must be in 1..5")
    run(args.corpus, json.loads(args.provenance.read_text(encoding="utf-8")), args.output,
        args.repeats)


if __name__ == "__main__":
    main()
