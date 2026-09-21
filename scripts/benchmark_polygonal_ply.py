"""Deterministic synthetic polygonal meshes for the v3 element-aware layouts.

Synthetic geometry only: no licensed real polygonal PLY is available offline in
this workspace. Sizes here are measurements of the codec, not of real scans.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import struct
import sys
import time
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bellium.compression.codecs import _planes  # noqa: E402
from bellium.compression.container import METHODS, inspect_packet, pack  # noqa: E402
from bellium.compression.ply_archive import (  # noqa: E402
    BINARY_KIND, decode_ply, encode_ply, inspect_ply,
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def mesh(*, vertices, faces, endian="little", normals=False):
    """Build a mesh whose vertex coordinates come from a fixed formula."""
    order = "<" if endian == "little" else ">"
    header = (f"ply\nformat binary_{endian}_endian 1.0\ncomment synthetic gate fixture\n"
              f"element vertex {vertices}\n"
              "property float x\nproperty float y\nproperty float z\n"
              + ("property float nx\nproperty float ny\nproperty float nz\n" if normals else "")
              + f"element face {len(faces)}\n"
              "property list uchar int vertex_indices\nend_header\n").encode()
    body = bytearray()
    for index in range(vertices):
        body += struct.pack(order + "fff", index * 0.03125, (index % 97) * 0.5, -index * 0.125)
        if normals:
            body += struct.pack(order + "fff", 0.0, 1.0, 0.0)
    for face in faces:
        body += struct.pack(order + "B", len(face))
        body += struct.pack(order + "i" * len(face), *face)
    return header + bytes(body)


def triangles(vertices):
    return [tuple(range(index, index + 3)) for index in range(0, vertices - 2, 3)]


def mixed(vertices):
    faces = []
    index = 0
    while index + 4 <= vertices:
        faces.append(tuple(range(index, index + 3)))
        faces.append(tuple(range(index, index + 4)))
        index += 4
    return faces


def shapes():
    yield "triangles-96", mesh(vertices=96, faces=triangles(96))
    yield "triangles-with-normals-480", mesh(vertices=480, faces=triangles(480), normals=True)
    yield "triangles-4800", mesh(vertices=4800, faces=triangles(4800))
    yield "mixed-arity-640", mesh(vertices=640, faces=mixed(640))
    yield "triangles-big-endian", mesh(vertices=300, faces=triangles(300), endian="big")


def timed(data, *, layout, method, repeats):
    encode_ms, decode_ms = [], []
    for _ in range(repeats):
        start = time.perf_counter()
        packet = encode_ply(data, layout=layout, method=method)
        encode_ms.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        if decode_ply(packet) != data:
            raise RuntimeError("polygonal reconstruction differs")
        decode_ms.append((time.perf_counter() - start) * 1000)
    return packet, {"packet_bytes": len(packet), "exact": True,
                    "encode_ms_median": statistics.median(encode_ms),
                    "decode_ms_median": statistics.median(decode_ms),
                    "method": inspect_packet(packet)["method"]}


def diagnostic(data, repeats, transposed):
    """Measure a range set the public API does not expose, e.g. vertices only."""
    info = inspect_ply(data)
    planes = bytearray(data)
    records = []
    for element in info.elements:
        if element.name not in transposed or not element.planar:
            continue
        chunk = bytes(planes[element.start:element.end])
        planes[element.start:element.end] = _planes(chunk, element.stride)
        records.append([element.start, element.end, element.stride])
    packet = pack(bytes(planes), {"kind": BINARY_KIND, "layout": "byte-planes",
                                  "source_sha256": digest(data), "planar": records},
                  method="zlib")
    if decode_ply(packet) != data:
        raise RuntimeError("diagnostic reconstruction differs")
    return {"bytes": len(packet), "ranges": records, "exact": True}


def run(output, repeats):
    rows = []
    for name, data in shapes():
        info = inspect_ply(data)
        row = {"shape": name, "source_bytes": len(data),
               "bare_zlib_bytes": len(zlib.compress(data, level=6)),
               "source_sha256": digest(data),
               "elements": [{"name": element.name, "count": element.count,
                             "stride": element.stride, "planar": element.planar,
                             "start": element.start, "end": element.end}
                            for element in info.elements],
               "methods": {}}
        for layout in ("original", "byte-planes", "auto"):
            for method in (*METHODS, "auto"):
                key = f"{layout}/{method}"
                try:
                    _, stats = timed(data, layout=layout, method=method, repeats=repeats)
                except ValueError as exc:
                    # The learned byte predictors are capped at 64 KiB per packet.
                    row["methods"][key] = {"skipped": str(exc)}
                else:
                    row["methods"][key] = stats
        row["vertices_only"] = diagnostic(data, repeats, {"vertex"})
        row["all_planar"] = diagnostic(data, repeats, {"vertex", "face"})
        row["auto"] = row["methods"]["auto/auto"]
        row["auto_vs_bare_zlib"] = round(1 - row["auto"]["packet_bytes"] / row["bare_zlib_bytes"], 5)
        rows.append(row)
        print(f"{name}: {row['auto']['packet_bytes']} bytes, exact", flush=True)

    root = Path(__file__).resolve().parents[1]
    tracked = [Path(__file__).resolve(), root / "bellium/compression/ply_archive.py",
               root / "benchmarks/protocols/polygonal-ply-v3.md"]
    report = {"schema": "bellium-polygonal-ply/1",
              "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "python": platform.python_version(), "platform": platform.platform(),
              "zlib": zlib.ZLIB_VERSION, "repeats": repeats,
              "all_exact": True, "synthetic": True,
              "code_sha256": {path.relative_to(root).as_posix(): digest(path.read_bytes())
                              for path in tracked},
              "shapes": rows,
              "limitations": ["Synthetic geometry, not scanned meshes",
                              "No licensed real polygonal PLY available offline",
                              "Timings are single-machine Python",
                              "No Godot, Zig, GPU or VR measurement"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output), "shapes": len(rows),
                      "all_exact": True, "zlib": report["zlib"]}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 5:
        parser.error("repeats must be in 1..5")
    run(args.output, args.repeats)


if __name__ == "__main__":
    main()
