"""Coverage and size gate for the PLY archive, driven by an external manifest.

The manifest keeps environment-specific paths, rights labels and `.fovea`
references outside the repository. The script itself is generic.
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

from bellium.compression.container import ABSOLUTE_MAX_BYTES, MAX_RAW_BYTES  # noqa: E402
from bellium.compression.ply_archive import decode_ply, encode_ply, inspect_ply  # noqa: E402

BUFFER = ABSOLUTE_MAX_BYTES


def digest(data):
    return hashlib.sha256(data).hexdigest()


def classify(path, limit):
    """Accept or refuse without reading more than the declared limit."""
    with path.open("rb") as handle:
        data = handle.read(limit + 1)
    if len(data) > limit:
        return data, False, "larger than the declared limit", None
    try:
        info = inspect_ply(data, limit=limit)
    except ValueError as exc:
        return data, False, str(exc), None
    return data, True, None, info


def measure(data, limit, repeats):
    encode_ms, decode_ms = [], []
    for _ in range(repeats):
        start = time.perf_counter()
        packet = encode_ply(data, limit=limit)
        encode_ms.append((time.perf_counter() - start) * 1000)
        start = time.perf_counter()
        if decode_ply(packet, max_output_bytes=limit) != data:
            raise RuntimeError("whole-file reconstruction differs")
        decode_ms.append((time.perf_counter() - start) * 1000)
    report = inspect_ply(data, limit=limit)
    return {
        "packet_bytes": len(packet),
        "bare_zlib_bytes": len(zlib.compress(data, level=6)),
        "encode_ms_median": statistics.median(encode_ms),
        "decode_ms_median": statistics.median(decode_ms),
        "packet_sha256": digest(packet),
        "encoding": report.encoding,
        "vertex_count": report.vertex_count,
        "property_count": len(report.properties),
        "exact": True,
    }


def run(manifest, output, legacy_packet, repeats):
    entries = []
    for asset in manifest["assets"]:
        path = Path(asset["path"])
        limit = int(asset.get("limit", manifest.get("default_limit", MAX_RAW_BYTES)))
        row = {"id": asset["id"], "path": str(path), "kind": asset["kind"],
               "rights": asset["rights"], "redistributable": bool(asset["redistributable"]),
               "limit": limit, "file_bytes": path.stat().st_size, "accepted": False}
        if not path.is_file():
            row["refusal"] = "missing"
            entries.append(row)
            continue
        data, accepted, refusal, _ = classify(path, limit)
        row["sha256"] = digest(data) if len(data) == row["file_bytes"] else None
        if not accepted:
            row["refusal"] = refusal
        else:
            row["accepted"] = True
            row.update(measure(data, limit, repeats))
        fovea = asset.get("fovea_path")
        if fovea and Path(fovea).is_file():
            reference = Path(fovea)
            row["fovea"] = {"path": str(reference), "bytes": reference.stat().st_size,
                            "sha256": digest(reference.read_bytes()),
                            "lossy": True, "note": "quantised .fovea container, not byte-exact"}
            if row["accepted"]:
                row["fovea_ratio_of_archive"] = round(reference.stat().st_size / row["packet_bytes"], 4)
        entries.append(row)
        print(f"{row['id']}: {'accepted' if row['accepted'] else 'refused'}", flush=True)

    legacy = None
    if legacy_packet:
        path = Path(legacy_packet)
        packet = path.read_bytes()
        expected_sha = hashlib.sha256(decode_ply(packet, max_output_bytes=BUFFER)).hexdigest()
        legacy = {"path": str(path), "packet_bytes": len(packet),
                  "decoded_sha256": expected_sha, "decodes_under_v2": True,
                  "sha256": digest(packet)}

    root = Path(__file__).resolve().parents[1]
    tracked = [Path(__file__).resolve(), root / "bellium/compression/ply_archive.py",
               root / "bellium/compression/container.py",
               root / "benchmarks/protocols/ply-coverage-v2.md"]
    report = {
        "schema": "bellium-ply-coverage/1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "zlib": zlib.ZLIB_VERSION, "repeats": repeats,
        "manifest_sha256": digest(json.dumps(manifest, sort_keys=True).encode()),
        "code_sha256": {path.relative_to(root).as_posix(): digest(path.read_bytes())
                        for path in tracked},
        "accepted": sum(row["accepted"] for row in entries),
        "refused": sum(not row["accepted"] for row in entries),
        "all_accepted_exact": all(row.get("exact", False) for row in entries if row["accepted"]),
        "legacy_v1_packet": legacy,
        "assets": entries,
        "limitations": [
            "Local corpus, not a population sample; several assets come from the same project",
            ".fovea is lossy and is reported beside the archive, never as an equivalent",
            "Rights-restricted assets are measured locally and must not be redistributed",
            "No Godot, Zig, GPU or VR measurement; no .fovea writer",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output), "accepted": report["accepted"],
                      "refused": report["refused"],
                      "all_exact": report["all_accepted_exact"],
                      "legacy_v1_ok": bool(legacy)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--legacy-packet", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 5:
        parser.error("repeats must be in 1..5")
    run(json.loads(args.manifest.read_text(encoding="utf-8")), args.output,
        args.legacy_packet, args.repeats)


if __name__ == "__main__":
    main()
