"""Download public sources and prepare a hashed, source-disjoint local corpus.

No source assets are added to the Python package. Requires Pillow. The local
Fovea asset is read-only; its 14-field adaptation is explicitly not a PLY archive.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct
import urllib.request

from PIL import Image

HORSE_SHA256 = "4392920074f41204ec900244da06505a8696312ea989340df88b1bd9c025806a"
DATA_DOC = "https://scikit-image.org/docs/stable/api/skimage.data.html"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fetch(url, limit=8 * 1024 * 1024):
    request = urllib.request.Request(url, headers={"User-Agent": "BelliumCompressionResearch/1"})
    with urllib.request.urlopen(request, timeout=45) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError("source exceeds download budget")
    return data


def canonical_splats(ply, start, count=512):
    marker = b"end_header\n"
    stop = ply.find(marker, 0, 16384)
    if stop < 0:
        raise ValueError("missing bounded PLY header")
    header = ply[:stop].decode("ascii").splitlines()
    if header[:2] != ["ply", "format binary_little_endian 1.0"]:
        raise ValueError("requires binary little-endian PLY")
    vertices, properties = None, []
    for line in header[2:]:
        words = line.split()
        if words[:2] == ["element", "vertex"] and vertices is None:
            vertices = int(words[2])
        elif words[:2] == ["property", "float"] and len(words) == 3:
            properties.append(words[2])
        elif words and words[0] != "comment":
            raise ValueError("unsupported PLY element or property")
    if len(set(properties)) != len(properties) or not properties or vertices is None:
        raise ValueError("invalid PLY schema")
    stride = len(properties) * 4
    offset = stop + len(marker)
    if len(ply) != offset + vertices * stride or not 0 <= start <= vertices - count:
        raise ValueError("invalid PLY count, size or sample interval")
    required = ["x", "y", "z", *(f"rot_{i}" for i in range(4)),
                *(f"scale_{i}" for i in range(3)), "opacity", *(f"f_dc_{i}" for i in range(3))]
    if not set(required) <= set(properties):
        raise ValueError("missing Gaussian fields")
    output = bytearray()
    for index in range(start, start + count):
        values = dict(zip(properties, struct.unpack_from(f"<{len(properties)}f", ply,
                                                         offset + index * stride)))
        if not all(math.isfinite(values[name]) for name in required):
            raise ValueError("non-finite Gaussian fields")
        q = [values[f"rot_{i}"] for i in (1, 2, 3, 0)]
        norm = math.sqrt(sum(value * value for value in q))
        if norm < 1e-12 or any(abs(values[f"scale_{i}"]) > 80 for i in range(3)):
            raise ValueError("invalid Gaussian rotation or scale")
        opacity = values["opacity"]
        opacity = (1 / (1 + math.exp(-opacity)) if opacity >= 0 else
                   math.exp(opacity) / (1 + math.exp(opacity)))
        fields = [values[name] for name in ("x", "y", "z")]
        fields += [value / norm for value in q]
        fields += [math.exp(values[f"scale_{i}"]) for i in range(3)]
        fields += [opacity]
        fields += [max(0, min(1, .5 + .28209479177387814 * values[f"f_dc_{i}"]))
                   for i in range(3)]
        output.extend(struct.pack("<14f", *fields))
    return bytes(output), vertices


def prepare(destination, horse_ply):
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "sources").mkdir()
    (destination / "samples").mkdir()
    sources, samples = [], []

    def source(identifier, data, **info):
        path = f"sources/{identifier}"
        (destination / path).write_bytes(data)
        sources.append({"id": identifier, "path": path, "sha256": sha(data), **info})

    def sample(identifier, data, source_id, split, kind, **info):
        path = f"samples/{identifier}.bin"
        (destination / path).write_bytes(data)
        samples.append({"id": identifier, "path": path, "sha256": sha(data),
                        "bytes": len(data), "source_id": source_id, "split": split,
                        "kind": kind, **info})

    for name, split, author in (("camera", "development", "Lav Varshney"),
                                ("coffee", "development", "Rachel Michetti"),
                                ("astronaut", "holdout", "NASA"),
                                ("chelsea", "holdout", "Stefan van der Walt")):
        url = f"https://raw.githubusercontent.com/scikit-image/scikit-image/v0.25.2/skimage/data/{name}.png"
        data = fetch(url)
        filename = f"{name}.png"
        source(filename, data, url=url, license="Public domain" if name == "astronaut" else "CC0",
               license_evidence=DATA_DOC + f"#skimage.data.{name}", author=author,
               classification="photograph", split=split)
        with Image.open(destination / "sources" / filename) as image:
            if image.mode not in ("L", "RGB"):
                raise ValueError("unexpected source image mode")
            w, h = image.size
            positions = ((0, 0), ((w - 96) // 2, (h - 96) // 2), (w - 96, h - 96))
            for index, (x, y) in enumerate(positions):
                crop = image.crop((x, y, x + 96, y + 96))
                sample(f"{name}-{index}", crop.tobytes(), filename, split, "image",
                       width=96, height=96, mode=image.mode, crop_xy=[x, y])
        print(f"prepared image {name}", flush=True)

    commit_info = json.loads(fetch("https://api.github.com/repos/davidmegginson/ourairports-data/commits?per_page=1"))
    commit = commit_info[0]["sha"]
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ValueError("invalid source revision")
    for name, split in (("countries", "development"), ("regions", "holdout")):
        url = f"https://raw.githubusercontent.com/davidmegginson/ourairports-data/{commit}/{name}.csv"
        data = fetch(url)
        source(f"{name}.csv", data, url=url, revision=commit, license="Public domain",
               license_evidence="https://ourairports.com/data/", author="OurAirports contributors",
               classification="public_tabular_data", split=split)
        positions = (0, len(data) - 8192) if split == "development" else (
            0, (len(data) - 8192) // 2, len(data) - 8192)
        if any(pos < 0 for pos in positions):
            raise ValueError("source too small")
        for index, pos in enumerate(positions):
            sample(f"{name}-{index}", data[pos:pos + 8192], f"{name}.csv", split,
                   "stream", byte_offset=pos)
        print(f"prepared stream {name}", flush=True)

    ply = horse_ply.read_bytes()
    if sha(ply) != HORSE_SHA256:
        raise ValueError("local splat source differs from the reviewed provenance")
    proof_path = horse_ply.parent / "runtime_proof_manifest_v1.json"
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    if (proof["source"]["source_model"]["license"] != "CC0"
            or proof["artifacts"]["runtime_ply"]["sha256"] != HORSE_SHA256):
        raise ValueError("splat provenance does not match")
    source("horse.ply", ply, url="https://polyhaven.com/a/horse_statue_01",
           license="CC0 source; local derived reconstruction",
           license_evidence="https://polyhaven.com/license", author="Rico Cilliers / local reconstruction",
           classification="trained_splats_from_synthetic_multiview_video", split="holdout",
           provenance_manifest_sha256=sha(proof_path.read_bytes()))
    _, count = canonical_splats(ply, 0)
    for index, pos in enumerate((0, (count - 512) // 2, count - 512)):
        records, _ = canonical_splats(ply, pos)
        sample(f"horse-{index}", records, "horse.ply", "holdout", "splats", record_offset=pos,
               record_count=512, full_ply_preserved=False,
               adaptation="14 f32 physical fields; normalized xyzw; exp(scale), sigmoid(opacity), clamped SH DC RGB; higher SH dropped")
    manifest = {"schema": "bellium-independent-compression-corpus/1",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "sources": sources, "samples": samples,
                "limitations": ["No natural splat motion", "One synthetic-video-derived splat scene",
                                "Losslessness measured after the explicitly lossy PLY adaptation",
                                "Small fixed crops/chunks; source-disjoint split, not population representativeness"]}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"corpus": str(destination), "sources": len(sources), "samples": len(samples)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--horse-ply", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.output, args.horse_ply)


if __name__ == "__main__":
    main()
