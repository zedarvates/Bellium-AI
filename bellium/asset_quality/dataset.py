"""Labeled matte fixtures and group-disjoint dataset loading.

Real datasets stay local. Paths resolve relative to their manifest, never a URL.
The fixture label is determined by the applied transformation, not by a predictor.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from .features import extract_features, read_asset
from .models import digest

DATASET_SCHEMA = "bellium.alpha-dataset/v1"


def _row(image: Image.Image, *, identity: str, group: str, split: str,
         label: str, reason: str, source_sha256: str | None = None) -> dict:
    started = time.perf_counter()
    features = extract_features(image)
    elapsed = (time.perf_counter() - started) * 1000
    rgba = image.convert("RGBA")
    pixel_digest = hashlib.sha256(str(rgba.size).encode() + rgba.tobytes()).hexdigest()
    return {"id": identity, "group": group, "split": split, "label": label,
            "reason": reason, "sha256": pixel_digest, "source_sha256": source_sha256,
            "features": features, "feature_ms": elapsed}


def synthetic_rows(seed: int = 17) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for index in range(60):
        # Every derivative of one original belongs to the same partition.
        split = "train" if index < 40 else "validation" if index < 50 else "test"
        group = f"fixture-{index:03}"
        color = tuple(rng.randrange(25, 230) for _ in range(3))
        base = Image.new("RGBA", (64, 64), (*color, 0))
        draw = ImageDraw.Draw(base)
        left, top = rng.randrange(10, 19), rng.randrange(9, 19)
        right, bottom = rng.randrange(44, 56), rng.randrange(44, 56)
        if index % 3 == 0:
            draw.ellipse((left, top, right, bottom), fill=(*color, 255))
        elif index % 3 == 1:
            draw.rectangle((left, top, right, bottom), fill=(*color, 255))
        else:
            draw.polygon(((left, 32), (32, top), (right, 32), (32, bottom)), fill=(*color, 255))
        feathered = base.copy()
        feathered.putalpha(base.getchannel("A").filter(ImageFilter.GaussianBlur(0.5)))
        clipped = Image.new("RGBA", base.size, (*color, 0))
        clipped.paste(base, (-left - 5, 0))
        faded = base.copy()
        faded.putalpha(base.getchannel("A").point(lambda alpha: 160 if alpha else 0))
        tiny = Image.new("RGBA", base.size, (*color, 0))
        tiny.paste(base.resize((6, 6), Image.Resampling.NEAREST), (29, 29))
        blank = Image.new("RGBA", base.size, (*color, 0))
        for reason, image, label in (
            ("clean", base, "candidate"), ("edge_feather", feathered, "candidate"),
            ("clipped", clipped, "review"), ("broad_fade", faded, "review"),
            ("tiny", tiny, "review"), ("blank", blank, "review"),
        ):
            rows.append(_row(image, identity=f"{group}-{reason}", group=group,
                             split=split, label=label, reason=reason))
    # Separate stress partition. It is not used for fitting or threshold selection.
    for index in range(12):
        color = tuple(rng.randrange(256) for _ in range(3))
        image = Image.new("RGBA", (64, 64), (*color, 255))
        reason = "opaque_canvas"
        if index % 2:
            reason = "scattered_alpha"
            draw = ImageDraw.Draw(image)
            for y in range(0, 64, 4):
                for x in range(0, 64, 4):
                    draw.rectangle((x, y, x + 1, y + 1), fill=(*color, 0))
        rows.append(_row(image, identity=f"stress-{index}", group=f"stress-{index}",
                         split="ood", label="review", reason=reason))
    validate_splits(rows)
    return rows


def validate_splits(rows: list[dict]) -> None:
    if not rows or len(rows) > 5000:
        raise ValueError("Dataset must contain 1..5000 rows")
    groups, hashes, ids = {}, {}, set()
    for row in rows:
        if row["split"] not in ("train", "validation", "test", "ood"):
            raise ValueError("Unknown dataset split")
        if row["label"] not in ("candidate", "review"):
            raise ValueError("Unknown dataset label")
        if row["id"] in ids:
            raise ValueError("Duplicate case ID")
        ids.add(row["id"])
        previous = groups.setdefault(row["group"], row["split"])
        if previous != row["split"]:
            raise ValueError("Source group leaks between splits")
        previous_hash = hashes.setdefault(row["sha256"], (row["split"], row["label"]))
        if previous_hash[0] != row["split"]:
            raise ValueError("Identical decoded image leaks between splits")
        if previous_hash[1] != row["label"]:
            raise ValueError("Conflicting labels for the same decoded image")
    for split in ("train", "validation", "test"):
        if {r["label"] for r in rows if r["split"] == split} != {"candidate", "review"}:
            raise ValueError(f"{split} requires both labels")


def load_dataset(path: str | Path) -> tuple[list[dict], str]:
    manifest = Path(path).resolve()
    if manifest.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("Dataset manifest exceeds 8 MiB")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if (not isinstance(data, dict) or set(data) != {"schema", "provenance", "cases"}
            or data["schema"] != DATASET_SCHEMA):
        raise ValueError("Invalid dataset schema or unknown fields")
    if not isinstance(data["provenance"], str) or not 1 <= len(data["provenance"]) <= 1000:
        raise ValueError("Dataset must identify label provenance")
    if not isinstance(data["cases"], list) or not 1 <= len(data["cases"]) <= 5000:
        raise ValueError("Dataset must contain 1..5000 cases")
    rows = []
    for item in data["cases"]:
        if (not isinstance(item, dict)
                or set(item) != {"id", "group", "split", "label", "path", "reason"}
                or any(not isinstance(v, str) or not 1 <= len(v) <= 512 for v in item.values())):
            raise ValueError("Invalid dataset case or unknown fields")
        relative = Path(item["path"])
        source = (manifest.parent / relative).resolve()
        if relative.is_absolute() or not source.is_relative_to(manifest.parent):
            raise ValueError("Asset path must stay within the dataset directory")
        image, file_sha, _ = read_asset(source)
        rows.append(_row(image, identity=item["id"], group=item["group"], split=item["split"],
                         label=item["label"], reason=item["reason"], source_sha256=file_sha))
    validate_splits(rows)
    return rows, data["provenance"]


def dataset_digest(rows: list[dict]) -> str:
    return digest([{k: v for k, v in row.items() if k != "feature_ms"} for row in rows])
