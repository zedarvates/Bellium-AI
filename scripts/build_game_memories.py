#!/usr/bin/env python3
"""Build the game-oriented k-NN memories from synthetic authored fixtures.

The fixtures are generated here so every stored feature vector can be
recomputed. Labels are authored from how each fixture was built, not measured
on real game assets: the memories document asset-pipeline conventions, they do
not prove image quality. Existing files are preserved unless --force is passed.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bellium.knn._texture import seam_features, texture_features  # noqa: E402
from bellium.knn._texture import direction_scan  # noqa: E402
from bellium.knn._sheet import loop_features  # noqa: E402
from bellium.knn.consequence import (  # noqa: E402
    FEATURE_NAMES as CONSEQUENCE_FEATURES,
    deterministic_risk_label,
)
from bellium.knn.sprite_anchor import anchor_features  # noqa: E402
from bellium.knn.tileability import axis_features  # noqa: E402
from bellium.knn.texture_orientation import (  # noqa: E402
    deterministic_grain_verdict,
    orientation_features,
)
from bellium.knn._motion import REFERENCE_CURVES  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models" / "knn" / "visual"
LAYOUT_MODELS = ROOT / "models" / "knn" / "layout"
TOOL_MODELS = ROOT / "models" / "knn" / "tools"
MOTION_MODELS = ROOT / "models" / "knn" / "motion"
SOURCE = "fixture:game-lab-synthetic"
SIZE = 64
CLIP_CELL = 24

Image = list[list[tuple[int, int, int]]]
Mask = list[list[int]]


def gray(value: float) -> tuple[int, int, int]:
    level = max(0, min(255, int(value)))
    return level, level, level


def transpose(image: Image) -> Image:
    return [list(column) for column in zip(*image)]


def sine(period: int = 8, amplitude: float = 90.0) -> Image:
    return [
        [gray(128 + amplitude * math.sin(2 * math.pi * c / period)) for c in range(SIZE)]
        for _ in range(SIZE)
    ]


def wave2d(period: int = 16, amplitude: float = 70.0) -> Image:
    return [
        [
            gray(128 + amplitude * math.sin(2 * math.pi * c / period)
                 + amplitude * math.cos(2 * math.pi * r / period))
            for c in range(SIZE)
        ]
        for r in range(SIZE)
    ]


def checker(cell: int = 8, low: int = 60, high: int = 200) -> Image:
    return [
        [gray(high if ((r // cell) + (c // cell)) % 2 else low) for c in range(SIZE)]
        for r in range(SIZE)
    ]


def tiled_noise(tile: int = 16, seed: int = 3) -> Image:
    rng = random.Random(seed)
    block = [[gray(rng.randrange(0, 256)) for _ in range(tile)] for _ in range(tile)]
    return [[block[r % tile][c % tile] for c in range(SIZE)] for r in range(SIZE)]


def drifting_wave(step: float = 90.0, period: int = 16, amplitude: float = 40.0) -> Image:
    """Smooth field drifting by step across the width: only the wrap plane jumps."""
    return [
        [
            gray(120 + amplitude * math.sin(2 * math.pi * c / period) + step * (c / SIZE))
            for c in range(SIZE)
        ]
        for _ in range(SIZE)
    ]


def noise(seed: int = 4) -> Image:
    rng = random.Random(seed)
    return [[gray(rng.randrange(0, 256)) for _ in range(SIZE)] for _ in range(SIZE)]


def ramp() -> Image:
    return [[gray(40 + 200 * (c / SIZE)) for c in range(SIZE)] for _ in range(SIZE)]


def wrapped_step(step: int = 120) -> Image:
    """Smooth field whose wrap plane carries a hard, visible discontinuity."""
    return [
        [
            gray(90 + (step / 2 if c < SIZE // 2 else -step / 2)
                 + 6 * math.sin(2 * math.pi * c / 16))
            for c in range(SIZE)
        ]
        for _ in range(SIZE)
    ]


def flat(level: int = 180) -> Image:
    return [[gray(level) for _ in range(SIZE)] for _ in range(SIZE)]


def rotated_stripes(period: int = 12, angle_deg: float = 30.0,
                    amplitude: float = 100.0) -> Image:
    """Stripes whose repeat direction is not aligned with an image axis."""
    radians = math.radians(angle_deg)
    image = []
    for r in range(SIZE):
        row = []
        for c in range(SIZE):
            offset = c * math.cos(radians) + r * math.sin(radians)
            row.append(gray(128 + amplitude * math.sin(2 * math.pi * offset / period)))
        image.append(row)
    return image


def orientation_memory() -> dict:
    """Authored grains per family: three variants of each direction class."""
    per_family: dict[str, list[tuple[str, Image]]] = {
        "texture": (
            [("repeat-x", sine(period=period)) for period in (10, 14, 18)]
            + [("repeat-y", transpose(sine(period=period))) for period in (10, 14, 18)]
            + [("repeat-both", checker(cell=8)), ("repeat-both", wave2d(period=16)),
               ("repeat-both", tiled_noise(tile=16, seed=21))]
            + [("repeat-diagonal", rotated_stripes(angle_deg=angle)) for angle in (25.0, 40.0, 55.0)]
            + [("no-repeat", noise(seed=seed)) for seed in (71, 72)]
            + [("no-repeat", ramp())]
        ),
        "tilemap": (
            [("repeat-x", sine(period=period, amplitude=70.0)) for period in (8, 12, 16)]
            + [("repeat-y", transpose(sine(period=period, amplitude=70.0))) for period in (8, 12, 16)]
            + [("repeat-both", checker(cell=16)), ("repeat-both", wave2d(period=8)),
               ("repeat-both", tiled_noise(tile=8, seed=22))]
            + [("repeat-diagonal", rotated_stripes(angle_deg=angle, period=16))
               for angle in (30.0, 45.0, 60.0)]
            + [("no-repeat", noise(seed=seed)) for seed in (73, 74)]
            + [("no-repeat", wrapped_step(step=110))]
        ),
        "sprite": (
            [("repeat-x", sine(period=period, amplitude=60.0)) for period in (6, 9, 12)]
            + [("repeat-y", transpose(sine(period=period, amplitude=60.0))) for period in (6, 9, 12)]
            + [("repeat-both", checker(cell=4)), ("repeat-both", wave2d(period=12)),
               ("repeat-both", tiled_noise(tile=16, seed=23))]
            + [("repeat-diagonal", rotated_stripes(angle_deg=angle, period=9))
               for angle in (35.0, 50.0, 65.0)]
            + [("no-repeat", noise(seed=seed)) for seed in (75, 76)]
            + [("no-repeat", ramp())]
        ),
    }
    items = []
    for family, fixtures in per_family.items():
        for index, (grain, image) in enumerate(fixtures, start=1):
            scan = direction_scan(image, tuple(float(angle) for angle in range(0, 180, 15)))
            measured = deterministic_grain_verdict(scan)
            if measured != grain:
                raise SystemExit(
                    f"orientation fixture {family}/{index} is authored as {grain} "
                    f"but measured as {measured}; fix the fixture, not the label"
                )
            items.append({
                "id": f"to_{family}_{index:02d}_{grain}",
                "family": family,
                "grain": grain,
                "source": SOURCE,
                "raw_image_stored": False,
                "features": orientation_features(image, scan, grain),
            })
    return {
        "schema": "bellium.texture-orientation-memory/v1",
        "notes": "Authored grains. The published axis rule must agree at inference.",
        "items": items,
    }


def easing_memory() -> dict:
    """The published reference curves, per motion family."""
    items = []
    for family in ("ui", "character", "effect"):
        for label, curve in sorted(REFERENCE_CURVES.items()):
            items.append({
                "id": f"ea_{family}_{label}",
                "family": family,
                "label": label,
                "source": "reference-curve:bellium/_motion.py",
                "samples": [round(value, 6) for value in curve],
            })
    return {
        "schema": "bellium.easing-profile-memory/v1",
        "notes": (
            "Published reference curves only, no measurement. The threshold rule and the "
            "nearest curve must agree at inference."
        ),
        "items": items,
    }


def humanoid(scale: float = 1.0) -> Mask:
    mask = [[0] * SIZE for _ in range(SIZE)]
    half = int(round(9 * scale))
    top = int(round(20 - 14 * scale))
    bottom = int(round(20 + 14 * scale))
    head_bottom = int(round(20 - 6 * scale))
    for r in range(top, bottom):
        for c in range(32 - half // 2, 32 + half // 2):
            mask[r][c] = 1
    for r in range(top, head_bottom):
        for c in range(32 - half, 32 + half):
            mask[r][c] = 1
    arm_row = int(round(20 + 2 * scale))
    arm_half = int(round(16 * scale))
    for r in range(arm_row, arm_row + max(2, int(round(4 * scale)))):
        for c in range(32 - arm_half, 32 + arm_half):
            mask[r][c] = 1
    return mask


def diamond(radius: int = 6) -> Mask:
    return [
        [1 if abs(r - 32) + abs(c - 32) <= radius else 0 for c in range(SIZE)]
        for r in range(SIZE)
    ]


def disc(radius: int = 10) -> Mask:
    return [
        [1 if (r - 32) ** 2 + (c - 32) ** 2 <= radius * radius else 0 for c in range(SIZE)]
        for r in range(SIZE)
    ]


def crate(half: int = 8) -> Mask:
    return [
        [1 if 32 - half <= r < 32 + half and 32 - half <= c < 32 + half else 0
         for c in range(SIZE)]
        for r in range(SIZE)
    ]


def image_from_mask(mask: Mask) -> Image:
    return [[(210, 60, 60) if value else (255, 255, 255) for value in row] for row in mask]


def shape_mask(kind: str, size: int = CLIP_CELL, offset: int = 0) -> Mask:
    mask = [[0] * size for _ in range(size)]
    center = size // 2
    for r in range(size):
        for c in range(size):
            x = c - offset
            if kind == "square" and abs(r - center) <= 4 and abs(x - center) <= 4:
                mask[r][c] = 1
            elif kind == "circle" and (r - center) ** 2 + (x - center) ** 2 <= 25:
                mask[r][c] = 1
            elif kind == "diamond" and abs(r - center) + abs(x - center) <= 6:
                mask[r][c] = 1
            elif kind == "tall" and abs(r - center) <= 8 and abs(x - center) <= 2:
                mask[r][c] = 1
    return mask


def clip_loop_memory() -> dict:
    """Authored clip pairs: a clean loop repeats the first frame, a drift moves it."""
    families = (
        ("character", "tall"),
        ("effect", "diamond"),
        ("ui", "square"),
        ("prop", "circle"),
    )
    items = []
    for family, kind in families:
        for index, (verdict, offset, tint) in enumerate(
            (
                ("loops", 0, 0), ("loops", 0, 8), ("loops", 0, 16),
                ("drifts", 6, 0), ("drifts", 8, 0), ("drifts", 10, 8),
            ),
            start=1,
        ):
            first_mask = shape_mask(kind)
            last_mask = shape_mask(kind, offset=offset)
            first_image = image_from_mask(first_mask)
            last_image = [
                [
                    (min(255, pixel[0] + tint), pixel[1], pixel[2]) if mask_value else pixel
                    for pixel, mask_value in zip(row, mask_row)
                ]
                for row, mask_row in zip(image_from_mask(last_mask), last_mask)
            ]
            features = loop_features(
                (first_mask, first_image),
                (last_mask, last_image),
            )
            items.append({
                "id": f"cl_{family}_{index:02d}_{verdict}",
                "family": family,
                "verdict": verdict,
                "source": SOURCE,
                "raw_image_stored": False,
                "shift_px": offset,
                "tint": tint,
                "features": features,
            })
    return {
        "schema": "bellium.clip-loop-memory/v1",
        "notes": "Authored clip pairs. The baseline threshold must agree at inference.",
        "items": items,
    }


CONSEQUENCE_BASES = {
    "safe": {
        "reversible": 1.0, "scope_ratio": 0.15, "touches_protected": 0.0, "has_backup": 1.0,
        "dry_run_available": 1.0, "blast_radius": 0.10, "idempotent": 1.0, "recent_failures": 0.05,
    },
    "review": {
        # Kept far enough from both risk limits so the authored jitter cannot
        # move a fixture across a class boundary.
        "reversible": 1.0, "scope_ratio": 0.70, "touches_protected": 0.0, "has_backup": 1.0,
        "dry_run_available": 0.0, "blast_radius": 0.70, "idempotent": 0.30, "recent_failures": 0.50,
    },
    "dangerous": {
        "reversible": 0.0, "scope_ratio": 0.90, "touches_protected": 1.0, "has_backup": 0.0,
        "dry_run_available": 0.0, "blast_radius": 0.90, "idempotent": 0.0, "recent_failures": 0.60,
    },
}


def consequence_memory() -> dict:
    """Authored precedents labelled by the documented review rule."""
    families = ("filesystem", "asset-pipeline", "engine", "deployment", "network")
    items = []
    for family_index, family in enumerate(families):
        for label, base in CONSEQUENCE_BASES.items():
            for variant in range(4):
                jitter = 0.02 * (variant + 1) * (1 if (family_index + variant) % 2 else -1)
                features = {
                    name: round(min(1.0, max(0.0, base[name] + jitter)), 4)
                    for name in CONSEQUENCE_FEATURES
                }
                measured = deterministic_risk_label(features)
                if measured != label:
                    raise SystemExit(
                        f"fixture {family}/{label}/{variant} is labelled {measured}; "
                        "adjust the authored base vectors instead of the label"
                    )
                items.append({
                    "id": f"cp_{family}_{label}_{variant}",
                    "family": family,
                    "label": label,
                    "verified": True,
                    "source": "fixture:game-lab-authored-precedents",
                    "features": features,
                })
    return {
        "schema": "bellium.consequence-precedent-memory/v1",
        "notes": (
            "Authored synthetic precedents, not recorded incidents. Labels come from the "
            "published risk rule; verified means the fixture was constructed deliberately."
        ),
        "items": items,
    }


def repeat_memory() -> dict:
    """Each family needs at least three agreeing neighbors at inference."""
    fixtures: list[tuple[str, str, Image, int | None, int | None]] = []
    for period in (8, 12, 16, 24):
        # A constant-row profile has no measurable vertical period.
        fixtures.append(("periodic", "texture", sine(period=period), period, None))
    for seed in (4, 9, 14, 21):
        fixtures.append(("nonperiodic", "texture", noise(seed=seed), None, None))
    for tile in (8, 16, 32):
        fixtures.append(("periodic", "tilemap", tiled_noise(tile=tile, seed=tile), tile, tile))
    fixtures.append(("periodic", "tilemap", checker(cell=8), 8, 8))
    fixtures.append(("periodic", "tilemap", checker(cell=16), 16, 16))
    for seed in (31, 41, 51):
        fixtures.append(("nonperiodic", "tilemap", noise(seed=seed), None, None))
    fixtures.append(("nonperiodic", "tilemap", ramp(), None, None))
    for period in (8, 16, 24):
        fixtures.append(("periodic", "sprite", wave2d(period=period), period, period))
    fixtures.append(("periodic", "sprite", checker(cell=4), 4, 4))
    for step in (60, 90, 120):
        fixtures.append(("nonperiodic", "sprite", wrapped_step(step=step), None, None))
    fixtures.append(("nonperiodic", "sprite", noise(seed=61), None, None))
    items = []
    for index, (verdict, family, image, period_x, period_y) in enumerate(fixtures, start=1):
        items.append({
            "id": f"tr_{index:02d}_{family}_{verdict}",
            "family": family,
            "verdict": verdict,
            "period_x": period_x,
            "period_y": period_y,
            "source": SOURCE,
            "raw_image_stored": False,
            "features": texture_features(image),
        })
    return {
        "schema": "bellium.texture-repeat-memory/v1",
        "notes": "Authored synthetic fixtures. Rebuild with scripts/build_game_memories.py.",
        "items": items,
    }


def tileability_memory() -> dict:
    """Labels describe how each fixture meets itself on the wrap plane."""

    def family_fixtures(*, tile: int, seed: int, cell: int, period: int, step_a: float,
                        step_b: float, level: int) -> list[tuple[Image, str, str]]:
        tileable = [
            sine(period=period),
            wave2d(period=period),
            tiled_noise(tile=tile, seed=seed),
            checker(cell=cell),
            flat(level=level),
        ]
        # Each seam fixture jumps only on the wrap plane, never inside the field.
        seam_x = [
            ramp(),
            drifting_wave(step=step_a, period=period),
            drifting_wave(step=step_b, period=max(4, period // 2)),
            wrapped_step(step=int(step_a)),
        ]
        fixtures = [(image, "tileable", "tileable") for image in tileable]
        fixtures += [(image, "seam", "tileable") for image in seam_x]
        fixtures += [(transpose(image), "tileable", "seam") for image in seam_x]
        return fixtures

    per_family: dict[str, list[tuple[Image, str, str]]] = {
        "texture": family_fixtures(tile=16, seed=2, cell=8, period=16, step_a=90.0,
                                   step_b=140.0, level=180),
        "tilemap": family_fixtures(tile=8, seed=7, cell=16, period=8, step_a=110.0,
                                   step_b=160.0, level=140),
        "sprite": family_fixtures(tile=16, seed=10, cell=4, period=12, step_a=100.0,
                                  step_b=150.0, level=220),
    }
    items = []
    for family, fixtures in per_family.items():
        for index, (image, verdict_x, verdict_y) in enumerate(fixtures, start=1):
            seams = seam_features(image)
            for axis, verdict in (("x", verdict_x), ("y", verdict_y)):
                items.append({
                    "id": f"tl_{family}_{index:02d}_{axis}_{verdict}",
                    "family": family,
                    "axis": axis,
                    "verdict": verdict,
                    "source": SOURCE,
                    "raw_image_stored": False,
                    "features": axis_features(seams, axis),
                })
    return {
        "schema": "bellium.tileability-memory/v1",
        "notes": "Authored synthetic fixtures. The deterministic baseline must agree at inference.",
        "items": items,
    }


def anchor_memory() -> dict:
    fixtures: list[tuple[str, str, Mask, float, float]] = []
    for scale in (0.7, 0.85, 1.0, 1.15):
        fixtures.append(("character", "feet", humanoid(scale=scale), 0.5, 1.0))
    for radius in (4, 6, 8, 10):
        fixtures.append(("effect", "center", diamond(radius=radius), 0.5, 0.5))
    for radius in (8, 10, 12, 14):
        fixtures.append(("ui", "center", disc(radius=radius), 0.5, 0.5))
    for half in (6, 8, 9, 10):
        fixtures.append(("prop", "feet", crate(half=half), 0.5, 1.0))
    items = []
    for index, (family, anchor, mask, anchor_x, anchor_y) in enumerate(fixtures, start=1):
        items.append({
            "id": f"sa_{index:02d}_{family}_{anchor}",
            "family": family,
            "anchor": anchor,
            "anchor_x": anchor_x,
            "anchor_y": anchor_y,
            "source": SOURCE,
            "raw_image_stored": False,
            "features": anchor_features(mask),
        })
    return {
        "schema": "bellium.sprite-anchor-memory/v1",
        "notes": "Authored silhouettes. Anchors are content-box ratios, declared by hand.",
        "items": items,
    }


def write(path: Path, payload: dict[str, Any], *, force: bool) -> str:
    if path.exists() and not force:
        return f"kept {path.name}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return f"wrote {path.name} ({len(payload['items'])} items)"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="overwrite existing memories")
    options = parser.parse_args()
    print(write(MODELS / "texture-repeat-v0.json", repeat_memory(), force=options.force))
    print(write(MODELS / "tileability-v0.json", tileability_memory(), force=options.force))
    print(write(MODELS / "sprite-anchor-v0.json", anchor_memory(), force=options.force))
    print(write(LAYOUT_MODELS / "clip-loop-v0.json", clip_loop_memory(), force=options.force))
    print(write(TOOL_MODELS / "consequence-precedents-v0.json", consequence_memory(),
                force=options.force))
    print(write(MODELS / "texture-orientation-v0.json", orientation_memory(),
                force=options.force))
    print(write(MOTION_MODELS / "easing-profiles-v0.json", easing_memory(),
                force=options.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
