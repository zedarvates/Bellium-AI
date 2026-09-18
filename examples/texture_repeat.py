"""Texture repetition example: analyze an image or run synthetic demo cases."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random

from PIL import Image

from bellium.texture import TextureError, detect_repeat


def analyze_texture(path: str | Path, *, axis: str = "both", **options) -> dict:
    """Callable tool adapter: returns a JSON-serializable repeat report."""
    with Image.open(path) as opened:
        image = opened.copy()
    report = detect_repeat(image, axis=axis, **options)
    return {"input": str(path), **report.to_dict()}


def synthetic_cases() -> dict[str, Image.Image]:
    """Original in-memory fixtures; no files, downloads or learned models."""
    board = Image.new("L", (64, 64), 0)
    for y in range(64):
        for x in range(64):
            if ((x // 8) + (y // 8)) % 2 == 0:
                board.putpixel((x, y), 255)
    stripes = Image.new("L", (60, 40))
    stripes.putdata([
        int(round(127.5 + 127.5 * math.sin(2.0 * math.pi * x / 10.0)))
        for _y in range(40)
        for x in range(60)
    ])
    rng = random.Random(2026)
    noise = Image.new("L", (64, 64))
    noise.putdata([rng.randrange(256) for _ in range(64 * 64)])
    return {
        "checkerboard_8px_cells": board,
        "sine_stripes_10px": stripes,
        "noise": noise,
    }


def demo_report() -> dict:
    """Analyze every synthetic case and return the reports."""
    return {name: detect_repeat(image).to_dict() for name, image in synthetic_cases().items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    sub.add_parser("demo", help="Analyze built-in synthetic textures")
    analyze = sub.add_parser("analyze", help="Analyze an image file")
    analyze.add_argument("--input", required=True)
    analyze.add_argument("--axis", choices=["both", "x", "y"], default="both")
    analyze.add_argument("--min-period", type=int, default=2)
    analyze.add_argument("--max-period", type=int)
    analyze.add_argument("--min-match", type=float, default=0.8)
    analyze.add_argument("--min-contrast", type=float, default=0.05)
    analyze.add_argument("--max-analysis", type=int, default=256)
    args = parser.parse_args(argv)
    try:
        if args.operation == "demo":
            print(json.dumps(demo_report(), indent=2))
            return 0
        report = analyze_texture(
            args.input,
            axis=args.axis,
            min_period=args.min_period,
            max_period=args.max_period,
            min_match=args.min_match,
            min_contrast=args.min_contrast,
            max_analysis=args.max_analysis,
        )
    except (OSError, TextureError) as exc:
        print(json.dumps({"status": "error", "reason": str(exc)}))
        return 1
    print(json.dumps(report, indent=2))
    return 0 if report["reliable"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
