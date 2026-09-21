"""Rasterize atlas pages and hand back exact pixels or exact PNG bytes.

The specialist composes the plan it is given, verifies its own pages against the
plan and the sources, and returns either pixels or encoded PNG bytes. Nothing is
written to disk, no engine is invoked, and a page that fails its own check is
reported instead of returned.
"""

from __future__ import annotations

import hashlib
from typing import Any

from bellium.atlas.packing import check_plan, pack, parse_frames
from bellium.atlas.png import decode_png, encode_pages, inspect_png
from bellium.atlas.raster import check_raster, check_sources, rasterize
from bellium.contracts.schema import AuthorityMode, SpecialistResult

SPECIALIST_ID = "bellium/hybrid/atlas-raster:v0"
QUERY_KEYS = (
    "frames",
    "sources",
    "width",
    "height",
    "method",
    "padding",
    "max_pages",
    "background",
    "encode",
)
ENCODINGS = ("none", "png")
READY_CONFIDENCE = 0.9


def rasterize_atlas(query: dict[str, Any]) -> SpecialistResult:
    if not isinstance(query, dict):
        raise ValueError("query must be an object")
    unknown = sorted(set(query) - set(QUERY_KEYS))
    if unknown:
        raise ValueError(f"unknown query keys refuse to be ignored: {', '.join(unknown)}")
    encoding = query.get("encode", "none")
    if encoding not in ENCODINGS:
        raise ValueError(f"encode must be one of: {', '.join(ENCODINGS)}")
    frames = parse_frames(query.get("frames"))
    if "sources" not in query:
        raise ValueError("query needs sources")
    sources = query["sources"]
    if not isinstance(sources, dict):
        raise ValueError("sources must be an object mapping names to images")
    plan = pack(
        frames,
        width=query.get("width"),
        height=query.get("height"),
        method=query.get("method", "skyline"),
        padding=query.get("padding", 0),
        max_pages=query.get("max_pages", 1),
    )
    shared: dict[str, Any] = {
        "encode": encoding,
        "page_count": plan.page_count,
        "page_width": plan.page_width,
        "page_height": plan.page_height,
        "padding": plan.padding,
        "complete": plan.complete,
        "written_files": False,
        "certified": False,
    }
    if not plan.complete:
        return SpecialistResult(
            SPECIALIST_ID,
            {**shared, "status": "abstain", "reason": "frames_do_not_fit",
             "unplaced": [{"name": item.name, "reason": item.reason} for item in plan.unplaced]},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Nothing is rasterized while frames are unplaced.",),
        )
    violations = check_plan(frames, plan)
    if violations:
        return SpecialistResult(
            SPECIALIST_ID,
            {**shared, "status": "abstain", "reason": "plan_failed_verification",
             "violations": violations},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("A plan that fails its own invariants is never rasterized.",),
        )
    problems = check_sources(frames, sources)
    if problems:
        return SpecialistResult(
            SPECIALIST_ID,
            {**shared, "status": "abstain", "reason": "source_contract_violated",
             "source_violations": problems},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=(
                "Sources must match the declared frames exactly; nothing is scaled, "
                "cropped or coerced.",
            ),
        )
    raster = rasterize(frames, plan, sources, background=query.get("background"))
    mismatches = check_raster(frames, plan, raster, sources)
    if mismatches:
        return SpecialistResult(
            SPECIALIST_ID,
            {**shared, "status": "abstain", "reason": "raster_failed_verification",
             "raster_violations": mismatches},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Pages are re-derived and compared before they are returned.",),
        )
    shared.update({
        "channels": raster.channels,
        "background": list(raster.background),
        "placed": raster.placed,
        "raw_bytes": raster.bytes_written,
    })
    if encoding == "none":
        return SpecialistResult(
            SPECIALIST_ID,
            {**shared, "status": "ready", "reason": None, "pages": raster.to_dict()["pages"]},
            READY_CONFIDENCE,
            False,
            AuthorityMode.CONSULTATIVE,
            notes=(
                "Exact pixels: a copy inside every placement, the declared background elsewhere.",
                "Nothing was written to disk; the caller owns the file writing.",
            ),
        )
    encoded = encode_pages(raster)
    png: list[dict[str, Any]] = []
    for index, data in enumerate(encoded):
        info = inspect_png(data)
        if not info.ok:
            return SpecialistResult(
                SPECIALIST_ID,
                {**shared, "status": "abstain", "reason": "png_failed_verification",
                 "png_violations": list(info.violations)},
                0.0, True, AuthorityMode.CONSULTATIVE,
                notes=("An encoded page that fails its own inspection is never returned.",),
            )
        if decode_png(data).digest != raster.pages[index].digest:
            return SpecialistResult(
                SPECIALIST_ID,
                {**shared, "status": "abstain", "reason": "png_failed_verification",
                 "png_violations": [f"page {index} does not decode back to its pixels"]},
                0.0, True, AuthorityMode.CONSULTATIVE,
                notes=("The encoded page must decode back to the raster it came from.",),
            )
        png.append({
            "index": index,
            "width": info.width,
            "height": info.height,
            "channels": info.channels,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        })
    return SpecialistResult(
        SPECIALIST_ID,
        {**shared, "status": "ready", "reason": None, "png": png, "page_bytes": list(encoded)},
        READY_CONFIDENCE,
        False,
        AuthorityMode.CONSULTATIVE,
        notes=(
            "Encoded pages are 8-bit RGB or RGBA PNG, filter 0, no interlacing.",
            "The caller writes the bytes; this specialist touches no file.",
        ),
    )
