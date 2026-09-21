"""Remove exact duplicate frames before packing: one stored copy, aliases for the rest.

The specialist analyses the declared frames, proves the alias map, packs only the
distinct images, rasterizes that atlas and optionally encodes it. An animation keeps
its frame list and the stored atlas is smaller. Merging is exact unless the caller
declares a max_delta bound, and then the worst measured difference is reported.
"""

from __future__ import annotations

import hashlib
from typing import Any

from bellium.atlas.dedup import analyze_frames, bound, check_aliases, unique_frames
from bellium.atlas.packing import check_plan, pack, parse_frames
from bellium.atlas.png import decode_png, encode_pages, inspect_png
from bellium.atlas.raster import check_raster, check_sources, rasterize
from bellium.contracts.schema import AuthorityMode, SpecialistResult

SPECIALIST_ID = "bellium/hybrid/atlas-dedup:v0"
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
    "max_delta",
)
ENCODINGS = ("none", "png")
READY_CONFIDENCE = 0.9


def dedup_atlas(query: dict[str, Any]) -> SpecialistResult:
    if not isinstance(query, dict):
        raise ValueError("query must be an object")
    unknown = sorted(set(query) - set(QUERY_KEYS))
    if unknown:
        raise ValueError(f"unknown query keys refuse to be ignored: {', '.join(unknown)}")
    encoding = query.get("encode", "none")
    if encoding not in ENCODINGS:
        raise ValueError(f"encode must be one of: {', '.join(ENCODINGS)}")
    frames = parse_frames(query.get("frames"))
    tolerance = bound(query.get("max_delta", 0))
    if "sources" not in query:
        raise ValueError("query needs sources")
    sources = query["sources"]
    if not isinstance(sources, dict):
        raise ValueError("sources must be an object mapping names to images")
    shared: dict[str, Any] = {
        "encode": encoding,
        "frames_declared": len(frames),
        "max_delta": tolerance,
        "written_files": False,
        "certified": False,
    }
    problems = check_sources(frames, sources)
    if problems:
        return SpecialistResult(
            SPECIALIST_ID,
            {**shared, "status": "abstain", "reason": "source_contract_violated",
             "source_violations": problems},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Duplicates are only removed between frames whose sources match the declaration.",),
        )
    report = analyze_frames(frames, sources, max_delta=tolerance)
    alias_violations = check_aliases(frames, sources, report)
    if alias_violations:
        return SpecialistResult(
            SPECIALIST_ID,
            {**shared, "status": "abstain", "reason": "alias_failed_verification",
             "alias_violations": alias_violations},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("An alias map that does not reproduce the declared pixels is never used.",),
        )
    stored = unique_frames(frames, report)
    stored_sources = {frame.name: sources[frame.name] for frame in stored}
    plan = pack(
        stored,
        width=query.get("width"),
        height=query.get("height"),
        method=query.get("method", "skyline"),
        padding=query.get("padding", 0),
        max_pages=query.get("max_pages", 1),
    )
    shared.update({
        **report.to_dict(),
        "frames_stored": len(stored),
        "page_count": plan.page_count,
        "page_width": plan.page_width,
        "page_height": plan.page_height,
        "padding": plan.padding,
        "occupancy": round(plan.occupancy, 4),
        "complete": plan.complete,
    })
    if not plan.complete:
        return SpecialistResult(
            SPECIALIST_ID,
            {**shared, "status": "abstain", "reason": "frames_do_not_fit",
             "unplaced": [{"name": item.name, "reason": item.reason} for item in plan.unplaced]},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Distinct frames that do not fit are reported unplaced.",),
        )
    violations = check_plan(stored, plan)
    if violations:
        return SpecialistResult(
            SPECIALIST_ID,
            {**shared, "status": "abstain", "reason": "plan_failed_verification",
             "violations": violations},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("The plan over the distinct frames is re-checked before anything is drawn.",),
        )
    raster = rasterize(stored, plan, stored_sources, background=query.get("background"))
    mismatches = check_raster(stored, plan, raster, stored_sources)
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
        "page_digests": [page.digest for page in raster.pages],
        "placements": [
            {"name": item.name, "page": item.page, "x": item.x, "y": item.y,
             "width": item.width, "height": item.height}
            for page in plan.pages
            for item in page
        ],
    })
    if encoding == "none":
        notes = (
            (
                "Near duplicates were not merged: max_delta is 0, so only exact matches "
                "share a copy.",
                "Nothing was written to disk.",
            )
            if tolerance == 0
            else (
                f"Frames merged within the declared max_delta of {tolerance}; the worst "
                f"measured difference is {report.worst_delta}.",
                "Alpha counts in that bound, and nothing was written to disk.",
            )
        )
        return SpecialistResult(
            SPECIALIST_ID,
            {**shared, "status": "ready", "reason": None, "pages": raster.to_dict()["pages"]},
            READY_CONFIDENCE,
            False,
            AuthorityMode.CONSULTATIVE,
            notes=notes,
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
            "Aliased frames draw from one stored region; the animation list is unchanged.",
            "The caller writes the bytes; this specialist touches no file.",
        ),
    )
