"""Consultative 2D drafting document: validate, emit SVG and DXF, round-trip."""

from __future__ import annotations

from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.drafting.document import (
    UnsupportedDrafting,
    check_drawing,
    drawing_from_dict,
    drawing_to_dict,
    drawings_match,
)
from bellium.drafting.dxf import drawing_from_dxf, drawing_to_dxf
from bellium.drafting.svg import drawing_from_svg, drawing_to_svg

SPECIALIST_ID = "bellium/deterministic/drafting-document:v0"
QUERY_KEYS = {"drawing", "svg", "dxf"}
READY_CONFIDENCE = 0.9


def _notes(*lines: str) -> tuple[str, ...]:
    return lines


def inspect_drawing(query: dict[str, Any]) -> SpecialistResult:
    if not isinstance(query, dict):
        raise ValueError("query must be an object")
    unknown = sorted(set(query) - QUERY_KEYS)
    if unknown:
        raise ValueError(f"unknown query keys refuse to be ignored: {', '.join(unknown)}")
    present = [key for key in ("drawing", "svg", "dxf") if query.get(key) is not None]
    if len(present) != 1:
        raise ValueError("query needs exactly one of drawing, svg or dxf")
    source = present[0]
    try:
        if source == "drawing":
            drawing = drawing_from_dict(query["drawing"])
        elif source == "svg":
            drawing = drawing_from_svg(query["svg"])
        else:
            drawing = drawing_from_dxf(query["dxf"])
    except UnsupportedDrafting as error:
        return SpecialistResult(
            SPECIALIST_ID,
            {
                "status": "abstain",
                "reason": error.reason,
                "details": list(error.details),
                "certified": False,
                "pixels_written": False,
            },
            0.0,
            True,
            AuthorityMode.CONSULTATIVE,
            notes=_notes("Unsupported geometry is reported, never approximated."),
        )
    violations = check_drawing(drawing)
    shared = {
        "drawing": drawing_to_dict(drawing),
        "entity_count": len(drawing.entities),
        "layer_count": len(drawing.layers),
        "unit": drawing.unit,
        "source": source,
        "certified": False,
        "pixels_written": False,
    }
    if violations:
        return SpecialistResult(
            SPECIALIST_ID,
            {**shared, "status": "abstain", "reason": "drawing_failed_verification",
             "violations": violations},
            0.0,
            True,
            AuthorityMode.CONSULTATIVE,
            notes=_notes("A drawing that fails its own invariants is not emitted."),
        )
    if not drawing.entities:
        return SpecialistResult(
            SPECIALIST_ID,
            {**shared, "status": "abstain", "reason": "empty_drawing"},
            0.0,
            True,
            AuthorityMode.CONSULTATIVE,
            notes=_notes("An empty sheet is valid paper, not a drawing."),
        )
    svg = drawing_to_svg(drawing)
    dxf = drawing_to_dxf(drawing)
    svg_ok = drawings_match(drawing, drawing_from_svg(svg))
    dxf_ok = drawings_match(drawing, drawing_from_dxf(dxf))
    if not svg_ok or not dxf_ok:
        return SpecialistResult(
            SPECIALIST_ID,
            {
                **shared,
                "status": "abstain",
                "reason": "roundtrip_failed",
                "svg_roundtrip": svg_ok,
                "dxf_roundtrip": dxf_ok,
            },
            0.0,
            True,
            AuthorityMode.CONSULTATIVE,
            notes=_notes("Emit is refused when SVG or DXF cannot reconstruct the document."),
        )
    return SpecialistResult(
        SPECIALIST_ID,
        {
            **shared,
            "status": "ready",
            "reason": None,
            "svg": svg,
            "dxf": dxf,
            "svg_roundtrip": True,
            "dxf_roundtrip": True,
        },
        READY_CONFIDENCE,
        False,
        AuthorityMode.CONSULTATIVE,
        notes=_notes(
            "Deterministic 2D document only: line, polyline, circle, arc, text.",
            "Y-up, paper origin bottom-left. No pixels, no offset, no cotation.",
        ),
    )
