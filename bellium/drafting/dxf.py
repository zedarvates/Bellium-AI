"""DXF R12 ASCII subset for Bellium drafting documents."""

from __future__ import annotations

from bellium.drafting.document import (
    Arc,
    Circle,
    Drawing,
    Line,
    Text,
    UnsupportedDrafting,
    drawing_from_dict,
)

INSUNITS = {"in": 1, "mm": 4}
UNITS_FROM_INS = {1: "in", 4: "mm"}
SUPPORTED = {"LINE", "CIRCLE", "ARC", "TEXT", "POLYLINE", "VERTEX", "SEQEND"}
SKIP = {
    "SECTION",
    "ENDSEC",
    "TABLE",
    "ENDTAB",
    "LAYER",
    "EOF",
    "HEADER",
    "TABLES",
    "ENTITIES",
}


def _fmt(value: float) -> str:
    return format(value, ".12g")


def _pair(code: int, value: object) -> list[str]:
    return [str(code), str(value)]


def drawing_to_dxf(drawing: Drawing) -> str:
    lines: list[str] = []
    lines.extend(_pair(0, "SECTION"))
    lines.extend(_pair(2, "HEADER"))
    lines.extend(_pair(9, "$ACADVER"))
    lines.extend(_pair(1, "AC1009"))
    lines.extend(_pair(9, "$INSUNITS"))
    lines.extend(_pair(70, INSUNITS[drawing.unit]))
    lines.extend(_pair(9, "$LIMMIN"))
    lines.extend(_pair(10, "0"))
    lines.extend(_pair(20, "0"))
    lines.extend(_pair(9, "$LIMMAX"))
    lines.extend(_pair(10, _fmt(drawing.paper.width)))
    lines.extend(_pair(20, _fmt(drawing.paper.height)))
    lines.extend(_pair(9, "$BELLIUM"))
    lines.extend(_pair(1, drawing.name))
    lines.extend(_pair(0, "ENDSEC"))
    lines.extend(_pair(0, "SECTION"))
    lines.extend(_pair(2, "TABLES"))
    lines.extend(_pair(0, "TABLE"))
    lines.extend(_pair(2, "LAYER"))
    for layer in drawing.layers:
        lines.extend(_pair(0, "LAYER"))
        lines.extend(_pair(2, layer.name))
        lines.extend(_pair(70, 0))
        lines.extend(_pair(62, 7))
        lines.extend(_pair(6, "CONTINUOUS"))
    lines.extend(_pair(0, "ENDTAB"))
    lines.extend(_pair(0, "ENDSEC"))
    lines.extend(_pair(0, "SECTION"))
    lines.extend(_pair(2, "ENTITIES"))
    for entity in drawing.entities:
        if isinstance(entity, Line):
            lines.extend(_pair(0, "LINE"))
            lines.extend(_pair(8, entity.layer))
            lines.extend(_pair(10, _fmt(entity.x1)))
            lines.extend(_pair(20, _fmt(entity.y1)))
            lines.extend(_pair(11, _fmt(entity.x2)))
            lines.extend(_pair(21, _fmt(entity.y2)))
        elif isinstance(entity, Circle):
            lines.extend(_pair(0, "CIRCLE"))
            lines.extend(_pair(8, entity.layer))
            lines.extend(_pair(10, _fmt(entity.cx)))
            lines.extend(_pair(20, _fmt(entity.cy)))
            lines.extend(_pair(40, _fmt(entity.r)))
        elif isinstance(entity, Arc):
            lines.extend(_pair(0, "ARC"))
            lines.extend(_pair(8, entity.layer))
            lines.extend(_pair(10, _fmt(entity.cx)))
            lines.extend(_pair(20, _fmt(entity.cy)))
            lines.extend(_pair(40, _fmt(entity.r)))
            lines.extend(_pair(50, _fmt(entity.start)))
            lines.extend(_pair(51, _fmt(entity.end)))
        elif isinstance(entity, Text):
            lines.extend(_pair(0, "TEXT"))
            lines.extend(_pair(8, entity.layer))
            lines.extend(_pair(10, _fmt(entity.x)))
            lines.extend(_pair(20, _fmt(entity.y)))
            lines.extend(_pair(40, _fmt(entity.height)))
            lines.extend(_pair(1, entity.content))
            lines.extend(_pair(50, _fmt(entity.rotation)))
        else:
            flags = 1 if entity.closed else 0
            lines.extend(_pair(0, "POLYLINE"))
            lines.extend(_pair(8, entity.layer))
            lines.extend(_pair(66, 1))
            lines.extend(_pair(70, flags))
            for x, y in entity.points:
                lines.extend(_pair(0, "VERTEX"))
                lines.extend(_pair(8, entity.layer))
                lines.extend(_pair(10, _fmt(x)))
                lines.extend(_pair(20, _fmt(y)))
            lines.extend(_pair(0, "SEQEND"))
            lines.extend(_pair(8, entity.layer))
    lines.extend(_pair(0, "ENDSEC"))
    lines.extend(_pair(0, "EOF"))
    return chr(10).join(lines) + chr(10)


def _pairs(text: str) -> list[tuple[int, str]]:
    raw = text.replace(chr(13) + chr(10), chr(10)).replace(chr(13), chr(10))
    chunks = raw.split(chr(10))
    pairs: list[tuple[int, str]] = []
    index = 0
    while index < len(chunks):
        code_line = chunks[index].strip()
        if code_line == "" and index == len(chunks) - 1:
            break
        if index + 1 >= len(chunks):
            raise ValueError("dxf group code is missing a value")
        pairs.append((int(code_line), chunks[index + 1]))
        index += 2
    return pairs


def _collect(pairs: list[tuple[int, str]], start: int) -> tuple[dict[int, str], int]:
    values: dict[int, str] = {}
    index = start + 1
    while index < len(pairs) and pairs[index][0] != 0:
        values[pairs[index][0]] = pairs[index][1]
        index += 1
    return values, index


def drawing_from_dxf(text: object) -> Drawing:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("dxf must be a non-empty string")
    pairs = _pairs(text)
    name = None
    unit = None
    layers: list[dict[str, str]] = []
    entities: list[dict[str, object]] = []
    section = None
    index = 0
    pending: dict[str, object] | None = None
    while index < len(pairs):
        code, value = pairs[index]
        if code != 0:
            index += 1
            continue
        kind = value.strip()
        if kind == "SECTION":
            payload, index = _collect(pairs, index)
            section = payload.get(2)
            continue
        if kind == "ENDSEC":
            section = None
            index += 1
            continue
        if kind == "LAYER" and section == "TABLES":
            payload, index = _collect(pairs, index)
            layer_name = payload.get(2)
            if layer_name:
                layers.append({"name": layer_name.strip()})
            continue
        if section == "HEADER" or kind in SKIP:
            _, index = _collect(pairs, index)
            if kind == "EOF":
                break
            continue
        if section == "ENTITIES" and kind in SUPPORTED:
            payload, index = _collect(pairs, index)
            if kind == "LINE":
                entities.append(
                    {
                        "kind": "line",
                        "layer": payload[8].strip(),
                        "x1": float(payload[10]),
                        "y1": float(payload[20]),
                        "x2": float(payload[11]),
                        "y2": float(payload[21]),
                    }
                )
            elif kind == "CIRCLE":
                entities.append(
                    {
                        "kind": "circle",
                        "layer": payload[8].strip(),
                        "cx": float(payload[10]),
                        "cy": float(payload[20]),
                        "r": float(payload[40]),
                    }
                )
            elif kind == "ARC":
                entities.append(
                    {
                        "kind": "arc",
                        "layer": payload[8].strip(),
                        "cx": float(payload[10]),
                        "cy": float(payload[20]),
                        "r": float(payload[40]),
                        "start": float(payload[50]),
                        "end": float(payload[51]),
                    }
                )
            elif kind == "TEXT":
                entities.append(
                    {
                        "kind": "text",
                        "layer": payload[8].strip(),
                        "x": float(payload[10]),
                        "y": float(payload[20]),
                        "height": float(payload[40]),
                        "content": payload.get(1, ""),
                        "rotation": float(payload.get(50, "0")),
                    }
                )
            elif kind == "POLYLINE":
                pending = {
                    "kind": "polyline",
                    "layer": payload[8].strip(),
                    "points": [],
                    "closed": bool(int(payload.get(70, "0")) & 1),
                }
            elif kind == "VERTEX":
                if pending is None:
                    raise ValueError("dxf VERTEX without POLYLINE")
                points = pending["points"]
                assert isinstance(points, list)
                points.append([float(payload[10]), float(payload[20])])
            elif kind == "SEQEND" and pending is not None:
                entities.append(pending)
                pending = None
            continue
        if section == "HEADER":
            index += 1
            continue
        if kind not in SKIP and kind not in SUPPORTED:
            raise UnsupportedDrafting("unsupported_dxf_entity", (kind,))
        _, index = _collect(pairs, index)
    # Second pass for header scalars keeps the parser local to group codes.
    capture = None
    width = None
    height = None
    for code, value in pairs:
        if code == 9:
            capture = value.strip()
            continue
        if capture == "$INSUNITS" and code == 70:
            unit = UNITS_FROM_INS.get(int(value.strip()))
            capture = None
        elif capture == "$BELLIUM" and code == 1:
            name = value.strip()
            capture = None
        elif capture == "$LIMMAX" and code == 10:
            width = float(value)
        elif capture == "$LIMMAX" and code == 20:
            height = float(value)
            capture = None
        elif code == 9:
            capture = value.strip()
    if unit is None:
        raise ValueError("dxf $INSUNITS must be mm or in")
    if not name:
        raise ValueError("dxf $BELLIUM name is required")
    if not layers:
        raise ValueError("dxf needs at least one LAYER")
    if pending is not None:
        raise ValueError("dxf POLYLINE is missing SEQEND")
    if width is None or height is None:
        raise ValueError("dxf $LIMMAX paper size is required")
    return drawing_from_dict(
        {
            "name": name,
            "unit": unit,
            "paper": {"width": width, "height": height},
            "layers": layers,
            "entities": entities,
        }
    )
