"""Bellium drafting SVG profile: exact primitives, Y-up model space."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

from bellium.drafting.document import (
    Arc,
    Circle,
    Drawing,
    Line,
    Polyline,
    UnsupportedDrafting,
    drawing_from_dict,
)

NS = "http://www.w3.org/2000/svg"
PROFILE = "drafting-v0"


def _fmt(value: float) -> str:
    text = format(value, ".12g")
    return text


def _qname(tag: str) -> str:
    return f"{{{NS}}}{tag}"


def _local(tag: str) -> str:
    if tag.startswith("{") and "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _arc_path(entity: Arc) -> str:
    start = math.radians(entity.start)
    end = math.radians(entity.end)
    x1 = entity.cx + entity.r * math.cos(start)
    y1 = entity.cy + entity.r * math.sin(start)
    x2 = entity.cx + entity.r * math.cos(end)
    y2 = entity.cy + entity.r * math.sin(end)
    delta = (entity.end - entity.start) % 360.0
    large = 1 if delta > 180.0 else 0
    return (
        f"M {_fmt(x1)} {_fmt(y1)} A {_fmt(entity.r)} {_fmt(entity.r)} "
        f"0 {large} 1 {_fmt(x2)} {_fmt(y2)}"
    )


def drawing_to_svg(drawing: Drawing) -> str:
    width = _fmt(drawing.paper.width)
    height = _fmt(drawing.paper.height)
    lines = [
        f'<svg xmlns="{NS}" width="{width}{drawing.unit}" height="{height}{drawing.unit}" '
        f'viewBox="0 0 {width} {height}" data-bellium="{PROFILE}" '
        f'data-name="{escape(drawing.name)}" data-unit="{drawing.unit}" data-y="up">',
        f'  <g data-space="model" transform="matrix(1 0 0 -1 0 {height})">',
    ]
    by_layer: dict[str, list[str]] = {layer.name: [] for layer in drawing.layers}
    for entity in drawing.entities:
        if isinstance(entity, Line):
            markup = (
                f'<line data-kind="line" x1="{_fmt(entity.x1)}" y1="{_fmt(entity.y1)}" '
                f'x2="{_fmt(entity.x2)}" y2="{_fmt(entity.y2)}" fill="none" stroke="black" />'
            )
        elif isinstance(entity, Polyline):
            points = " ".join(f"{_fmt(x)},{_fmt(y)}" for x, y in entity.points)
            tag = "polygon" if entity.closed else "polyline"
            closed = "true" if entity.closed else "false"
            markup = (
                f'<{tag} data-kind="polyline" data-closed="{closed}" points="{points}" '
                f'fill="none" stroke="black" />'
            )
        elif isinstance(entity, Circle):
            markup = (
                f'<circle data-kind="circle" cx="{_fmt(entity.cx)}" cy="{_fmt(entity.cy)}" '
                f'r="{_fmt(entity.r)}" fill="none" stroke="black" />'
            )
        elif isinstance(entity, Arc):
            markup = (
                f'<path data-kind="arc" data-cx="{_fmt(entity.cx)}" data-cy="{_fmt(entity.cy)}" '
                f'data-r="{_fmt(entity.r)}" data-start="{_fmt(entity.start)}" '
                f'data-end="{_fmt(entity.end)}" d="{_arc_path(entity)}" fill="none" stroke="black" />'
            )
        else:
            rot = _fmt(entity.rotation)
            markup = (
                f'<text data-kind="text" data-x="{_fmt(entity.x)}" data-y="{_fmt(entity.y)}" '
                f'data-height="{_fmt(entity.height)}" data-rotation="{rot}" '
                f'x="{_fmt(entity.x)}" y="{_fmt(entity.y)}" font-size="{_fmt(entity.height)}" '
                f'transform="translate({_fmt(entity.x)} {_fmt(entity.y)}) rotate({rot}) '
                f"scale(1,-1) translate({_fmt(-entity.x)} {_fmt(-entity.y)})"
                f'">{escape(entity.content)}</text>'
            )
        by_layer[entity.layer].append(markup)
    for layer in drawing.layers:
        lines.append(f'    <g data-layer="{layer.name}" fill="none" stroke="black">')
        for markup in by_layer[layer.name]:
            lines.append(f"      {markup}")
        lines.append("    </g>")
    lines.append("  </g>")
    lines.append("</svg>")
    return chr(10).join(lines) + chr(10)


def _require(element: ET.Element, name: str) -> str:
    value = element.get(name)
    if value is None:
        raise ValueError(f"svg {name} is required")
    return value


def _entity_from_element(element: ET.Element, layer: str) -> dict[str, object]:
    kind = element.get("data-kind")
    if kind == "line":
        return {
            "kind": "line",
            "layer": layer,
            "x1": float(_require(element, "x1")),
            "y1": float(_require(element, "y1")),
            "x2": float(_require(element, "x2")),
            "y2": float(_require(element, "y2")),
        }
    if kind == "polyline":
        raw = _require(element, "points").split()
        points = []
        for item in raw:
            x_text, y_text = item.split(",", 1)
            points.append([float(x_text), float(y_text)])
        return {
            "kind": "polyline",
            "layer": layer,
            "points": points,
            "closed": element.get("data-closed") == "true",
        }
    if kind == "circle":
        return {
            "kind": "circle",
            "layer": layer,
            "cx": float(_require(element, "cx")),
            "cy": float(_require(element, "cy")),
            "r": float(_require(element, "r")),
        }
    if kind == "arc":
        return {
            "kind": "arc",
            "layer": layer,
            "cx": float(_require(element, "data-cx")),
            "cy": float(_require(element, "data-cy")),
            "r": float(_require(element, "data-r")),
            "start": float(_require(element, "data-start")),
            "end": float(_require(element, "data-end")),
        }
    if kind == "text":
        return {
            "kind": "text",
            "layer": layer,
            "x": float(_require(element, "data-x")),
            "y": float(_require(element, "data-y")),
            "height": float(_require(element, "data-height")),
            "content": "".join(element.itertext()),
            "rotation": float(element.get("data-rotation") or "0"),
        }
    if kind:
        raise UnsupportedDrafting("unsupported_svg_entity", (kind,))
    raise UnsupportedDrafting("unsupported_svg_profile", (_local(element.tag),))


def drawing_from_svg(text: object) -> Drawing:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("svg must be a non-empty string")
    root = ET.fromstring(text)
    if _local(root.tag) != "svg" or root.get("data-bellium") != PROFILE:
        raise UnsupportedDrafting("unsupported_svg_profile")
    name = root.get("data-name")
    unit = root.get("data-unit")
    if not name or unit is None:
        raise ValueError("svg data-name and data-unit are required")
    viewbox = (root.get("viewBox") or "").split()
    if len(viewbox) != 4:
        raise ValueError("svg viewBox is required")
    paper = {"width": float(viewbox[2]), "height": float(viewbox[3])}
    layers: list[dict[str, str]] = []
    entities: list[dict[str, object]] = []
    for group in root.iter():
        if _local(group.tag) != "g" or group.get("data-layer") is None:
            continue
        layer = group.get("data-layer")
        if layer is None:
            continue
        layers.append({"name": layer})
        for child in list(group):
            entities.append(_entity_from_element(child, layer))
    return drawing_from_dict(
        {
            "name": name,
            "unit": unit,
            "paper": paper,
            "layers": layers,
            "entities": entities,
        }
    )
