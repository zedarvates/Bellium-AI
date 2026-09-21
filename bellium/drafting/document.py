"""Canonical 2D drawing: units, layers and exact primitives.

The document is Y-up, origin at the paper bottom-left, coordinates in the
declared unit. This is a drafting record, not an image and not a CAD kernel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

UNITS = ("mm", "in")
KINDS = ("line", "polyline", "circle", "arc", "text")
LAYER_CHARS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-")
TOLERANCE = 1e-6
DRAWING_KEYS = {"name", "unit", "paper", "layers", "entities"}
PAPER_KEYS = {"width", "height"}
LAYER_KEYS = {"name"}
LINE_KEYS = {"kind", "layer", "x1", "y1", "x2", "y2"}
POLYLINE_KEYS = {"kind", "layer", "points", "closed"}
CIRCLE_KEYS = {"kind", "layer", "cx", "cy", "r"}
ARC_KEYS = {"kind", "layer", "cx", "cy", "r", "start", "end"}
TEXT_KEYS = {"kind", "layer", "x", "y", "height", "content", "rotation"}


class UnsupportedDrafting(ValueError):
    """Well-formed input that this gate does not represent."""

    def __init__(self, reason: str, details: tuple[str, ...] = ()) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details = details


@dataclass(frozen=True)
class Paper:
    width: float
    height: float


@dataclass(frozen=True)
class Layer:
    name: str


@dataclass(frozen=True)
class Line:
    layer: str
    x1: float
    y1: float
    x2: float
    y2: float
    kind: str = "line"


@dataclass(frozen=True)
class Polyline:
    layer: str
    points: tuple[tuple[float, float], ...]
    closed: bool = False
    kind: str = "polyline"


@dataclass(frozen=True)
class Circle:
    layer: str
    cx: float
    cy: float
    r: float
    kind: str = "circle"


@dataclass(frozen=True)
class Arc:
    layer: str
    cx: float
    cy: float
    r: float
    start: float
    end: float
    kind: str = "arc"


@dataclass(frozen=True)
class Text:
    layer: str
    x: float
    y: float
    height: float
    content: str
    rotation: float = 0.0
    kind: str = "text"


Entity = Line | Polyline | Circle | Arc | Text


@dataclass(frozen=True)
class Drawing:
    name: str
    unit: str
    paper: Paper
    layers: tuple[Layer, ...]
    entities: tuple[Entity, ...]


def _unknown(payload: dict[str, Any], allowed: set[str], label: str) -> None:
    extra = sorted(set(payload) - allowed)
    if extra:
        raise ValueError(f"unknown {label} keys refuse to be ignored: {', '.join(extra)}")


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _positive(value: object, name: str) -> float:
    number = _finite(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return number


def _norm_deg(value: object, name: str) -> float:
    number = _finite(value, name) % 360.0
    if number < 0.0:
        number += 360.0
    return number


def _layer_name(value: object) -> str:
    if not isinstance(value, str) or not value or any(ch not in LAYER_CHARS for ch in value):
        raise ValueError("layer names use ASCII letters, digits, _ and -")
    return value


def _parse_layer(payload: object) -> Layer:
    if not isinstance(payload, dict):
        raise ValueError("each layer must be an object")
    _unknown(payload, LAYER_KEYS, "layer")
    return Layer(_layer_name(payload.get("name")))


def _parse_point(payload: object, name: str) -> tuple[float, float]:
    if not isinstance(payload, (list, tuple)) or len(payload) != 2:
        raise ValueError(f"{name} must be an [x, y] pair")
    return (_finite(payload[0], f"{name}.x"), _finite(payload[1], f"{name}.y"))


def _parse_line(payload: dict[str, Any]) -> Line:
    _unknown(payload, LINE_KEYS, "line")
    line = Line(
        layer=_layer_name(payload.get("layer")),
        x1=_finite(payload.get("x1"), "x1"),
        y1=_finite(payload.get("y1"), "y1"),
        x2=_finite(payload.get("x2"), "x2"),
        y2=_finite(payload.get("y2"), "y2"),
    )
    if math.hypot(line.x2 - line.x1, line.y2 - line.y1) <= TOLERANCE:
        raise ValueError("a line must have length")
    return line


def _parse_polyline(payload: dict[str, Any]) -> Polyline:
    _unknown(payload, POLYLINE_KEYS, "polyline")
    closed = payload.get("closed", False)
    if not isinstance(closed, bool):
        raise ValueError("closed must be a boolean")
    raw_points = payload.get("points")
    if not isinstance(raw_points, (list, tuple)) or len(raw_points) < 2:
        raise ValueError("a polyline needs at least two points")
    points = tuple(_parse_point(item, f"points[{index}]") for index, item in enumerate(raw_points))
    if closed and len(points) < 3:
        raise ValueError("a closed polyline needs at least three points")
    return Polyline(layer=_layer_name(payload.get("layer")), points=points, closed=closed)


def _parse_circle(payload: dict[str, Any]) -> Circle:
    _unknown(payload, CIRCLE_KEYS, "circle")
    return Circle(
        layer=_layer_name(payload.get("layer")),
        cx=_finite(payload.get("cx"), "cx"),
        cy=_finite(payload.get("cy"), "cy"),
        r=_positive(payload.get("r"), "r"),
    )


def _parse_arc(payload: dict[str, Any]) -> Arc:
    _unknown(payload, ARC_KEYS, "arc")
    start = _norm_deg(payload.get("start"), "start")
    end = _norm_deg(payload.get("end"), "end")
    if abs(start - end) <= TOLERANCE:
        raise ValueError("arc spans 0 degrees; use a circle")
    return Arc(
        layer=_layer_name(payload.get("layer")),
        cx=_finite(payload.get("cx"), "cx"),
        cy=_finite(payload.get("cy"), "cy"),
        r=_positive(payload.get("r"), "r"),
        start=start,
        end=end,
    )


def _parse_text(payload: dict[str, Any]) -> Text:
    _unknown(payload, TEXT_KEYS, "text")
    content = payload.get("content")
    if not isinstance(content, str) or not content:
        raise ValueError("text content must be a single non-empty line")
    if any(ord(ch) in (10, 13) for ch in content):
        raise ValueError("text content must be a single non-empty line")
    rotation = payload.get("rotation", 0.0)
    return Text(
        layer=_layer_name(payload.get("layer")),
        x=_finite(payload.get("x"), "x"),
        y=_finite(payload.get("y"), "y"),
        height=_positive(payload.get("height"), "height"),
        content=content,
        rotation=_norm_deg(rotation, "rotation"),
    )


def _parse_entity(payload: object) -> Entity:
    if not isinstance(payload, dict):
        raise ValueError("each entity must be an object")
    kind = payload.get("kind")
    if kind not in KINDS:
        if isinstance(kind, str) and kind:
            raise UnsupportedDrafting("unsupported_entity", (kind,))
        raise ValueError("entity kind is required")
    parsers = {
        "line": _parse_line,
        "polyline": _parse_polyline,
        "circle": _parse_circle,
        "arc": _parse_arc,
        "text": _parse_text,
    }
    return parsers[kind](payload)


def drawing_from_dict(payload: object) -> Drawing:
    if not isinstance(payload, dict):
        raise ValueError("drawing must be an object")
    _unknown(payload, DRAWING_KEYS, "drawing")
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("drawing name is required")
    unit = payload.get("unit")
    if unit not in UNITS:
        raise ValueError("unit must be mm or in")
    paper_payload = payload.get("paper")
    if not isinstance(paper_payload, dict):
        raise ValueError("paper must be an object")
    _unknown(paper_payload, PAPER_KEYS, "paper")
    paper = Paper(
        width=_positive(paper_payload.get("width"), "paper.width"),
        height=_positive(paper_payload.get("height"), "paper.height"),
    )
    raw_layers = payload.get("layers")
    if not isinstance(raw_layers, (list, tuple)) or not raw_layers:
        raise ValueError("at least one layer is required")
    layers = tuple(_parse_layer(item) for item in raw_layers)
    names = [layer.name for layer in layers]
    if len(names) != len(set(names)):
        raise ValueError("layer names must be unique")
    raw_entities = payload.get("entities")
    if not isinstance(raw_entities, (list, tuple)):
        raise ValueError("entities must be a list")
    entities = tuple(_parse_entity(item) for item in raw_entities)
    return Drawing(name=name.strip(), unit=unit, paper=paper, layers=layers, entities=entities)


def _entity_to_dict(entity: Entity) -> dict[str, Any]:
    if isinstance(entity, Line):
        return {
            "kind": "line",
            "layer": entity.layer,
            "x1": entity.x1,
            "y1": entity.y1,
            "x2": entity.x2,
            "y2": entity.y2,
        }
    if isinstance(entity, Polyline):
        return {
            "kind": "polyline",
            "layer": entity.layer,
            "points": [list(point) for point in entity.points],
            "closed": entity.closed,
        }
    if isinstance(entity, Circle):
        return {
            "kind": "circle",
            "layer": entity.layer,
            "cx": entity.cx,
            "cy": entity.cy,
            "r": entity.r,
        }
    if isinstance(entity, Arc):
        return {
            "kind": "arc",
            "layer": entity.layer,
            "cx": entity.cx,
            "cy": entity.cy,
            "r": entity.r,
            "start": entity.start,
            "end": entity.end,
        }
    return {
        "kind": "text",
        "layer": entity.layer,
        "x": entity.x,
        "y": entity.y,
        "height": entity.height,
        "content": entity.content,
        "rotation": entity.rotation,
    }


def drawing_to_dict(drawing: Drawing) -> dict[str, Any]:
    return {
        "name": drawing.name,
        "unit": drawing.unit,
        "paper": {"width": drawing.paper.width, "height": drawing.paper.height},
        "layers": [{"name": layer.name} for layer in drawing.layers],
        "entities": [_entity_to_dict(entity) for entity in drawing.entities],
    }


def check_drawing(drawing: Drawing) -> list[str]:
    violations: list[str] = []
    layer_names = {layer.name for layer in drawing.layers}
    if len(layer_names) != len(drawing.layers):
        violations.append("duplicate_layer")
    if drawing.unit not in UNITS:
        violations.append("unknown_unit")
    if drawing.paper.width <= 0.0 or drawing.paper.height <= 0.0:
        violations.append("paper_not_positive")
    for index, entity in enumerate(drawing.entities):
        if entity.layer not in layer_names:
            violations.append(f"entity_{index}_missing_layer")
    return violations


def _close(left: float, right: float) -> bool:
    return abs(left - right) <= TOLERANCE


def _points_match(
    left: tuple[tuple[float, float], ...],
    right: tuple[tuple[float, float], ...],
) -> bool:
    if len(left) != len(right):
        return False
    return all(_close(a[0], b[0]) and _close(a[1], b[1]) for a, b in zip(left, right))


def _entities_match(left: Entity, right: Entity) -> bool:
    if type(left) is not type(right) or left.layer != right.layer:
        return False
    if isinstance(left, Line) and isinstance(right, Line):
        return (
            _close(left.x1, right.x1)
            and _close(left.y1, right.y1)
            and _close(left.x2, right.x2)
            and _close(left.y2, right.y2)
        )
    if isinstance(left, Polyline) and isinstance(right, Polyline):
        return left.closed == right.closed and _points_match(left.points, right.points)
    if isinstance(left, Circle) and isinstance(right, Circle):
        return _close(left.cx, right.cx) and _close(left.cy, right.cy) and _close(left.r, right.r)
    if isinstance(left, Arc) and isinstance(right, Arc):
        return (
            _close(left.cx, right.cx)
            and _close(left.cy, right.cy)
            and _close(left.r, right.r)
            and _close(left.start, right.start)
            and _close(left.end, right.end)
        )
    if isinstance(left, Text) and isinstance(right, Text):
        return (
            left.content == right.content
            and _close(left.x, right.x)
            and _close(left.y, right.y)
            and _close(left.height, right.height)
            and _close(left.rotation, right.rotation)
        )
    return False


def drawings_match(left: Drawing, right: Drawing) -> bool:
    if left.name != right.name or left.unit != right.unit:
        return False
    if not _close(left.paper.width, right.paper.width):
        return False
    if not _close(left.paper.height, right.paper.height):
        return False
    if [layer.name for layer in left.layers] != [layer.name for layer in right.layers]:
        return False
    if len(left.entities) != len(right.entities):
        return False
    return all(
        _entities_match(first, second)
        for first, second in zip(left.entities, right.entities)
    )
