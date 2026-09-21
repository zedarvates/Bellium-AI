"""Hand-authored drawings. No scanned sheet and no traced photograph."""

from __future__ import annotations

from bellium.drafting.document import Drawing, drawing_from_dict

_PLATE = {
    "name": "plate-a4",
    "unit": "mm",
    "paper": {"width": 210.0, "height": 297.0},
    "layers": [{"name": "outline"}, {"name": "notes"}],
    "entities": [
        {"kind": "line", "layer": "outline", "x1": 15.0, "y1": 20.0, "x2": 195.0, "y2": 20.0},
        {"kind": "line", "layer": "outline", "x1": 195.0, "y1": 20.0, "x2": 195.0, "y2": 270.0},
        {"kind": "line", "layer": "outline", "x1": 195.0, "y1": 270.0, "x2": 15.0, "y2": 270.0},
        {"kind": "line", "layer": "outline", "x1": 15.0, "y1": 270.0, "x2": 15.0, "y2": 20.0},
        {"kind": "line", "layer": "outline", "x1": 15.0, "y1": 20.0, "x2": 195.0, "y2": 270.0},
        {
            "kind": "text",
            "layer": "notes",
            "x": 20.0,
            "y": 8.0,
            "height": 4.0,
            "content": "BELLIUM PLATE",
            "rotation": 0.0,
        },
    ],
}

_WASHER = {
    "name": "washer",
    "unit": "mm",
    "paper": {"width": 100.0, "height": 100.0},
    "layers": [{"name": "outline"}, {"name": "holes"}],
    "entities": [
        {"kind": "circle", "layer": "outline", "cx": 50.0, "cy": 50.0, "r": 40.0},
        {"kind": "circle", "layer": "holes", "cx": 50.0, "cy": 50.0, "r": 15.0},
    ],
}

_BRACKET = {
    "name": "bracket",
    "unit": "mm",
    "paper": {"width": 80.0, "height": 70.0},
    "layers": [{"name": "outline"}],
    "entities": [
        {
            "kind": "polyline",
            "layer": "outline",
            "closed": True,
            "points": [
                [10.0, 10.0],
                [60.0, 10.0],
                [60.0, 25.0],
                [25.0, 25.0],
                [25.0, 55.0],
                [10.0, 55.0],
            ],
        }
    ],
}

_SLOT = {
    "name": "slot",
    "unit": "mm",
    "paper": {"width": 100.0, "height": 80.0},
    "layers": [{"name": "outline"}],
    "entities": [
        {"kind": "line", "layer": "outline", "x1": 30.0, "y1": 48.0, "x2": 70.0, "y2": 48.0},
        {
            "kind": "arc",
            "layer": "outline",
            "cx": 70.0,
            "cy": 40.0,
            "r": 8.0,
            "start": 90.0,
            "end": 270.0,
        },
        {"kind": "line", "layer": "outline", "x1": 70.0, "y1": 32.0, "x2": 30.0, "y2": 32.0},
        {
            "kind": "arc",
            "layer": "outline",
            "cx": 30.0,
            "cy": 40.0,
            "r": 8.0,
            "start": 270.0,
            "end": 90.0,
        },
    ],
}

_INCH_CARD = {
    "name": "inch-card",
    "unit": "in",
    "paper": {"width": 6.0, "height": 4.0},
    "layers": [{"name": "outline"}, {"name": "notes"}],
    "entities": [
        {"kind": "line", "layer": "outline", "x1": 0.5, "y1": 0.5, "x2": 5.5, "y2": 0.5},
        {"kind": "line", "layer": "outline", "x1": 5.5, "y1": 0.5, "x2": 5.5, "y2": 3.5},
        {"kind": "line", "layer": "outline", "x1": 5.5, "y1": 3.5, "x2": 0.5, "y2": 3.5},
        {"kind": "line", "layer": "outline", "x1": 0.5, "y1": 3.5, "x2": 0.5, "y2": 0.5},
        {
            "kind": "circle",
            "layer": "outline",
            "cx": 3.0,
            "cy": 2.0,
            "r": 0.75,
        },
        {
            "kind": "text",
            "layer": "notes",
            "x": 0.6,
            "y": 0.15,
            "height": 0.2,
            "content": "INCH CARD",
            "rotation": 0.0,
        },
    ],
}

_FIXTURES = {
    "plate-a4": _PLATE,
    "washer": _WASHER,
    "bracket": _BRACKET,
    "slot": _SLOT,
    "inch-card": _INCH_CARD,
}

FIXTURE_NAMES = tuple(_FIXTURES)


def fixture(name: str) -> Drawing:
    if name not in _FIXTURES:
        raise KeyError(name)
    return drawing_from_dict(_FIXTURES[name])


def fixtures() -> tuple[Drawing, ...]:
    return tuple(fixture(name) for name in FIXTURE_NAMES)
