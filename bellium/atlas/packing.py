"""Deterministic atlas packing: place declared frames on bounded pages.

A plan is geometry, not pixels: nothing here is rasterized, rotated, trimmed,
scaled or exported. Two published baselines (insertion-order shelf and
height-sorted shelf) stay beside the shipped skyline packer, so occupancy is
compared instead of assumed. A frame that does not fit is reported unplaced
with a reason; it is never shrunk to make the page succeed.
"""

from __future__ import annotations

from dataclasses import dataclass

METHODS = ("skyline", "shelf", "next-fit")
FRAME_KEYS = ("name", "width", "height")
OVERSIZED = "oversized"
NO_ROOM = "no_room"
MAX_PAGES = 64
Rect = tuple[int, int, int, int]


@dataclass(frozen=True)
class Frame:
    """One declared frame: a name and its exact pixel size."""

    name: str
    width: int
    height: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("a frame needs a name")
        _positive_int("frame width", self.width)
        _positive_int("frame height", self.height)


@dataclass(frozen=True)
class Placement:
    """Where one frame goes. x and y are the content origin, padding excluded."""

    name: str
    page: int
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class Unplaced:
    name: str
    reason: str


@dataclass(frozen=True)
class AtlasPlan:
    pages: tuple[tuple[Placement, ...], ...]
    page_width: int
    page_height: int
    padding: int
    method: str
    unplaced: tuple[Unplaced, ...]
    used_area: int

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def complete(self) -> bool:
        return not self.unplaced

    @property
    def occupancy(self) -> float:
        """Frame area divided by the page area the plan actually occupies."""
        area = self.page_count * self.page_width * self.page_height
        return self.used_area / area if area else 0.0

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "page_width": self.page_width,
            "page_height": self.page_height,
            "padding": self.padding,
            "page_count": self.page_count,
            "complete": self.complete,
            "occupancy": round(self.occupancy, 4),
            "used_area": self.used_area,
            "pages": [
                [
                    {
                        "name": item.name, "x": item.x, "y": item.y,
                        "width": item.width, "height": item.height,
                    }
                    for item in page
                ]
                for page in self.pages
            ],
            "unplaced": [{"name": item.name, "reason": item.reason} for item in self.unplaced],
        }


def _positive_int(label: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _non_negative_int(label: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def power_of_two_ceiling(value: object) -> int:
    """Smallest power of two that still contains the declared extent."""
    number = _positive_int("extent", value)
    result = 1
    while result < number:
        result *= 2
    return result


def parse_frames(raw: object) -> tuple[Frame, ...]:
    """Validate declared frames. Duplicate names raise: a plan must be unambiguous."""
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("frames must be a non-empty list")
    frames = []
    seen = set()
    for index, item in enumerate(raw):
        if isinstance(item, Frame):
            frame = item
        elif isinstance(item, dict):
            extra = sorted(set(item) - set(FRAME_KEYS))
            if extra:
                raise ValueError(f"frame {index} has unknown keys: {', '.join(extra)}")
            missing = [key for key in FRAME_KEYS if key not in item]
            if missing:
                raise ValueError(f"frame {index} is missing {', '.join(missing)}")
            frame = Frame(name=item["name"], width=item["width"], height=item["height"])
        else:
            raise ValueError(f"frame {index} must be an object or a Frame")
        if frame.name in seen:
            raise ValueError(f"duplicate frame name: {frame.name}")
        seen.add(frame.name)
        frames.append(frame)
    return tuple(frames)


def cells(placement: Placement, padding: int) -> Rect:
    """The content rectangle inflated by the gutter two frames must not share."""
    return (
        placement.x - padding,
        placement.y - padding,
        placement.x + placement.width + padding,
        placement.y + placement.height + padding,
    )


def _overlaps(left: Rect, right: Rect) -> bool:
    return left[0] < right[2] and right[0] < left[2] and left[1] < right[3] and right[1] < left[3]


def check_plan(frames: tuple[Frame, ...], plan: AtlasPlan) -> list[str]:
    """Recompute every invariant the plan claims, including its own bookkeeping.

    An incomplete plan is not a violation: a frame that does not fit belongs in
    unplaced. The violation is a frame that is neither placed nor reported.
    """
    violations: list[str] = []
    by_name = {frame.name: frame for frame in frames}
    placed: dict[str, Placement] = {}
    area = 0
    for index, page in enumerate(plan.pages):
        if not page:
            violations.append(f"page {index} is empty")
        for placement in page:
            frame = by_name.get(placement.name)
            if frame is None:
                violations.append(f"unknown frame placed: {placement.name}")
                continue
            if placement.name in placed:
                violations.append(f"frame placed twice: {placement.name}")
            placed[placement.name] = placement
            if (placement.width, placement.height) != (frame.width, frame.height):
                violations.append(f"placement size changed for {placement.name}")
            area += frame.width * frame.height
            left, top, right, bottom = cells(placement, plan.padding)
            if left < 0 or top < 0 or right > plan.page_width or bottom > plan.page_height:
                violations.append(f"frame out of bounds: {placement.name}")
    reported = {item.name for item in plan.unplaced}
    for frame in frames:
        if frame.name not in placed and frame.name not in reported:
            violations.append(f"frame neither placed nor reported unplaced: {frame.name}")
    for name in sorted(reported):
        if name not in by_name:
            violations.append(f"unknown frame reported unplaced: {name}")
        elif name in placed:
            violations.append(f"frame placed and reported unplaced: {name}")
    if area != plan.used_area:
        violations.append(f"used_area is {plan.used_area}, placed frames total {area}")
    order = sorted(placed)
    for first in range(len(order)):
        for second in range(first + 1, len(order)):
            left = placed[order[first]]
            right = placed[order[second]]
            if left.page == right.page and _overlaps(
                cells(left, plan.padding), cells(right, plan.padding)
            ):
                violations.append(f"frames overlap: {left.name} and {right.name}")
    return violations


def _shelf_page(boxes, page_width: int, page_height: int):
    """One page of shelf rows in the given order. Returns (placed, remaining)."""
    placed: list[tuple[str, int, int]] = []
    remaining: list[tuple[str, int, int]] = []
    x = 0
    y = 0
    row_height = 0
    for box in boxes:
        name, width, height = box
        if x + width > page_width:
            y += row_height
            x = 0
            row_height = 0
        if width > page_width or y + height > page_height:
            remaining.append(box)
            continue
        placed.append((name, x, y))
        x += width
        row_height = max(row_height, height)
    return placed, remaining


def _add_level(skyline: list, x: int, width: int, top: int) -> None:
    right = x + width
    updated: list[tuple[int, int, int]] = []
    for segment_x, segment_y, segment_width in skyline:
        if segment_x + segment_width <= x or segment_x >= right:
            updated.append((segment_x, segment_y, segment_width))
            continue
        if segment_x < x:
            updated.append((segment_x, segment_y, x - segment_x))
        if segment_x + segment_width > right:
            updated.append((right, segment_y, segment_x + segment_width - right))
    updated.append((x, top, width))
    updated.sort()
    merged: list[tuple[int, int, int]] = []
    for segment in updated:
        if merged and merged[-1][1] == segment[1] and merged[-1][0] + merged[-1][2] == segment[0]:
            previous = merged.pop()
            merged.append((previous[0], previous[1], previous[2] + segment[2]))
        else:
            merged.append(segment)
    skyline[:] = merged


def _skyline_page(boxes, page_width: int, page_height: int):
    """One page of bottom-left skyline placement. Returns (placed, remaining)."""
    skyline: list[tuple[int, int, int]] = [(0, 0, page_width)]
    placed: list[tuple[str, int, int]] = []
    remaining: list[tuple[str, int, int]] = []
    for box in boxes:
        name, width, height = box
        if width > page_width or height > page_height:
            remaining.append(box)
            continue
        best: tuple[tuple[int, int, int], int] | None = None
        for index, (segment_x, segment_y, _) in enumerate(skyline):
            level = segment_y
            needed = width
            cursor = index
            while needed > 0 and cursor < len(skyline):
                segment = skyline[cursor]
                level = max(level, segment[1])
                needed -= segment[2]
                cursor += 1
            if needed > 0 or level + height > page_height:
                continue
            score = (level + height, level, segment_x)
            if best is None or score < best[0]:
                best = (score, segment_x)
        if best is None:
            remaining.append(box)
            continue
        _, x = best
        level = best[0][1]
        placed.append((name, x, level))
        _add_level(skyline, x, width, level + height)
    return placed, remaining


def _sorted(boxes, method: str):
    if method == "skyline":
        return sorted(boxes, key=lambda box: (-box[2], -box[1], box[0]))
    if method == "shelf":
        return sorted(boxes, key=lambda box: (-box[2], box[0]))
    return list(boxes)


def pack(
    frames: object,
    *,
    width: object,
    height: object,
    method: str = "skyline",
    padding: int = 0,
    max_pages: int = 1,
) -> AtlasPlan:
    """Place every frame that fits inside the declared pages.

    padding is a gutter: content is kept that many pixels from the page border and
    from any other frame. Frames that do not fit stay unplaced with a reason.
    """
    page_width = _positive_int("page width", width)
    page_height = _positive_int("page height", height)
    gutter = _non_negative_int("padding", padding)
    page_limit = _positive_int("max_pages", max_pages)
    if page_limit > MAX_PAGES:
        raise ValueError(f"max_pages must be at most {MAX_PAGES}")
    if method not in METHODS:
        raise ValueError(f"method must be one of: {', '.join(METHODS)}")
    declared = parse_frames(frames)
    boxes = [
        (frame.name, frame.width + 2 * gutter, frame.height + 2 * gutter)
        for frame in declared
    ]
    page_function = _skyline_page if method == "skyline" else _shelf_page
    remaining = _sorted(boxes, method)
    raw_pages: list[list[tuple[str, int, int]]] = []
    while remaining and len(raw_pages) < page_limit:
        placed, remaining = page_function(remaining, page_width, page_height)
        if not placed:
            break  # nothing fits on a fresh page: report the rest instead of looping
        raw_pages.append(placed)
    sizes = {frame.name: frame for frame in declared}
    pages = tuple(
        tuple(
            Placement(name=name, page=index, x=x + gutter, y=y + gutter,
                      width=sizes[name].width, height=sizes[name].height)
            for name, x, y in page
        )
        for index, page in enumerate(raw_pages)
    )
    unplaced = tuple(
        Unplaced(
            name=name,
            reason=(
                OVERSIZED
                if width_ + 2 * gutter > page_width or height_ + 2 * gutter > page_height
                else NO_ROOM
            ),
        )
        for name, width_, height_ in remaining
    )
    used_area = sum(sizes[item.name].width * sizes[item.name].height for page in pages for item in page)
    return AtlasPlan(
        pages=pages,
        page_width=page_width,
        page_height=page_height,
        padding=gutter,
        method=method,
        unplaced=unplaced,
        used_area=used_area,
    )
