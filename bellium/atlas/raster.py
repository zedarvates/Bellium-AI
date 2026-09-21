"""Compose atlas pages from placed frames, pixel for pixel.

Rasterization is an exact copy: every pixel inside a placement is the source pixel
at the same offset, and every other pixel is the declared background. Nothing is
scaled, resampled, blended or rotated, so a page can be re-derived and checked
against its plan instead of trusted. Alpha travels as a fourth channel.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

from bellium.atlas.packing import AtlasPlan, Frame, check_plan

CHANNELS = (3, 4)
DEFAULT_BACKGROUND = {3: (0, 0, 0), 4: (0, 0, 0, 0)}
Pixel = tuple[int, ...]


@dataclass(frozen=True)
class RasterPage:
    index: int
    width: int
    height: int
    channels: int
    pixels: tuple[tuple[Pixel, ...], ...]
    digest: str
    raw_bytes: int


@dataclass(frozen=True)
class AtlasRaster:
    pages: tuple[RasterPage, ...]
    channels: int
    background: Pixel
    placed: int

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def bytes_written(self) -> int:
        return sum(page.raw_bytes for page in self.pages)

    def to_dict(self) -> dict:
        return {
            "pages": [
                {
                    "index": page.index,
                    "width": page.width,
                    "height": page.height,
                    "channels": page.channels,
                    "digest": page.digest,
                    "raw_bytes": page.raw_bytes,
                    "pixels": [[list(pixel) for pixel in row] for row in page.pixels],
                }
                for page in self.pages
            ],
            "page_count": self.page_count,
            "channels": self.channels,
            "background": list(self.background),
            "placed": self.placed,
            "bytes_written": self.bytes_written,
        }


def pixel_problem(image: object) -> str | None:
    """Structural problem of one source image, or None when it is well formed."""
    if not isinstance(image, list) or not image:
        return "image must be a non-empty list of rows"
    if not isinstance(image[0], list) or not image[0]:
        return "image rows must be non-empty lists"
    width = len(image[0])
    channels: int | None = None
    for row in image:
        if not isinstance(row, list) or len(row) != width:
            return "image rows must have equal width"
        for pixel in row:
            if not isinstance(pixel, (tuple, list)) or len(pixel) not in CHANNELS:
                return "pixels must be RGB or RGBA tuples"
            if channels is None:
                channels = len(pixel)
            elif len(pixel) != channels:
                return "pixels must share one channel count"
            for value in pixel:
                if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 255:
                    return "channels must be integers in [0, 255]"
    return None


def source_shape(image: object) -> tuple[int, int, int]:
    problem = pixel_problem(image)
    if problem:
        raise ValueError(problem)
    assert isinstance(image, list)
    return len(image), len(image[0]), len(image[0][0])


def _background(value: object, channels: int) -> Pixel:
    if value is None:
        return DEFAULT_BACKGROUND[channels]
    if not isinstance(value, (tuple, list)) or len(value) != channels:
        raise ValueError(f"background must have {channels} channels")
    for channel in value:
        if isinstance(channel, bool) or not isinstance(channel, int) or not 0 <= channel <= 255:
            raise ValueError("background channels must be integers in [0, 255]")
    return tuple(value)


def check_sources(frames: tuple[Frame, ...], sources: object) -> list[str]:
    """Compare declared frames with the provided images. Never mutates either."""
    violations: list[str] = []
    if not isinstance(sources, dict):
        return ["sources must be an object mapping names to images"]
    declared = {frame.name: frame for frame in frames}
    missing = sorted(name for name in declared if name not in sources)
    unused = sorted(name for name in sources if name not in declared)
    for name in missing:
        violations.append(f"missing source image for frame {name}")
    for name in unused:
        violations.append(f"source image without a declared frame: {name}")
    counts: dict[int, list[str]] = {}
    for name, frame in declared.items():
        if name not in sources:
            continue
        image = sources[name]
        problem = pixel_problem(image)
        if problem:
            violations.append(f"source {name}: {problem}")
            continue
        height, width, channels = source_shape(image)
        if (width, height) != (frame.width, frame.height):
            violations.append(
                f"source {name} is {width} x {height} but the frame declares "
                f"{frame.width} x {frame.height}"
            )
            continue
        counts.setdefault(channels, []).append(name)
    if len(counts) > 1:
        described = "; ".join(
            f"{channels} channels: {', '.join(sorted(names))}"
            for channels, names in sorted(counts.items())
        )
        violations.append(f"sources must share one channel count ({described})")
    return violations


def page_digest(rows) -> str:
    """sha256 of the raw byte stream in row-major order.

    The digest describes the content bytes; page width, height and channels are
    checked separately, so a caller must keep them together with the digest.
    """
    hasher = hashlib.sha256()
    for row in rows:
        for pixel in row:
            hasher.update(bytes(pixel))
    return hasher.hexdigest()


def rasterize(
    frames: tuple[Frame, ...],
    plan: AtlasPlan,
    sources: object,
    *,
    background: object = None,
) -> AtlasRaster:
    problems = check_sources(frames, sources)
    if problems:
        raise ValueError("; ".join(problems))
    violations = check_plan(frames, plan)
    if violations:
        raise ValueError("plan is invalid: " + "; ".join(violations))
    if not plan.complete:
        names = ", ".join(item.name for item in plan.unplaced)
        raise ValueError(f"plan leaves frames unplaced: {names}")
    channels = source_shape(sources[frames[0].name])[2]
    fill = _background(background, channels)
    pages = []
    placed = 0
    for index, page in enumerate(plan.pages):
        rows = [[fill] * plan.page_width for _ in range(plan.page_height)]
        for item in page:
            image = sources[item.name]
            for offset_y in range(item.height):
                target = rows[item.y + offset_y]
                source_row = image[offset_y]
                for offset_x in range(item.width):
                    target[item.x + offset_x] = tuple(source_row[offset_x])
            placed += 1
        frozen = tuple(tuple(row) for row in rows)
        pages.append(RasterPage(
            index=index,
            width=plan.page_width,
            height=plan.page_height,
            channels=channels,
            pixels=frozen,
            digest=page_digest(frozen),
            raw_bytes=plan.page_width * plan.page_height * channels,
        ))
    return AtlasRaster(tuple(pages), channels, fill, placed)


def check_raster(
    frames: tuple[Frame, ...],
    plan: AtlasPlan,
    raster: object,
    sources: object,
) -> list[str]:
    """Recompute the claims of a raster: copies, background, gutter, digests, sizes."""
    violations: list[str] = []
    if not isinstance(raster, AtlasRaster):
        return ["raster must be an AtlasRaster"]
    if raster.channels not in CHANNELS:
        violations.append(f"raster channels must be one of {CHANNELS}")
    if len(raster.background) != raster.channels:
        violations.append("background does not match the channel count")
    if raster.page_count != plan.page_count:
        violations.append(f"raster has {raster.page_count} pages, the plan has {plan.page_count}")
    expected_placed = sum(len(page) for page in plan.pages)
    if raster.placed != expected_placed:
        violations.append(f"raster reports {raster.placed} placements, the plan has {expected_placed}")
    covered: dict[int, set[tuple[int, int]]] = {}
    for index, page in enumerate(plan.pages):
        covered[index] = {
            (item.y + row, item.x + column)
            for item in page
            for row in range(item.height)
            for column in range(item.width)
        }
    for index, page in enumerate(raster.pages):
        if index >= len(plan.pages):
            violations.append(f"raster page {index} has no plan page")
            continue
        if (page.width, page.height) != (plan.page_width, plan.page_height):
            violations.append(
                f"page {index} is {page.width} x {page.height}, the plan declares "
                f"{plan.page_width} x {plan.page_height}"
            )
            continue
        if page.raw_bytes != page.width * page.height * page.channels:
            violations.append(f"page {index} reports {page.raw_bytes} raw bytes")
        if page.digest != page_digest(page.pixels):
            violations.append(f"page {index} digest does not describe its pixels")
        holes = covered.get(index, set())
        for row in range(page.height):
            for column in range(page.width):
                pixel = page.pixels[row][column]
                if len(pixel) != page.channels:
                    violations.append(f"page {index} pixel {column},{row} has the wrong width")
                    continue
                if (row, column) in holes:
                    continue
                if pixel != raster.background:
                    violations.append(
                        f"page {index} pixel {column},{row} is not the declared background"
                    )
        for item in plan.pages[index] if index < len(plan.pages) else ():
            image = sources[item.name]
            for offset_y in range(item.height):
                for offset_x in range(item.width):
                    got = page.pixels[item.y + offset_y][item.x + offset_x]
                    if got != tuple(image[offset_y][offset_x]):
                        violations.append(
                            f"page {index} pixel {item.x + offset_x},{item.y + offset_y} "
                            f"does not match frame {item.name}"
                        )
        if plan.padding >= 1:
            band = plan.padding - 1
            for row in range(page.height):
                for column in range(page.width):
                    if row <= band or column <= band or row >= page.height - plan.padding \
                            or column >= page.width - plan.padding:
                        if page.pixels[row][column] != raster.background:
                            violations.append(
                                f"page {index} gutter pixel {column},{row} is not background"
                            )
    return violations
