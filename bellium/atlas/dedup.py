"""Exact duplicate frames: one stored copy per distinct image, aliases for the rest.

An animation keeps its frame list; the atlas stores one copy of each distinct image
and an alias map says which stored copy every declared frame draws from. Exactness
is deliberate: two frames merge only when their dimensions, channel count and pixel
bytes are identical, so a near-duplicate is never silently dropped. Removing a
duplicate must change the bytes stored, never the pixels drawn.

All frames of one atlas must share a channel count, because a stored page has one;
a mixed RGB and RGBA set is refused rather than silently converted.

A caller may declare a bounded tolerance instead of exactness: max_delta lets two frames
merge when no single channel value differs by more than that bound. The bound is explicit,
alpha always counts, and the worst measured difference is reported. max_delta 0 means
exact matches only, which is the default.

A frame may also be stored as a mirror of another: none, flip-x, flip-y or rotate-180. Those
four keep the drawn size, unlike a quarter turn, which would need an engine that rotates a
region. The alias records the transform the caller applies when drawing.
"""

from __future__ import annotations

from dataclasses import dataclass

from bellium.atlas.packing import Frame
from bellium.atlas.raster import check_sources, page_digest, source_shape

Key = tuple[int, int, int, str]
MAX_DELTA = 255
TRANSFORMS = ("none", "flip-x", "flip-y", "rotate-180")


@dataclass(frozen=True)
class Alias:
    name: str
    canonical: str
    width: int
    height: int
    channels: int
    digest: str
    delta: int = 0
    transform: str = "none"


@dataclass(frozen=True)
class DuplicateGroup:
    canonical: str
    members: tuple[str, ...]
    pixels: int
    stored_bytes: int
    saved_bytes: int
    max_delta: int = 0
    transforms: tuple[str, ...] = ()


@dataclass(frozen=True)
class DedupReport:
    aliases: tuple[Alias, ...]
    unique: tuple[str, ...]
    groups: tuple[DuplicateGroup, ...]
    original_bytes: int
    stored_bytes: int
    max_delta: int = 0
    worst_delta: int = 0

    @property
    def saved_bytes(self) -> int:
        return self.original_bytes - self.stored_bytes

    @property
    def saved_ratio(self) -> float:
        return self.saved_bytes / self.original_bytes if self.original_bytes else 0.0

    def canonical_of(self, name: str) -> str:
        for alias in self.aliases:
            if alias.name == name:
                return alias.canonical
        raise KeyError(name)

    def to_dict(self) -> dict:
        return {
            "aliases": [
                {
                    "name": alias.name, "canonical": alias.canonical,
                    "width": alias.width, "height": alias.height,
                    "channels": alias.channels, "digest": alias.digest,
                    "delta": alias.delta,
                    "transform": alias.transform,
                }
                for alias in self.aliases
            ],
            "unique": list(self.unique),
            "groups": [
                {
                    "canonical": group.canonical, "members": list(group.members),
                    "pixels": group.pixels, "stored_bytes": group.stored_bytes,
                    "saved_bytes": group.saved_bytes,
                    "max_delta": group.max_delta,
                    "transforms": list(group.transforms),
                }
                for group in self.groups
            ],
            "original_bytes": self.original_bytes,
            "stored_bytes": self.stored_bytes,
            "saved_bytes": self.saved_bytes,
            "saved_ratio": round(self.saved_ratio, 4),
            "max_delta": self.max_delta,
            "worst_delta": self.worst_delta,
            "transforms": {
                name: sum(1 for alias in self.aliases if alias.transform == name)
                for name in TRANSFORMS
                if any(alias.transform == name for alias in self.aliases)
            },
        }


def content_key(image: object) -> Key:
    """The identity of one image: size, channel count and the digest of its bytes."""
    height, width, channels = source_shape(image)
    return width, height, channels, page_digest(image)


def apply_transform(image: object, transform: object) -> list[list[tuple[int, ...]]]:
    """Return the declared transform of an image. A copy is always returned."""
    if not isinstance(transform, str) or transform not in TRANSFORMS:
        raise ValueError(f"transform must be one of: {', '.join(TRANSFORMS)}")
    source_shape(image)
    if transform == "flip-x":
        return [[tuple(pixel) for pixel in reversed(row)] for row in image]
    if transform == "flip-y":
        return [[tuple(pixel) for pixel in row] for row in reversed(image)]
    if transform == "rotate-180":
        return [[tuple(pixel) for pixel in reversed(row)] for row in reversed(image)]
    return [[tuple(pixel) for pixel in row] for row in image]


def pixel_delta(left: object, right: object) -> int:
    """Largest absolute channel difference between two pixels, alpha included."""
    if not isinstance(left, (tuple, list)) or not isinstance(right, (tuple, list)):
        raise ValueError("pixels must be tuples or lists")
    if len(left) != len(right):
        raise ValueError("pixels must share a channel count")
    differences = [abs(int(a) - int(b)) for a, b in zip(left, right)]
    return max(differences, default=0)


def image_delta(left: object, right: object) -> int | None:
    """Worst channel difference between two images, or None when they cannot merge."""
    try:
        if source_shape(left) != source_shape(right):
            return None
    except ValueError:
        return None
    worst = 0
    for left_row, right_row in zip(left, right):
        for left_pixel, right_pixel in zip(left_row, right_row):
            difference = pixel_delta(left_pixel, right_pixel)
            if difference > worst:
                worst = difference
    return worst


def bound(max_delta: object) -> int:
    if isinstance(max_delta, bool) or not isinstance(max_delta, int):
        raise ValueError("max_delta must be an integer")
    if not 0 <= max_delta <= MAX_DELTA:
        raise ValueError(f"max_delta must be between 0 and {MAX_DELTA}")
    return max_delta


def analyze_frames(
    frames: tuple[Frame, ...],
    sources: object,
    *,
    max_delta: object = 0,
) -> DedupReport:
    tolerance = bound(max_delta)
    problems = check_sources(frames, sources)
    if problems:
        raise ValueError("; ".join(problems))
    canonicals: dict[Key, str] = {}
    keys_by_name: dict[str, Key] = {}
    images_by_name: dict[str, object] = {}
    transformed: dict[Key, tuple[str, str]] = {}
    order: list[str] = []
    members: dict[str, list[str]] = {}
    aliases: list[Alias] = []
    original_bytes = 0
    stored_bytes = 0
    for frame in frames:
        image = sources[frame.name]
        key = content_key(image)
        width, height, channels, digest = key
        original_bytes += width * height * channels
        canonical = None
        delta = 0
        transform = "none"
        if tolerance == 0:
            match = transformed.get(key)
            if match is not None:
                canonical, transform = match
        else:
            for candidate in order:
                if keys_by_name[candidate][:3] != key[:3]:
                    continue
                base = images_by_name[candidate]
                for name in TRANSFORMS:
                    measured = image_delta(apply_transform(base, name), image)
                    if measured is not None and measured <= tolerance:
                        canonical, delta, transform = candidate, measured, name
                        break
                if canonical is not None:
                    break
        if canonical is None:
            canonicals[key] = frame.name
            keys_by_name[frame.name] = key
            images_by_name[frame.name] = image
            order.append(frame.name)
            members[frame.name] = [frame.name]
            stored_bytes += width * height * channels
            for name in TRANSFORMS:
                transformed_key = (
                    width, height, channels, page_digest(apply_transform(image, name))
                )
                transformed.setdefault(transformed_key, (frame.name, name))
        else:
            members[canonical].append(frame.name)
        aliases.append(Alias(
            name=frame.name,
            canonical=canonical or frame.name,
            width=width,
            height=height,
            channels=channels,
            digest=digest,
            delta=delta,
            transform=transform,
        ))
    groups = []
    for canonical in order:
        stored = members[canonical]
        if len(stored) < 2:
            continue
        width, height, channels, _ = keys_by_name[canonical]
        deltas = [item.delta for item in aliases if item.canonical == canonical]
        shapes = [item.transform for item in aliases if item.canonical == canonical]
        groups.append(DuplicateGroup(
            canonical=canonical,
            members=tuple(stored),
            pixels=width * height,
            stored_bytes=width * height * channels,
            saved_bytes=width * height * channels * (len(stored) - 1),
            max_delta=max(deltas, default=0),
            transforms=tuple(shapes),
        ))
    worst = max((item.delta for item in aliases), default=0)
    return DedupReport(
        aliases=tuple(aliases),
        unique=tuple(order),
        groups=tuple(groups),
        original_bytes=original_bytes,
        stored_bytes=stored_bytes,
        max_delta=tolerance,
        worst_delta=worst,
    )


def unique_frames(frames: tuple[Frame, ...], report: DedupReport) -> tuple[Frame, ...]:
    stored = set(report.unique)
    return tuple(frame for frame in frames if frame.name in stored)


def check_aliases(
    frames: tuple[Frame, ...],
    sources: object,
    report: object,
) -> list[str]:
    """Recompute an alias map: every declared frame must resolve to identical pixels."""
    violations: list[str] = []
    if not isinstance(report, DedupReport):
        return ["report must be a DedupReport"]
    try:
        declared_bound = bound(report.max_delta)
    except ValueError as error:
        return [f"report max_delta is invalid: {error}"]
    worst_measured = 0

    def image_of(name: str):
        return sources.get(name) if isinstance(sources, dict) else None
    declared = {frame.name: frame for frame in frames}
    seen: dict[str, Alias] = {}
    for alias in report.aliases:
        if alias.name not in declared:
            violations.append(f"alias for an undeclared frame: {alias.name}")
            continue
        if alias.name in seen:
            violations.append(f"frame has more than one alias: {alias.name}")
            continue
        seen[alias.name] = alias
        image = image_of(alias.name)
        try:
            height, width, channels = source_shape(image)
        except ValueError as error:
            violations.append(f"source {alias.name}: {error}")
            continue
        digest = page_digest(image)
        if (alias.width, alias.height, alias.channels) != (width, height, channels):
            violations.append(f"alias {alias.name} describes the wrong size")
        if alias.digest != digest:
            violations.append(f"alias {alias.name} digest does not describe its pixels")
        if alias.canonical == alias.name and alias.delta != 0:
            violations.append(f"stored frame {alias.name} reports a non-zero delta")
        if not 0 <= alias.delta <= declared_bound:
            violations.append(
                f"alias {alias.name} reports delta {alias.delta} outside the declared bound"
            )
        if not isinstance(alias.transform, str) or alias.transform not in TRANSFORMS:
            violations.append(f"alias {alias.name} declares an unknown transform")
    for frame in frames:
        if frame.name not in seen:
            violations.append(f"frame without an alias: {frame.name}")
    stored = set(report.unique)
    if len(stored) != len(report.unique):
        violations.append("unique frame names are not unique")
    for name in report.unique:
        if name not in declared:
            violations.append(f"stored frame is not declared: {name}")
        elif seen.get(name) is not None and seen[name].canonical != name:
            violations.append(f"stored frame {name} is an alias of {seen[name].canonical}")
    for alias in seen.values():
        if alias.canonical not in stored:
            violations.append(f"alias {alias.name} points at a frame that is not stored")
            continue
        if alias.canonical not in seen:
            violations.append(f"alias {alias.name} points at a frame without an alias")
            continue
        canonical_alias = seen[alias.canonical]
        if canonical_alias.canonical != alias.canonical:
            violations.append(f"stored frame {alias.canonical} is itself an alias")
        if alias.transform not in TRANSFORMS:
            continue
        measured = image_delta(
            apply_transform(image_of(alias.canonical), alias.transform),
            image_of(alias.name),
        )
        if measured is None:
            violations.append(f"alias {alias.name} cannot be compared with {alias.canonical}")
            continue
        worst_measured = max(worst_measured, measured)
        if measured > declared_bound:
            violations.append(
                f"alias {alias.name} differs from {alias.canonical} by {measured}, "
                f"beyond the declared {declared_bound}"
            )
        if measured != alias.delta:
            violations.append(
                f"alias {alias.name} reports delta {alias.delta}, measured {measured}"
            )
        if declared_bound == 0 and alias.transform == "none" \
                and alias.digest != canonical_alias.digest:
            violations.append(
                f"alias {alias.name} does not match the pixels of {alias.canonical}"
            )
        skipped = False
        for candidate in report.unique:
            if skipped:
                break
            for name in TRANSFORMS:
                if candidate == alias.canonical and name == alias.transform:
                    skipped = True
                    break
                earlier = image_delta(
                    apply_transform(image_of(candidate), name), image_of(alias.name)
                )
                if earlier is not None and earlier <= declared_bound:
                    violations.append(
                        f"alias {alias.name} skipped the earlier stored frame {candidate} "
                        f"with transform {name}"
                    )
                    skipped = True
                    break
    expected_order = tuple(frame.name for frame in frames if frame.name in stored)
    if tuple(report.unique) != expected_order:
        violations.append("unique frames are not in declaration order")
    grouped: dict[str, list[str]] = {}
    for alias in seen.values():
        grouped.setdefault(alias.canonical, []).append(alias.name)
    reported = {group.canonical: list(group.members) for group in report.groups}
    for canonical, members in grouped.items():
        if len(members) < 2:
            if canonical in reported:
                violations.append(f"group {canonical} is not a duplicate group")
            continue
        if canonical not in reported:
            violations.append(f"missing duplicate group for {canonical}")
            continue
        if reported[canonical] != members:
            violations.append(f"group {canonical} does not list its members in order")
    for canonical, members in reported.items():
        if canonical not in grouped:
            violations.append(f"group {canonical} has no aliases")
            continue
        if len(members) < 2:
            violations.append(f"group {canonical} has fewer than two members")
        group = next(item for item in report.groups if item.canonical == canonical)
        width = seen[canonical].width
        height = seen[canonical].height
        channels = seen[canonical].channels
        if group.pixels != width * height:
            violations.append(f"group {canonical} reports the wrong pixel count")
        if group.stored_bytes != width * height * channels:
            violations.append(f"group {canonical} reports the wrong stored bytes")
        if group.saved_bytes != width * height * channels * (len(members) - 1):
            violations.append(f"group {canonical} reports the wrong savings")
        worst_group = max((seen[name].delta for name in members if name in seen), default=0)
        if group.max_delta != worst_group:
            violations.append(
                f"group {canonical} reports max_delta {group.max_delta}, measured {worst_group}"
            )
        expected_transforms = tuple(seen[name].transform for name in members if name in seen)
        if tuple(group.transforms) != expected_transforms:
            violations.append(f"group {canonical} reports the wrong transforms")
    original = 0
    stored_bytes = 0
    for frame in frames:
        alias = seen.get(frame.name)
        if alias is None:
            continue
        original += alias.width * alias.height * alias.channels
        if alias.name in stored:
            stored_bytes += alias.width * alias.height * alias.channels
    if report.original_bytes != original:
        violations.append("original_bytes does not add up")
    if report.stored_bytes != stored_bytes:
        violations.append("stored_bytes does not add up")
    saved = sum(group.saved_bytes for group in report.groups)
    if saved != report.saved_bytes:
        violations.append("saved_bytes does not match the duplicate groups")
    if report.worst_delta != worst_measured:
        violations.append(
            f"report worst_delta {report.worst_delta} does not match the measured {worst_measured}"
        )
    return violations
