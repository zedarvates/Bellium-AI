"""Downsampled previews, kept explicitly distinct from the full-resolution export.

Area averaging is the only resampling here: it is deterministic, it does not
invent detail, and it makes the preview scale reportable. A preview is never an
export and the export is never a preview.
"""

from __future__ import annotations

from bellium.editing.fast import apply_filter_fast, downsample_fast, numpy_available
from bellium.editing.filters import apply_filter
from bellium.editing.selection import apply_mask
from bellium.editing.stack import EditStack
from bellium.knn._image import Image, clamp_rgb, shape


def downsample(image: Image, *, max_side: int) -> dict:
    """Area-average an image down to a bounded side, reporting the scale used."""
    height, width = shape(image)
    if isinstance(max_side, bool) or not isinstance(max_side, int) or max_side < 4:
        raise ValueError("max_side must be an integer of at least four")
    if max(height, width) <= max_side:
        return {"image": [list(row) for row in image], "scale": 1.0,
                "size": (width, height), "resampled": False}
    scale = max_side / float(max(height, width))
    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))
    reduced = []
    for r in range(target_height):
        row = []
        r0 = int(r * height / target_height)
        r1 = max(r0 + 1, int((r + 1) * height / target_height))
        for c in range(target_width):
            c0 = int(c * width / target_width)
            c1 = max(c0 + 1, int((c + 1) * width / target_width))
            totals = [0, 0, 0]
            count = 0
            for rr in range(r0, min(r1, height)):
                for cc in range(c0, min(c1, width)):
                    pixel = image[rr][cc]
                    totals[0] += pixel[0]
                    totals[1] += pixel[1]
                    totals[2] += pixel[2]
                    count += 1
            row.append((
                clamp_rgb(totals[0] / count),
                clamp_rgb(totals[1] / count),
                clamp_rgb(totals[2] / count),
            ))
        reduced.append(row)
    return {
        "image": reduced,
        "scale": round(scale, 6),
        "size": (target_width, target_height),
        "resampled": True,
    }


def downsample_with(image: Image, *, max_side: int, backend: str = "plain") -> dict:
    """Downsample through the declared backend, so a preview can stay interactive."""
    if backend not in ("plain", "fast"):
        raise ValueError("backend must be plain or fast")
    if backend == "fast" and numpy_available():
        return downsample_fast(image, max_side=max_side)
    return downsample(image, max_side=max_side)


def downsample_mask(mask: list[list[float]], *, size: tuple[int, int]) -> list[list[float]]:
    """Area-average a mask to the preview size, so a feather survives the preview."""
    if not isinstance(mask, list) or not mask or not mask[0]:
        raise ValueError("mask is empty")
    height = len(mask)
    width = len(mask[0])
    target_width, target_height = size
    reduced = []
    for r in range(target_height):
        row = []
        r0 = int(r * height / target_height)
        r1 = max(r0 + 1, int((r + 1) * height / target_height))
        for c in range(target_width):
            c0 = int(c * width / target_width)
            c1 = max(c0 + 1, int((c + 1) * width / target_width))
            total = 0.0
            count = 0
            for rr in range(r0, min(r1, height)):
                for cc in range(c0, min(c1, width)):
                    total += mask[rr][cc]
                    count += 1
            row.append(round(total / count, 6))
        reduced.append(row)
    return reduced


def preview_stack(stack: EditStack, *, max_side: int, backend: str = "plain") -> dict:
    """Apply the active steps at preview resolution, never as an export."""
    if not isinstance(stack, EditStack):
        raise ValueError("preview_stack expects an EditStack")
    if backend not in ("plain", "fast"):
        raise ValueError("backend must be plain or fast")
    if backend == "fast" and numpy_available():
        # One conversion for the whole preview: converting per step would cost
        # more than the arithmetic it accelerates.
        session = PreviewSession(stack.source_view(), backend="fast")
        return session.preview(stack.active_steps(), max_side=max_side)
    source = stack.source_view()
    reduced = downsample_with(source, max_side=max_side, backend=backend)
    height, width = shape(reduced["image"])
    working = reduced["image"]
    for step in stack.active_steps():
        filtered = (
            apply_filter_fast(working, step.name, **step.parameters)
            if backend == "fast"
            else apply_filter(working, step.name, **step.parameters)
        )
        if step.mask is not None:
            filtered = apply_mask(working, filtered, downsample_mask(
                step.mask, size=(width, height)
            ))
        working = filtered
    return {
        "image": working,
        "scale": reduced["scale"],
        "size": reduced["size"],
        "resampled": reduced["resampled"],
        "preview_only": True,
        "export_size": shape(source)[::-1],
    }


class PreviewSession:
    """A source kept in the array representation, so repeated previews stay cheap.

    The one-time conversion is the honest cost: a single preview of a large image
    pays it, an editing session pays it once. Without NumPy the session falls back
    to the plain pipeline, which is slower and says so through its backend.
    """

    def __init__(self, source: Image, *, backend: str = "fast") -> None:
        if backend not in ("plain", "fast"):
            raise ValueError("backend must be plain or fast")
        shape(source)
        self.backend = backend if (backend == "plain" or numpy_available()) else "plain"
        self._source = source if self.backend == "plain" else None
        self._array = None
        if self.backend == "fast":
            from bellium.editing.fast import to_array  # noqa: PLC0415

            self._array = to_array(source)
        self._size = shape(source)

    def preview(self, steps, *, max_side: int) -> dict:
        """Apply declared steps at preview resolution, never as an export."""
        steps = tuple(steps)
        if self.backend == "fast":
            from bellium.editing.fast import (  # noqa: PLC0415
                filter_array,
                from_array,
                reduce_array,
            )

            working, scale, size, resampled = reduce_array(
                self._array, max_side=max_side
            )
            for step in steps:
                filtered = filter_array(working, step.name, **step.parameters)
                if step.mask is not None:
                    mask_array, _, _, _ = reduce_array(
                        _mask_to_array(step.mask), max_side=max_side, round_values=False
                    )
                    filtered = _blend_arrays(working, filtered, mask_array)
                working = filtered
            return {
                "image": from_array(working),
                "scale": scale,
                "size": size,
                "resampled": resampled,
                "preview_only": True,
                "export_size": (self._size[1], self._size[0]),
                "backend": "fast",
            }
        stack = EditStack(self._source)
        for step in steps:
            stack.add(step)
        result = preview_stack(stack, max_side=max_side, backend="plain")
        result["backend"] = "plain"
        return result


def _mask_to_array(mask: list[list[float]]):
    from bellium.editing.fast import numpy_module  # noqa: PLC0415

    numpy = numpy_module()
    if numpy is None:
        raise ValueError("the array path needs NumPy")
    return numpy.asarray(mask, dtype=numpy.float64)


def _blend_arrays(base, filtered, mask_array):
    """Blend two arrays by a mask, the array twin of apply_mask."""
    from bellium.editing.fast import numpy_module  # noqa: PLC0415

    numpy = numpy_module()
    weight = mask_array[:, :, None]
    blended = numpy.round(base * (1.0 - weight) + filtered * weight)
    return numpy.clip(blended, 0.0, 255.0)
