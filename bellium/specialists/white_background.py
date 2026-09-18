"""White-background normalizer: cutout then composite. Rejects unsafe cases."""

from __future__ import annotations

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.knn._image import Image, shape
from bellium.knn.color_cutout import cutout, to_white_background

SPECIALIST_ID = "bellium/hybrid/white-background-normalizer:v0"


def _crop(image: Image, alpha: list[list[float]], margin: int = 1) -> Image:
    height, width = shape(image)
    rows = [r for r in range(height) if any(alpha[r])]
    cols = [c for c in range(width) if any(alpha[r][c] for r in range(height))]
    if not rows or not cols:
        return image
    r0 = max(0, min(rows) - margin)
    r1 = min(height - 1, max(rows) + margin)
    c0 = max(0, min(cols) - margin)
    c1 = min(width - 1, max(cols) + margin)
    return [row[c0:c1 + 1] for row in image[r0:r1 + 1]]


def normalize_white_background(image: Image) -> SpecialistResult:
    cut = cutout(image)
    if cut.abstained:
        return SpecialistResult(
            SPECIALIST_ID,
            {"reason": "cutout_abstained", "cutout": cut.output},
            cut.confidence,
            True,
            AuthorityMode.CONSULTATIVE,
            notes=("Unsafe to force a white background.",),
        )
    rgba = cut.output["rgba"]
    white = to_white_background(rgba)
    cropped = _crop(white, cut.output["alpha"])
    return SpecialistResult(
        SPECIALIST_ID,
        {
            "image": cropped,
            "foreground_ratio": cut.output["foreground_ratio"],
        },
        cut.confidence,
        False,
        AuthorityMode.CONSULTATIVE,
        notes=("Deterministic composite after a consultative color k-NN cutout.",),
    )

