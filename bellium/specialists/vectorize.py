from __future__ import annotations

from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.editing.vectorize import rasterize_regions, regions_to_svg, vectorize_regions
from bellium.knn._image import shape
from bellium.knn.vector_region import classify_vector_region

SPECIALIST_ID = "bellium/hybrid/raster-to-svg:v0"


def raster_to_svg(query: dict[str, Any]) -> SpecialistResult:
    image = query.get("image")
    if image is None:
        raise ValueError("query needs image")
    shape(image)
    max_colors = query.get("max_colors", 8)
    min_area = query.get("min_area", 4)
    epsilon = query.get("epsilon", 0.6)
    skip_background = query.get("skip_background", True)
    pixels = [px for row in image for px in row]
    unique = len(set(pixels))
    if unique > max(max_colors * 2, 16) and unique / len(pixels) > 0.20:
        return SpecialistResult(
            SPECIALIST_ID,
            {
                "status": "abstain",
                "reason": "photographic_unique_colors",
                "unique_colors": unique,
            },
            0.0,
            True,
            AuthorityMode.CONSULTATIVE,
            notes=("A field with too many distinct colours stays raster; SVG is for hard-edge art.",),
        )
    payload = vectorize_regions(
        image,
        max_colors=max_colors,
        min_area=min_area,
        epsilon=epsilon,
        skip_background=skip_background,
        max_regions=query.get("max_regions", 64),
    )
    accepted = []
    rejected = []
    for region in payload["regions"]:
        verdict = classify_vector_region(region["features"])
        entry = {
            "color": region["color"],
            "area": region["area"],
            "knn": verdict.output,
        }
        if verdict.abstained:
            rejected.append(entry)
        else:
            accepted.append(region)
    payload["regions"] = accepted
    if not accepted:
        return SpecialistResult(
            SPECIALIST_ID,
            {
                "status": "abstain",
                "reason": "no_vector_safe_regions",
                "rejected": len(rejected),
            },
            0.0,
            True,
            AuthorityMode.CONSULTATIVE,
            notes=("Photographic or noisy fields are not converted to SVG.",),
        )
    svg = regions_to_svg(payload)
    preview = rasterize_regions(payload)
    return SpecialistResult(
        SPECIALIST_ID,
        {
            "status": "suggest",
            "svg": svg,
            "region_count": len(accepted),
            "rejected_count": len(rejected),
            "preview": preview,
            "palette": payload["palette"],
            "certified": False,
        },
        0.88,
        False,
        AuthorityMode.CONSULTATIVE,
        notes=(
            "Deterministic contours with k-NN region routing; not a generative drawing.",
            "A YouTuber pelican-on-a-bike prompt test still needs a real image model.",
        ),
    )
