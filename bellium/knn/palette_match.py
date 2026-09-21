from __future__ import annotations

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.editing.quantize import _perceptual_dist_sq, quantize_nearest
from bellium.knn._image import Image, Rgb, shape

SPECIALIST_ID = 'bellium/knn/palette-match:v0'

def match_color_knn(color: Rgb, palette: list[Rgb], k: int = 1) -> list[tuple[int, Rgb, float]]:
    if not palette:
        raise ValueError('palette must contain at least one color')
    k_val = max(1, min(len(palette), int(k)))
    scored = []
    for idx, pal in enumerate(palette):
        dist = _perceptual_dist_sq(color, pal) ** 0.5
        scored.append((idx, pal, dist))
    scored.sort(key=lambda item: item[2])
    return scored[:k_val]

def quantize_palette_knn(image: Image, palette: list[Rgb]) -> SpecialistResult:
    height, width = shape(image)
    if not palette:
        raise ValueError('palette must contain at least one color')
    quantized = quantize_nearest(image, palette)
    total_dist = 0.0
    palette_usage = {idx: 0 for idx in range(len(palette))}
    palette_map = {pal: idx for idx, pal in enumerate(palette)}
    for r in range(height):
        for c in range(width):
            orig = image[r][c]
            mapped = quantized[r][c]
            dist = _perceptual_dist_sq(orig, mapped) ** 0.5
            total_dist += dist
            idx = palette_map.get(mapped, 0)
            palette_usage[idx] += 1
    mean_error = total_dist / (height * width)
    active_colors = sum(1 for count in palette_usage.values() if count > 0)
    confidence = max(0.0, min(1.0, 1.0 - mean_error / 255.0))
    return SpecialistResult(
        SPECIALIST_ID,
        {
            'image': quantized,
            'palette_size': len(palette),
            'active_colors': active_colors,
            'mean_error': round(mean_error, 2),
            'palette_usage': palette_usage,
        },
        confidence,
        False,
        AuthorityMode.CONSULTATIVE,
        notes=('k-NN nearest exemplar palette assignment in perceptual colour space.',),
    )
