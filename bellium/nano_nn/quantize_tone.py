from __future__ import annotations

import math
from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.editing.quantize import _perceptual_dist_sq, quantize_floyd_steinberg
from bellium.knn._image import Image, Rgb, shape
from bellium.micro_nn.mlp import load_mlp, predict_mlp
from bellium.nano_nn.contract import NanoBudget, inspect_model
from bellium.resources import model_path

SPECIALIST_ID = 'bellium/nano-nn/quantize-tone:v0'
MODEL_PATH = model_path('nano_nn', 'quantize-tone', 'v0.json')
QUANTIZE_TONE_BUDGET = NanoBudget(
    max_parameters=48,
    max_model_bytes=4096,
    precision='float-json-weights-v1',
)

def _luma(p: Rgb) -> float:
    return (2126 * p[0] + 7152 * p[1] + 722 * p[2]) / 10000.0

def nano_quantize(
    image: Image,
    palette: list[Rgb],
    *,
    model: dict[str, Any] | None = None,
) -> SpecialistResult:
    height, width = shape(image)
    if not palette:
        raise ValueError('palette must contain at least one color')
    loaded = model if model is not None else load_mlp(MODEL_PATH)
    contract = inspect_model(loaded, budget=QUANTIZE_TONE_BUDGET)
    luma_vals = [_luma(px) for row in image for px in row]
    mean_luma = sum(luma_vals) / len(luma_vals)
    var_luma = sum((v - mean_luma) ** 2 for v in luma_vals) / len(luma_vals)
    norm_var = min(1.0, math.sqrt(var_luma) / 128.0)
    norm_mean = min(1.0, mean_luma / 255.0)
    diff_sum = 0.0
    count = 0
    for r in range(height):
        for c in range(width - 1):
            diff_sum += abs(_luma(image[r][c + 1]) - _luma(image[r][c]))
            count += 1
    grad_mag = min(1.0, (diff_sum / max(1, count)) / 64.0)
    min_dist_sum = 0.0
    sample_count = min(100, height * width)
    step = max(1, (height * width) // sample_count)
    flat_pixels = [px for row in image for px in row]
    for i in range(0, len(flat_pixels), step):
        px = flat_pixels[i]
        d = min(_perceptual_dist_sq(px, pal) for pal in palette)
        min_dist_sum += math.sqrt(d)
    dist_centroid = min(1.0, (min_dist_sum / sample_count) / 128.0)
    features = [norm_var, grad_mag, dist_centroid, norm_mean]
    preds = predict_mlp(loaded, features)
    dither_intensity = float(preds[0])
    quantized = quantize_floyd_steinberg(image, palette, dither_strength=dither_intensity)
    return SpecialistResult(
        SPECIALIST_ID,
        {
            'image': quantized,
            'dither_intensity': round(dither_intensity, 4),
            'features': {
                'luma_variance': round(norm_var, 4),
                'gradient_magnitude': round(grad_mag, 4),
                'dist_centroid': round(dist_centroid, 4),
                'luma_mean': round(norm_mean, 4),
            },
            'model': contract,
        },
        0.91,
        False,
        AuthorityMode.CONSULTATIVE,
        notes=('Nano-NN modulates dithering intensity adaptively to prevent banding without flat-area noise.',),
    )
