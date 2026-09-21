from __future__ import annotations

import math
from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.editing.convolve import gaussian_blur, sharpen
from bellium.knn._image import Image, Rgb, clamp_rgb, shape
from bellium.micro_nn.mlp import load_mlp, predict_mlp
from bellium.nano_nn.contract import NanoBudget, inspect_model
from bellium.resources import model_path

SPECIALIST_ID = 'bellium/nano-nn/adaptive-filter:v0'
MODEL_PATH = model_path('nano_nn', 'adaptive-filter', 'v0.json')
ADAPTIVE_FILTER_BUDGET = NanoBudget(
    max_parameters=48,
    max_model_bytes=4096,
    precision='float-json-weights-v1',
)

def _luma(p: Rgb) -> float:
    return (2126 * p[0] + 7152 * p[1] + 722 * p[2]) / 10000.0

def nano_adaptive_filter(image: Image, *, model: dict[str, Any] | None = None) -> SpecialistResult:
    height, width = shape(image)
    loaded = model if model is not None else load_mlp(MODEL_PATH)
    contract = inspect_model(loaded, budget=ADAPTIVE_FILTER_BUDGET)
    blurred = gaussian_blur(image, radius=1, sigma=0.8)
    sharpened = sharpen(image, strength=0.8)
    out = []
    total_sharp_weight = 0.0
    for r in range(height):
        row = []
        for c in range(width):
            y_min, y_max = max(0, r - 1), min(height, r + 2)
            x_min, x_max = max(0, c - 1), min(width, c + 2)
            patch_lumas = [_luma(image[y][x]) for y in range(y_min, y_max) for x in range(x_min, x_max)]
            mean_l = sum(patch_lumas) / len(patch_lumas)
            var_l = sum((v - mean_l) ** 2 for v in patch_lumas) / len(patch_lumas)
            norm_var = min(1.0, math.sqrt(var_l) / 64.0)
            center_l = _luma(image[r][c])
            diff = sum(abs(v - center_l) for v in patch_lumas) / len(patch_lumas)
            grad_mag = min(1.0, diff / 48.0)
            hf_energy = min(1.0, abs(center_l - mean_l) / 32.0)
            noise_proxy = max(0.0, min(1.0, norm_var * 0.8 - grad_mag * 0.5))
            features = [grad_mag, norm_var, hf_energy, noise_proxy]
            preds = predict_mlp(loaded, features)
            w_sharp = float(preds[0])
            w_smooth = float(preds[1])
            total = w_sharp + w_smooth or 1.0
            w_sharp /= total
            w_smooth /= total
            total_sharp_weight += w_sharp
            p_blur = blurred[r][c]
            p_sharp = sharpened[r][c]
            blended = (
                clamp_rgb(p_sharp[0] * w_sharp + p_blur[0] * w_smooth),
                clamp_rgb(p_sharp[1] * w_sharp + p_blur[1] * w_smooth),
                clamp_rgb(p_sharp[2] * w_sharp + p_blur[2] * w_smooth),
            )
            row.append(blended)
        out.append(row)
    avg_sharp = total_sharp_weight / (height * width)
    return SpecialistResult(
        SPECIALIST_ID,
        {
            'image': out,
            'average_sharpness_weight': round(avg_sharp, 4),
            'average_smoothing_weight': round(1.0 - avg_sharp, 4),
            'model': contract,
        },
        0.93,
        False,
        AuthorityMode.CONSULTATIVE,
        notes=('Nano-NN adaptively blends sharpening on structural edges and smoothing on noisy textures.',),
    )
