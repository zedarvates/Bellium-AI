from __future__ import annotations

import math
from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.knn._image import Image, Rgb, clamp_rgb, shape
from bellium.micro_nn.mlp import load_mlp, predict_mlp
from bellium.resources import model_path

SPECIALIST_ID = 'bellium/micro-nn/tone-curve:v0'
MODEL_PATH = model_path('micro_nn', 'tone-curve', 'v0.json')

def _luma(p: Rgb) -> float:
    return (2126 * p[0] + 7152 * p[1] + 722 * p[2]) / 10000.0

def extract_tone_features(image: Image) -> list[float]:
    height, width = shape(image)
    lumas = sorted([_luma(px) for row in image for px in row])
    n = len(lumas)
    p5 = lumas[int(0.05 * n)] / 255.0
    p25 = lumas[int(0.25 * n)] / 255.0
    p50 = lumas[int(0.50 * n)] / 255.0
    p75 = lumas[int(0.75 * n)] / 255.0
    p95 = lumas[min(n - 1, int(0.95 * n))] / 255.0
    mean_l = (sum(lumas) / n) / 255.0
    std_l = math.sqrt(sum((v / 255.0 - mean_l) ** 2 for v in lumas) / n)
    dyn_range = (lumas[-1] - lumas[0]) / 255.0
    return [p5, p25, p50, p75, p95, mean_l, std_l, dyn_range]

def micro_tone_adjust(image: Image, *, model: dict[str, Any] | None = None) -> SpecialistResult:
    height, width = shape(image)
    loaded = model if model is not None else load_mlp(MODEL_PATH)
    feats = extract_tone_features(image)
    preds = predict_mlp(loaded, feats)
    gamma_norm = preds[0]
    lift = preds[1] * 0.1
    gain = 0.8 + preds[2] * 0.4
    pivot = preds[3]
    gamma = 0.5 + gamma_norm * 1.5
    lut = []
    for i in range(256):
        x = i / 255.0
        x_adj = (x ** (1.0 / gamma)) * gain + lift
        lut.append(clamp_rgb(x_adj * 255.0))
    out = []
    for r in range(height):
        row = []
        for c in range(width):
            px = image[r][c]
            row.append((lut[px[0]], lut[px[1]], lut[px[2]]))
        out.append(row)
    return SpecialistResult(
        SPECIALIST_ID,
        {
            'image': out,
            'curve_parameters': {
                'gamma': round(gamma, 4),
                'lift': round(lift, 4),
                'gain': round(gain, 4),
                'pivot': round(pivot, 4),
            },
            'features': [round(f, 4) for f in feats],
        },
        0.94,
        False,
        AuthorityMode.CONSULTATIVE,
        notes=('Micro-NN predicts smooth parametric tone curve from global luminance distribution.',),
    )
