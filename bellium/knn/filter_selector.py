from __future__ import annotations

import json
import math
from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.editing.convolve import gaussian_blur, sharpen, sobel_edges, emboss
from bellium.knn._image import Image, Rgb, shape
from bellium.resources import model_path

SPECIALIST_ID = 'bellium/knn/filter-selector:v0'
PRESETS_PATH = model_path('knn', 'visual', 'filter-presets-v0.json')

def _luma(p: Rgb) -> float:
    return (2126 * p[0] + 7152 * p[1] + 722 * p[2]) / 10000.0

def extract_filter_features(image: Image) -> list[float]:
    height, width = shape(image)
    lumas = [_luma(px) for row in image for px in row]
    mean_luma = sum(lumas) / len(lumas)
    var_luma = sum((v - mean_luma) ** 2 for v in lumas) / len(lumas)
    norm_var = min(1.0, math.sqrt(var_luma) / 128.0)
    diff_h = sum(abs(_luma(image[r][c + 1]) - _luma(image[r][c])) for r in range(height) for c in range(width - 1))
    diff_v = sum(abs(_luma(image[r + 1][c]) - _luma(image[r][c])) for r in range(height - 1) for c in range(width))
    total_pairs = max(1, height * (width - 1) + (height - 1) * width)
    mean_grad = min(1.0, ((diff_h + diff_v) / total_pairs) / 64.0)
    edge_count = 0
    for r in range(height - 1):
        for c in range(width - 1):
            if abs(_luma(image[r][c + 1]) - _luma(image[r][c])) > 30.0:
                edge_count += 1
    edge_density = min(1.0, edge_count / max(1, (height - 1) * (width - 1)))
    min_v, max_v = min(lumas), max(lumas)
    dyn_range = min(1.0, (max_v - min_v) / 255.0)
    hf_energy = min(1.0, (mean_grad * 1.5 + edge_density) / 2.0)
    noise_proxy = max(0.0, min(1.0, (mean_grad - edge_density * 0.8) * 1.2))
    return [mean_grad, norm_var, edge_density, hf_energy, dyn_range, noise_proxy]

def select_filter_knn(image: Image, *, presets: list[dict[str, Any]] | None = None) -> SpecialistResult:
    feats = extract_filter_features(image)
    if presets is None:
        if PRESETS_PATH.is_file():
            data = json.loads(PRESETS_PATH.read_text(encoding='utf-8'))
            presets = data.get('presets', [])
        else:
            presets = []
    if not presets:
        return SpecialistResult(
            SPECIALIST_ID,
            {'status': 'abstain', 'reason': 'no_presets_found', 'features': feats},
            0.0,
            True,
            AuthorityMode.CONSULTATIVE,
        )
    best_dist = float('inf')
    best_preset = presets[0]
    for p in presets:
        pf = p['features']
        dist = math.sqrt(sum((feats[i] - pf[i]) ** 2 for i in range(len(feats))))
        if dist < best_dist:
            best_dist = dist
            best_preset = p
    conf = max(0.4, min(0.95, 1.0 - best_dist / 2.0))
    action = best_preset.get('label', 'pass_through')
    if action == 'denoise_soft':
        filtered = gaussian_blur(image, radius=1, sigma=0.8)
    elif action == 'sharpen_subtle':
        filtered = sharpen(image, strength=0.5)
    elif action == 'edge_accentuate':
        filtered = sobel_edges(image)
    elif action == 'emboss_texture':
        filtered = emboss(image)
    else:
        filtered = [list(r) for r in image]
    return SpecialistResult(
        SPECIALIST_ID,
        {
            'recommended_filter': action,
            'exemplar_id': best_preset.get('id'),
            'distance': round(best_dist, 4),
            'features': [round(f, 4) for f in feats],
            'image': filtered,
        },
        round(conf, 4),
        False,
        AuthorityMode.CONSULTATIVE,
        notes=('k-NN exemplar selection of optimal spatial filtering operation.',),
    )
