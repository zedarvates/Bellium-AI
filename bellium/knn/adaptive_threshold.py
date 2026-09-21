from __future__ import annotations

import json
import math
from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.editing.morphology import adaptive_local_threshold, otsu_threshold
from bellium.knn._image import Image, Rgb, shape
from bellium.resources import model_path

SPECIALIST_ID = 'bellium/knn/adaptive-threshold:v0'
THRESH_PATH = model_path('knn', 'visual', 'threshold-patterns-v0.json')

def _luma(p: Rgb) -> float:
    return (2126 * p[0] + 7152 * p[1] + 722 * p[2]) / 10000.0

def extract_threshold_features(image: Image) -> list[float]:
    height, width = shape(image)
    lumas = [_luma(px) for row in image for px in row]
    mean_l = sum(lumas) / len(lumas)
    var_l = sum((v - mean_l) ** 2 for v in lumas) / len(lumas)
    norm_var = min(1.0, math.sqrt(var_l) / 128.0)
    hist = [0] * 16
    for luma_val in lumas:
        b = min(15, int(luma_val / 16.0))
        hist[b] += 1
    peaks = sum(1 for i in range(1, 15) if hist[i] > hist[i - 1] and hist[i] > hist[i + 1])
    bimodality = min(1.0, peaks / 3.0)
    diff_h = sum(abs(_luma(image[r][c + 1]) - _luma(image[r][c])) for r in range(height) for c in range(width - 1))
    grad_density = min(1.0, (diff_h / max(1, height * (width - 1))) / 48.0)
    norm_mean = mean_l / 255.0
    return [norm_mean, norm_var, bimodality, grad_density]

def knn_threshold(image: Image, *, patterns: list[dict[str, Any]] | None = None) -> SpecialistResult:
    feats = extract_threshold_features(image)
    if patterns is None:
        if THRESH_PATH.is_file():
            data = json.loads(THRESH_PATH.read_text(encoding='utf-8'))
            patterns = data.get('patterns', [])
        else:
            patterns = []
    if not patterns:
        mask, t = otsu_threshold(image)
        return SpecialistResult(
            SPECIALIST_ID,
            {'mask': mask, 'strategy': 'otsu_default', 'threshold': t},
            0.7,
            False,
            AuthorityMode.CONSULTATIVE,
        )
    best_dist = float('inf')
    best_pat = patterns[0]
    for p in patterns:
        pf = p['features']
        d = math.sqrt(sum((feats[i] - pf[i]) ** 2 for i in range(len(feats))))
        if d < best_dist:
            best_dist = d
            best_pat = p
    strat = best_pat.get('strategy', 'otsu')
    if feats[1] < 0.01:  # norm_var < 0.01 means flat/uniform image
        strat = 'abstain'
    if strat == 'abstain':
        return SpecialistResult(
            SPECIALIST_ID,
            {'status': 'abstain', 'reason': 'image_lacks_contrast_for_binarization', 'features': feats},
            0.0,
            True,
            AuthorityMode.CONSULTATIVE,
            notes=('Uniform or flat image lacks sufficient bimodal contrast.',),
        )
    if strat == 'adaptive_local':
        mask = adaptive_local_threshold(image, window_size=5, c_offset=5.0)
        thresh_info = {'window_size': 5, 'c_offset': 5.0}
    elif strat == 'high_contrast':
        mask, t = otsu_threshold(image)
        thresh_info = {'threshold': t, 'mode': 'high_contrast_otsu'}
    else:
        mask, t = otsu_threshold(image)
        thresh_info = {'threshold': t, 'mode': 'standard_otsu'}
    conf = max(0.5, min(0.96, 1.0 - best_dist / 2.0))
    return SpecialistResult(
        SPECIALIST_ID,
        {
            'mask': mask,
            'strategy': strat,
            'matched_exemplar': best_pat.get('id'),
            'threshold_info': thresh_info,
            'features': [round(f, 4) for f in feats],
        },
        round(conf, 4),
        False,
        AuthorityMode.CONSULTATIVE,
        notes=('k-NN classifies document/silhouette characteristics to choose optimal binarization.',),
    )
