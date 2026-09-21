from __future__ import annotations

import json
from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.editing.resample import resample_bilinear, _validate_target_dims
from bellium.knn._image import Image, Rgb, clamp_rgb, shape
from bellium.resources import model_path

SPECIALIST_ID = 'bellium/knn/resample-knn:v0'
PATTERN_PATH = model_path('knn', 'visual', 'resample-patterns-v0.json')

def _luma(p: Rgb) -> float:
    return (2126 * p[0] + 7152 * p[1] + 722 * p[2]) / 10000.0

def load_patterns() -> list[dict[str, Any]]:
    if not PATTERN_PATH.is_file():
        return []
    data = json.loads(PATTERN_PATH.read_text(encoding='utf-8'))
    return data.get('patterns', [])

def knn_resample(image: Image, target_height: int, target_width: int, *, patterns: list[dict[str, Any]] | None = None) -> SpecialistResult:
    in_h, in_w = shape(image)
    _validate_target_dims(target_height, target_width)
    pats = patterns if patterns is not None else load_patterns()
    bilinear_base = resample_bilinear(image, target_height, target_width)
    if not pats or in_h < 2 or in_w < 2:
        return SpecialistResult(
            SPECIALIST_ID,
            {'image': bilinear_base, 'method': 'bilinear_fallback', 'matched_patterns': 0},
            0.75,
            False,
            AuthorityMode.CONSULTATIVE,
            notes=('Fallback to bilinear due to small dimensions or missing patterns.',),
        )
    out = []
    matched_count = 0
    for r in range(target_height):
        src_y = (r + 0.5) * in_h / target_height - 0.5
        y0 = max(0, min(in_h - 1, int(src_y)))
        y1 = max(0, min(in_h - 1, y0 + 1))
        row = []
        for c in range(target_width):
            src_x = (c + 0.5) * in_w / target_width - 0.5
            x0 = max(0, min(in_w - 1, int(src_x)))
            x1 = max(0, min(in_w - 1, x0 + 1))
            tl = image[y0][x0]
            tr = image[y0][x1]
            bl = image[y1][x0]
            br = image[y1][x1]
            l_tl, l_tr = _luma(tl) / 255.0, _luma(tr) / 255.0
            l_bl, l_br = _luma(bl) / 255.0, _luma(br) / 255.0
            grad_h = abs(l_tr - l_tl + l_br - l_bl) / 2.0
            grad_v = abs(l_bl - l_tl + l_br - l_tr) / 2.0
            features = [l_tl, l_tr, l_bl, l_br]
            best_dist = float('inf')
            best_pat = None
            for pat in pats:
                pf = pat['features']
                dist = sum((features[i] - pf[i]) ** 2 for i in range(4))
                if dist < best_dist:
                    best_dist = dist
                    best_pat = pat
            if best_pat and best_dist < 0.1 and (grad_h > 0.15 or grad_v > 0.15):
                w = best_pat['weights']
                matched_count += 1
                px = []
                for ch in range(3):
                    val = tl[ch] * w[0] + tr[ch] * w[1] + bl[ch] * w[2] + br[ch] * w[3]
                    px.append(clamp_rgb(val))
                row.append((px[0], px[1], px[2]))
            else:
                row.append(bilinear_base[r][c])
        out.append(row)
    confidence = min(0.95, 0.80 + (matched_count / (target_height * target_width)) * 0.15)
    return SpecialistResult(
        SPECIALIST_ID,
        {
            'image': out,
            'method': 'knn_exemplar_resample',
            'matched_patterns': matched_count,
            'total_pixels': target_height * target_width,
        },
        confidence,
        False,
        AuthorityMode.CONSULTATIVE,
        notes=('k-NN subpixel exemplar lookup preserves oriented edge contrast.',),
    )
