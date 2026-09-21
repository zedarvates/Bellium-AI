from __future__ import annotations

from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.editing.resample import resample_bilinear, _validate_target_dims
from bellium.knn._image import Image, Rgb, clamp_rgb, shape
from bellium.micro_nn.mlp import load_mlp, predict_mlp
from bellium.nano_nn.contract import NanoBudget, inspect_model
from bellium.resources import model_path

SPECIALIST_ID = 'bellium/nano-nn/resample-edge:v0'
MODEL_PATH = model_path('nano_nn', 'resample-edge', 'v0.json')
NANO_RESAMPLE_BUDGET = NanoBudget(
    max_parameters=64,
    max_model_bytes=8192,
    precision='float-json-weights-v1',
)

def _luma(p: Rgb) -> float:
    return (2126 * p[0] + 7152 * p[1] + 722 * p[2]) / 10000.0

def nano_resample(image: Image, target_height: int, target_width: int, *, model: dict[str, Any] | None = None) -> SpecialistResult:
    in_h, in_w = shape(image)
    _validate_target_dims(target_height, target_width)
    loaded = model if model is not None else load_mlp(MODEL_PATH)
    contract = inspect_model(loaded, budget=NANO_RESAMPLE_BUDGET)
    base_bilinear = resample_bilinear(image, target_height, target_width)
    if in_h < 2 or in_w < 2:
        return SpecialistResult(
            SPECIALIST_ID,
            {'image': base_bilinear, 'refined_pixels': 0, 'model': contract},
            0.7,
            False,
            AuthorityMode.CONSULTATIVE,
            notes=('Dimensions too small for nano edge residual; used bilinear baseline.',),
        )
    out = []
    refined_count = 0
    for r in range(target_height):
        src_y = (r + 0.5) * in_h / target_height - 0.5
        y0 = max(0, min(in_h - 1, int(src_y)))
        y1 = max(0, min(in_h - 1, y0 + 1))
        row = []
        for c in range(target_width):
            src_x = (c + 0.5) * in_w / target_width - 0.5
            x0 = max(0, min(in_w - 1, int(src_x)))
            x1 = max(0, min(in_w - 1, x0 + 1))
            tl, tr = image[y0][x0], image[y0][x1]
            bl, br = image[y1][x0], image[y1][x1]
            l_tl, l_tr = _luma(tl) / 255.0, _luma(tr) / 255.0
            l_bl, l_br = _luma(bl) / 255.0, _luma(br) / 255.0
            grad_h = min(1.0, abs(l_tr - l_tl + l_br - l_bl) / 2.0)
            grad_v = min(1.0, abs(l_bl - l_tl + l_br - l_tr) / 2.0)
            if grad_h < 0.05 and grad_v < 0.05:
                row.append(base_bilinear[r][c])
                continue
            features = [l_tl, l_tr, l_bl, l_br, grad_h, grad_v]
            preds = predict_mlp(loaded, features)
            edge_res = preds[0] - 0.5
            conf = preds[1]
            if conf > 0.55 and abs(edge_res) > 0.02:
                refined_count += 1
                base_px = base_bilinear[r][c]
                scale = edge_res * 20.0 * conf
                px = (
                    clamp_rgb(base_px[0] + scale),
                    clamp_rgb(base_px[1] + scale),
                    clamp_rgb(base_px[2] + scale),
                )
                row.append(px)
            else:
                row.append(base_bilinear[r][c])
        out.append(row)
    return SpecialistResult(
        SPECIALIST_ID,
        {
            'image': out,
            'refined_pixels': refined_count,
            'total_pixels': target_height * target_width,
            'model': contract,
            'baseline': 'bilinear',
        },
        0.92,
        False,
        AuthorityMode.CONSULTATIVE,
        notes=('Nano-NN predicts high-frequency subpixel residual for sharp edge scaling.',),
    )
