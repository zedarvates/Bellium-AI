"""Micro-NN that decides whether patch k-NN inpainting is enough."""

from __future__ import annotations


from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.resources import model_path
from bellium.knn._image import Image, Mask, shape, validate_mask
from bellium.micro_nn.features import featurize
from bellium.micro_nn.mlp import load_mlp, predict_mlp

SPECIALIST_ID = "bellium/micro-nn/inpaint-router:v0"
MODEL_PATH = model_path() / "micro_nn" / "inpaint-router" / "v0.json"
LABELS = ("patch_knn", "escalate")


def _components(mask: Mask) -> int:
    height, width = len(mask), len(mask[0])
    seen = [[False] * width for _ in range(height)]
    count = 0
    for r in range(height):
        for c in range(width):
            if not mask[r][c] or seen[r][c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r][c] = True
            while stack:
                cr, cc = stack.pop()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr][nc] and not seen[nr][nc]:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
    return count


def mask_features(image: Image, mask: Mask) -> dict[str, float]:
    validate_mask(image, mask)
    height, width = shape(image)
    total = height * width
    coords = [(r, c) for r in range(height) for c in range(width) if mask[r][c]]
    area = len(coords) / total if total else 0.0
    if not coords:
        return {
            "mask_area_ratio": 0.0,
            "bbox_fill": 0.0,
            "border_touch": 0.0,
            "neighbor_std": 0.0,
            "hole_count_norm": 0.0,
            "max_span_ratio": 0.0,
        }
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    bbox = (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
    border = any(r in (0, height - 1) or c in (0, width - 1) for r, c in coords)
    neighbors = []
    for r, c in coords:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and not mask[nr][nc]:
                neighbors.append(image[nr][nc])
    if neighbors:
        means = [sum(pixel[i] for pixel in neighbors) / len(neighbors) for i in range(3)]
        var = sum((pixel[i] - means[i]) ** 2 for pixel in neighbors for i in range(3)) / (len(neighbors) * 3)
        std = (var ** 0.5) / 255.0
    else:
        std = 1.0
    span = max(max(rows) - min(rows) + 1, max(cols) - min(cols) + 1)
    return {
        "mask_area_ratio": area,
        "bbox_fill": len(coords) / bbox,
        "border_touch": 1.0 if border else 0.0,
        "neighbor_std": min(std, 1.0),
        "hole_count_norm": min(_components(mask) / 5.0, 1.0),
        "max_span_ratio": span / max(height, width),
    }


def route_inpaint(image: Image, mask: Mask) -> SpecialistResult:
    values = mask_features(image, mask)
    vector = featurize("inpaint_router", values)
    probs = predict_mlp(load_mlp(MODEL_PATH), vector)
    best = max(range(len(probs)), key=probs.__getitem__)
    confidence = float(probs[best])
    abstained = confidence < 0.55
    return SpecialistResult(
        specialist_id=SPECIALIST_ID,
        output={
            "label": None if abstained else LABELS[best],
            "probabilities": {LABELS[i]: float(probs[i]) for i in range(len(LABELS))},
            "features": values,
        },
        confidence=round(confidence, 4),
        abstained=abstained,
        authority_mode=AuthorityMode.CONSULTATIVE,
        evidence_ref=str(MODEL_PATH.name),
        notes=("Router proposes patch_knn or escalate; it never paints pixels.",),
    )

