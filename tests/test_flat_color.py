from copy import deepcopy

from bellium.knn.flat_color import classify_flat, load_flats
from bellium.specialists.flat_color import flatten_region


def _solid(color=(200, 40, 40)):
    return [[color for _ in range(12)] for _ in range(12)]


def _noisy():
    image = []
    for r in range(12):
        row = []
        for c in range(12):
            v = (r * 17 + c * 13) % 255
            row.append((v, (v * 3) % 255, 255 - v))
        image.append(row)
    return image


def test_uniform_field_is_flat() -> None:
    result = classify_flat({"image": _solid()})
    assert result.abstained is False
    assert result.output["label"] == "flat"


def test_noisy_field_is_textured() -> None:
    result = classify_flat({"image": _noisy()})
    assert result.output["label"] == "textured"


def test_flat_hole_fills_with_median() -> None:
    image = _solid()
    mask = [[0] * 12 for _ in range(12)]
    mask[5][5] = 1
    image[5][5] = (0, 0, 0)
    original = deepcopy(image)
    result = flatten_region({"image": image, "mask": mask})
    assert result.abstained is False
    assert result.output["filled"] == 1
    assert result.output["color"] == (200, 40, 40)
    assert result.output["image"][5][5] == (200, 40, 40)
    assert image == original


def test_textured_hole_is_not_flattened() -> None:
    image = _noisy()
    mask = [[0] * 12 for _ in range(12)]
    mask[4][4] = 1
    result = flatten_region({"image": image, "mask": mask})
    assert result.abstained is True
    assert result.output["reason"] == "region_not_flat"
    assert result.output["filled"] == 0


def test_missing_mask_value_is_not_zeroed() -> None:
    blocked = False
    try:
        flatten_region({"image": _solid()})
    except ValueError:
        blocked = True
    assert blocked


def test_raw_pixels_cannot_be_kept(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        '{"schema":"bellium.flat-color-memory/v1","items":[{'
        '"id":"x","label":"flat","source":"x","raw_image_stored":true,'
        '"features":{"luma_std":0.0,"chroma":0.1,"chroma_std":0.0,'
        '"unique_ratio":0.01,"channel_span":0.02}}]}',
        encoding="utf-8",
    )
    blocked = False
    try:
        load_flats(path)
    except ValueError:
        blocked = True
    assert blocked

