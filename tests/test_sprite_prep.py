from copy import deepcopy

from bellium.knn.sprite_anchor import propose_anchor
from bellium.specialists.sprite_prep import prepare_sprite_frame

SIZE = 32


def _humanoid(scale=1.0):
    mask = [[0] * SIZE for _ in range(SIZE)]
    half = int(round(9 * scale))
    top = int(round(16 - 11 * scale))
    bottom = int(round(16 + 11 * scale))
    for r in range(top, bottom):
        for c in range(16 - half // 2, 16 + half // 2):
            mask[r][c] = 1
    for r in range(top, top + max(2, int(round(6 * scale)))):
        for c in range(16 - half, 16 + half):
            mask[r][c] = 1
    return mask


def _image(mask, foreground=(210, 60, 60), background=(255, 255, 255)):
    return [[foreground if value else background for value in row] for row in mask]


def test_anchor_is_proposed_for_a_known_family() -> None:
    mask = _humanoid(scale=0.9)
    result = propose_anchor({"family": "character", "image": _image(mask), "mask": mask})
    assert result.abstained is False
    assert result.output["anchor"] == "feet"
    assert result.output["pixels_changed"] == 0


def test_frame_is_built_without_touching_the_source() -> None:
    mask = _humanoid(scale=0.9)
    image = _image(mask)
    original = deepcopy(image)
    result = prepare_sprite_frame({"family": "character", "image": image, "mask": mask})
    assert result.abstained is False
    assert result.output["status"] == "ready"
    assert result.output["source_preserved"] is True
    assert image == original
    frame = result.output["image"]
    transform = result.output["transform"]
    assert len(frame) == transform["frame_size"][1]
    assert len(frame[0]) == transform["frame_size"][0]
    assert transform["pad_color"] == (255, 255, 255)
    crop = transform["crop"]
    pad = transform["pad_px"]
    expected = [row[crop["x"]:crop["x"] + crop["w"]] for row in image[crop["y"]:crop["y"] + crop["h"]]]
    inside = [row[pad:pad + crop["w"]] for row in frame[pad:pad + crop["h"]]]
    assert inside == expected


def test_clipped_silhouette_refuses_preparation() -> None:
    mask = [[1] * SIZE for _ in range(SIZE)]
    result = prepare_sprite_frame({"family": "character", "image": _image(mask), "mask": mask})
    assert result.abstained is False
    assert result.output["status"] == "unsafe"
    assert result.output["reason"] == "content_touches_frame"
    assert result.output["image"] is None


def test_empty_mask_abstains() -> None:
    mask = [[0] * SIZE for _ in range(SIZE)]
    result = prepare_sprite_frame({"family": "character", "image": _image(mask), "mask": mask})
    assert result.abstained is True
    assert result.output["reason"] == "empty_foreground"


def test_unknown_family_abstains() -> None:
    mask = _humanoid()
    result = prepare_sprite_frame({"family": "portrait", "image": _image(mask), "mask": mask})
    assert result.abstained is True
    assert result.output["reason"] == "unknown_family"


def test_missing_mask_is_refused() -> None:
    mask = _humanoid()
    blocked = False
    try:
        prepare_sprite_frame({"family": "character", "image": _image(mask)})
    except ValueError:
        blocked = True
    assert blocked
