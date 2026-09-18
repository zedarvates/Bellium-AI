from copy import deepcopy

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.specialists import texture_tile
from bellium.specialists.texture_tile import fix_texture_seams


def _gray(value):
    level = max(0, min(255, int(value)))
    return (level, level, level)


def _wrapped_step(size=64, step=120):
    return [
        [_gray(90 + (step / 2 if c < size // 2 else -step / 2)) for c in range(size)]
        for _ in range(size)
    ]


def _tiled(size=64, tile=16, seed=3):
    import random

    rng = random.Random(seed)
    block = [[_gray(rng.randrange(256)) for _ in range(tile)] for _ in range(tile)]
    return [[block[r % tile][c % tile] for c in range(size)] for r in range(size)]


def test_confirmed_seam_is_feathered_and_reported() -> None:
    image = _wrapped_step()
    original = deepcopy(image)
    result = fix_texture_seams({"family": "texture", "image": image, "band_px": 4})
    assert result.abstained is False
    assert result.output["status"] == "suggest"
    assert result.output["source_preserved"] is True
    assert image == original
    report = result.output["corrections"][0]
    assert report["axis"] == "x"
    assert report["max_channel_delta"] > 0
    assert result.output["axes"]["x"]["improved"] is True
    assert result.output["axes"]["x"]["gap_after"] < result.output["axes"]["x"]["gap"]


def test_band_is_bounded_by_the_image() -> None:
    result = fix_texture_seams({"family": "texture", "image": _wrapped_step(), "band_px": 999})
    if not result.abstained:
        assert result.output["corrections"][0]["band_px"] <= 16


def test_continuous_texture_is_not_touched() -> None:
    image = _tiled()
    original = deepcopy(image)
    result = fix_texture_seams({"family": "tilemap", "image": image})
    assert result.abstained is True
    assert result.output["reason"] == "no_confirmed_seam"
    assert result.output["image"] is None
    assert image == original


def test_nano_disagreement_blocks_the_repair(monkeypatch) -> None:
    def fake_classify(features, *, model=None):
        return SpecialistResult(
            "bellium/nano-nn/tile-seam:v0",
            {"status": "suggest", "verdict": "continuous"},
            0.9, False, AuthorityMode.CONSULTATIVE,
        )

    monkeypatch.setattr(texture_tile, "classify_seam", fake_classify)
    result = fix_texture_seams({"family": "texture", "image": _wrapped_step()})
    assert result.abstained is True
    assert result.output["reason"] == "no_confirmed_seam"
    assert result.output["axes"]["x"]["reason"] == "nano_disagrees"


def test_unknown_family_abstains() -> None:
    result = fix_texture_seams({"family": "portrait", "image": _wrapped_step()})
    assert result.abstained is True
    assert result.output["reason"] == "unknown_family"


def test_small_images_abstain() -> None:
    tiny = [[(10, 10, 10) for _ in range(4)] for _ in range(4)]
    result = fix_texture_seams({"family": "texture", "image": tiny})
    assert result.abstained is True
    assert result.output["reason"] == "image_too_small"
