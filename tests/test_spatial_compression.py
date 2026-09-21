import random

import pytest

from bellium.compression import decode_image
from bellium.compression.container import pack
from bellium.compression.spatial_experiment import (
    KIND, PREDICTORS, _transform, decode_spatial_image, encode_spatial_image,
)


@pytest.mark.parametrize("predictor", PREDICTORS)
@pytest.mark.parametrize("mode,channels", (("L", 1), ("RGB", 3), ("RGBA", 4)))
@pytest.mark.parametrize("width,height", ((1, 1), (1, 8), (8, 1), (13, 11)))
def test_spatial_roundtrip(predictor, mode, channels, width, height):
    rng = random.Random(45)
    pixels = bytes(rng.randrange(256) for _ in range(width * height * channels))
    decoded = decode_spatial_image(encode_spatial_image(
        pixels, width=width, height=height, mode=mode, predictor=predictor))
    assert decoded.pixels == pixels
    assert (decoded.width, decoded.height, decoded.mode) == (width, height, mode)


def test_spatial_neuron_is_causal_within_each_plane():
    before = bytes(range(64))
    altered_future = before[:32] + bytes([255] * 32)
    assert _transform(before, 8, 1, "micro-nn")[:32] == _transform(
        altered_future, 8, 1, "micro-nn")[:32]


def test_spatial_preserves_invisible_rgb_and_is_not_promoted():
    pixels = bytes([150, 30, 240, 0]) * 64
    packet = encode_spatial_image(pixels, width=8, height=8, mode="RGBA")
    assert decode_spatial_image(packet).pixels == pixels
    with pytest.raises(ValueError, match="metadata"):
        decode_image(packet)


def test_spatial_rejects_inconsistent_metadata_and_unknown_predictor():
    with pytest.raises(ValueError, match="predictor"):
        encode_spatial_image(b"a", width=1, height=1, mode="L", predictor="unknown")
    metadata = {"kind": KIND, "width": 2, "height": 2, "mode": "L", "predictor": "micro-nn"}
    with pytest.raises(ValueError, match="dimensions"):
        decode_spatial_image(pack(b"a", metadata))


def test_spatial_refuses_work_above_the_packet_budget():
    with pytest.raises(ValueError, match="65536"):
        encode_spatial_image(bytes(65537), width=65537, height=1, mode="L")
