import hashlib
import json
import random
import struct
import zlib

import pytest

from bellium.compression import (
    decode_image, decode_splats, decode_stream, encode_image, encode_splats,
    encode_stream, inspect_packet,
)
from bellium.compression.codecs import SPLAT
from bellium.compression.container import HEADER, MAX_PREDICTIVE_BYTES, METHODS, pack
from bellium.compression.predictors import KNNPredictor, MicroNNPredictor, residuals
from bellium.specialists.compression import compress


def forge_packet(*, size, payload, code=1, meta=None, digest=None):
    metadata = json.dumps(meta or {"kind": "stream"}).encode()
    header = HEADER.pack(b"BLCP", 1, code, size, len(metadata), len(payload),
                         digest or bytes(32))
    body = header + metadata + payload
    return body + hashlib.sha256(body).digest()


def splats(count=8, tick=0):
    return b"".join(SPLAT.pack(i + tick / 8, -0.0, i / 3, 0, 0, 0, 1,
                               0.1, 0.2, 0.3, 0.5, 1, 0.25, 0)
                    for i in range(count))


@pytest.mark.parametrize("method", (*METHODS, "auto"))
@pytest.mark.parametrize("length", (0, 1, 63, 256, 4097))
def test_exact_stream_roundtrip_with_fresh_decoder(method, length):
    rng = random.Random(length)
    raw = bytes(rng.randrange(256) for _ in range(length))
    packet = encode_stream(raw, method=method)
    assert decode_stream(packet) == raw
    assert encode_stream(raw, method=method) == packet


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("mode,channels", (("L", 1), ("RGB", 3), ("RGBA", 4)))
def test_image_pixels_and_invisible_rgb_survive(method, mode, channels):
    raw = bytes((i * 19) % 256 for i in range(8 * 6 * channels))
    if channels == 4:
        raw = bytes(0 if i % 4 == 3 else value for i, value in enumerate(raw))
    decoded = decode_image(encode_image(raw, width=8, height=6, mode=mode, method=method))
    assert (decoded.width, decoded.height, decoded.mode) == (8, 6, mode)
    assert decoded.pixels == raw


@pytest.mark.parametrize("method", METHODS)
def test_splat_float32_bits_and_temporal_frame(method):
    base, current = splats(), splats(tick=3)
    static = encode_splats(base, method=method)
    assert decode_splats(static) == base  # Includes negative zero, exactly.
    delta = encode_splats(current, base=base, method=method)
    assert decode_splats(delta, base=base) == current
    with pytest.raises(ValueError, match="requires its base"):
        decode_splats(delta)
    with pytest.raises(ValueError, match="base fingerprint"):
        decode_splats(delta, base=splats(tick=1))
    with pytest.raises(ValueError, match="must not have a base"):
        decode_splats(static, base=base)


def test_animation_chain_stays_exact_and_rejects_reordered_base():
    base = splats()
    for tick in range(1, 33):
        current = splats(tick=tick)
        packet = encode_splats(current, base=base, method="micro-nn")
        assert decode_splats(packet, base=base) == current
        base = current
    with pytest.raises(ValueError, match="base fingerprint"):
        decode_splats(packet, base=base[SPLAT.size:] + base[:SPLAT.size])


def test_auto_selects_smallest_full_packet_and_reports_expansion():
    raw = bytes(range(256)) * 2
    candidates = [encode_stream(raw, method=method) for method in METHODS]
    chosen = encode_stream(raw)
    assert len(chosen) == min(map(len, candidates))
    assert decode_stream(chosen) == raw
    tiny = inspect_packet(encode_stream(b"x"))
    assert tiny["saved_bytes"] < 0
    assert tiny["packet_bytes"] > tiny["raw_bytes"]


@pytest.mark.parametrize("method", ("knn", "micro-nn"))
def test_neural_work_bound_does_not_prevent_classical_fallback(method):
    raw = b"a" * (MAX_PREDICTIVE_BYTES + 1)
    with pytest.raises(ValueError, match="limited"):
        encode_stream(raw, method=method)
    automatic = encode_stream(raw)
    assert inspect_packet(automatic)["method"] in ("raw", "zlib", "delta")
    assert decode_stream(automatic) == raw


def test_predictors_adapt_only_after_observations_and_stay_bounded():
    neural = MicroNNPredictor()
    before = neural.weights[:]
    for _ in range(100):
        neural.observe(32)
    assert neural.weights != before
    assert abs(neural.predict() - 32) < 8
    neighbor = KNNPredictor()
    for _ in range(100):
        neighbor.observe(32)
    assert neighbor.predict() == 32
    assert len(neighbor.memory) == 64


@pytest.mark.parametrize("method,expected", (
    ("knn", "000a0f1312f300021d1bd1f0"),
    ("micro-nn", "800a0e0c0bec0c0b0a09c3ff"),
))
def test_blcp_v1_predictor_golden_vectors(method, expected):
    # Persisted v1 vectors: simultaneous encoder/decoder changes must not
    # silently pass roundtrip tests while breaking existing packets.
    raw = bytes([0, 10, 20, 30, 40, 20, 30, 40, 50, 60, 0, 255])
    assert residuals(raw, method) == bytes.fromhex(expected)
    assert residuals(bytes.fromhex(expected), method, decode=True) == raw


@pytest.mark.parametrize("method,expected", (
    ("knn", "11849122418f457e626ee3504d6580e86f1395b40ea41452dd63c03e0b37face"),
    ("micro-nn", "b09346bdc3a939da22e0ff321d341603bc989a5e0b78bbf406621b2f7cd37d9e"),
))
def test_blcp_v1_state_reset_golden_digest(method, expected):
    raw = bytes(range(256)) * 17  # Crosses the specified 4096-byte reset.
    encoded = residuals(raw, method)
    assert hashlib.sha256(encoded).hexdigest() == expected
    assert residuals(encoded, method, decode=True) == raw


@pytest.mark.parametrize("mutate", (
    lambda p: p[:-1], lambda p: p + b"extra",
    lambda p: p[:60] + bytes([p[60] ^ 1]) + p[61:],
    lambda p: b"BAD!" + p[4:],
))
def test_corrupt_envelopes_rejected(mutate):
    with pytest.raises(ValueError):
        decode_stream(mutate(encode_stream(b"a" * 200)))


def test_output_cap_is_checked_before_decompression():
    with pytest.raises(ValueError, match="limit"):
        decode_stream(encode_stream(b"a" * 1024), max_output_bytes=64)


@pytest.mark.parametrize("payload", (
    zlib.compress(b"a" * 100000), zlib.compress(b"aaaa") + b"junk",
    zlib.compress(b"aaaa")[:-1], b"not zlib",
))
def test_valid_envelope_cannot_hide_bomb_trailing_bytes_or_broken_zlib(payload):
    packet = forge_packet(size=4, payload=payload, digest=hashlib.sha256(b"aaaa").digest())
    with pytest.raises(ValueError):
        decode_stream(packet)


def test_invalid_decoded_digest_and_kind_rejected():
    with pytest.raises(ValueError, match="integrity"):
        decode_stream(forge_packet(size=4, payload=b"aaaa", code=0))
    with pytest.raises(ValueError, match="kind"):
        decode_stream(encode_image(b"abc", width=1, height=1))


@pytest.mark.parametrize("width,height,mode,pixels", (
    (True, 1, "L", b"a"), (0, 1, "L", b""),
    (2**32, 1, "L", b""), (1, 1, "P", b"a"), (1, 1, "RGBA", b"abc"),
))
def test_invalid_images_refused(width, height, mode, pixels):
    with pytest.raises(ValueError):
        encode_image(pixels, width=width, height=height, mode=mode)


def test_image_metadata_cannot_silently_change_pixel_count():
    packet = pack(b"abc", {"kind": "image", "width": 2, "height": 1, "mode": "RGB"})
    with pytest.raises(ValueError, match="dimensions"):
        decode_image(packet)


@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf")))
def test_nonfinite_splat_fields_rejected_on_both_sides(value):
    raw = struct.pack("<f", value) + splats(1)[4:]
    with pytest.raises(ValueError, match="finite"):
        encode_splats(raw)
    # Correct framing alone must not admit non-finite splat fields.
    meta = inspect_packet(encode_splats(splats(1)))["metadata"]
    meta["records_sha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError, match="finite"):
        decode_splats(pack(raw, meta, method="raw"))


def test_empty_splats_and_mismatched_base_counts():
    assert decode_splats(encode_splats(b"")) == b""
    with pytest.raises(ValueError, match="56-byte"):
        encode_splats(b"short")
    with pytest.raises(ValueError, match="same record count"):
        encode_splats(splats(2), base=splats(1))


def test_specialist_is_consultative_and_never_invents_confidence():
    from bellium.contracts import AuthorityMode
    result = compress(b"hello" * 100, kind="stream")
    assert result.authority_mode is AuthorityMode.CONSULTATIVE
    assert result.confidence is None
    assert decode_stream(result.output["packet"]) == b"hello" * 100


def test_compression_specialists_are_discoverable_and_consultative():
    from bellium.contracts import AuthorityMode
    from bellium.registry.catalog import get_specialist
    identifiers = [f"bellium/hybrid/{kind}-compression:v0"
                   for kind in ("image", "stream", "splats")]
    identifiers += [f"bellium/{family}/byte-compression:v0" for family in ("knn", "micro-nn")]
    for identifier in identifiers:
        spec = get_specialist(identifier)
        assert not spec.decision_eligible
        assert spec.authority_mode is AuthorityMode.CONSULTATIVE
