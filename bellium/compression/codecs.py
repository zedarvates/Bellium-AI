"""Lossless byte streams, image pixels, and finite float32 Gaussian splats.

Images use channel planes. Splats use byte planes across 56-byte records:
position xyz, quaternion xyzw, scale xyz, opacity, RGB; all little-endian float32.
Temporal splat packets XOR an explicit, identical-sized base before compression.
"""

from dataclasses import dataclass
import hashlib
import math
import struct

from .container import MAX_RAW_BYTES, _bytes, _integer, inspect_packet, pack, unpack

CHANNELS = {"L": 1, "RGB": 3, "RGBA": 4}
SPLAT = struct.Struct("<14f")
SPLAT_LAYOUT = "position3-rotation_xyzw4-scale3-opacity1-rgb3/le-f32"


def _planes(data, width):
    return b"".join(data[offset::width] for offset in range(width))


def _interleave(data, width):
    output = bytearray(len(data))
    count = len(data) // width
    for offset in range(width):
        output[offset::width] = data[offset * count:(offset + 1) * count]
    return bytes(output)


def encode_stream(data: bytes, *, method="auto") -> bytes:
    """Encode an independent chunk, with all state reset at the packet boundary."""
    return pack(data, {"kind": "stream"}, method=method)


def decode_stream(packet: bytes, *, max_output_bytes=MAX_RAW_BYTES) -> bytes:
    data, meta = unpack(packet, kind="stream", max_output_bytes=max_output_bytes)
    if meta != {"kind": "stream"}:
        raise ValueError("unexpected stream metadata")
    return data


@dataclass(frozen=True)
class ImagePixels:
    width: int
    height: int
    mode: str
    pixels: bytes


def _image_size(width, height, mode):
    _integer(width, "width", 1, MAX_RAW_BYTES)
    _integer(height, "height", 1, MAX_RAW_BYTES)
    if not isinstance(mode, str) or mode not in CHANNELS:
        raise ValueError("image mode must be L, RGB or RGBA")
    size = width * height * CHANNELS[mode]
    if size > MAX_RAW_BYTES:
        raise ValueError("image exceeds the raw byte limit")
    return size


def encode_image(pixels: bytes, *, width: int, height: int, mode="RGB", method="auto"):
    size = _image_size(width, height, mode)
    if len(_bytes(pixels, "pixels")) != size:
        raise ValueError("pixel byte count does not match image dimensions")
    meta = {"kind": "image", "width": width, "height": height, "mode": mode}
    return pack(_planes(pixels, CHANNELS[mode]), meta, method=method)


def decode_image(packet: bytes, *, max_output_bytes=MAX_RAW_BYTES) -> ImagePixels:
    # Validate dimensions before allocating decoded pixels.
    meta = inspect_packet(packet, max_output_bytes=max_output_bytes)["metadata"]
    if set(meta) != {"kind", "width", "height", "mode"} or meta["kind"] != "image":
        raise ValueError("invalid image metadata")
    size = _image_size(meta["width"], meta["height"], meta["mode"])
    data, _ = unpack(packet, kind="image", max_output_bytes=max_output_bytes)
    if size != len(data):
        raise ValueError("decoded pixels do not match dimensions")
    return ImagePixels(meta["width"], meta["height"], meta["mode"],
                       _interleave(data, CHANNELS[meta["mode"]]))


def _validate_splats(data):
    _bytes(data, "splat records")
    if len(data) > MAX_RAW_BYTES or len(data) % SPLAT.size:
        raise ValueError("splats require bounded 56-byte float32 records")
    if any(not math.isfinite(value) for record in SPLAT.iter_unpack(data) for value in record):
        raise ValueError("splat fields must be finite")


def _xor(data, base):
    return bytes(a ^ b for a, b in zip(data, base))


def encode_splats(records: bytes, *, base: bytes | None = None, method="auto") -> bytes:
    _validate_splats(records)
    meta = {"kind": "splats", "layout": SPLAT_LAYOUT, "count": len(records) // SPLAT.size,
            "base_sha256": None, "records_sha256": hashlib.sha256(records).hexdigest()}
    data = records
    if base is not None:
        _validate_splats(base)
        if len(base) != len(records):
            raise ValueError("base must have the same record count and order")
        meta["base_sha256"] = hashlib.sha256(base).hexdigest()
        data = _xor(records, base)
    return pack(_planes(data, SPLAT.size), meta, method=method)


def decode_splats(packet: bytes, *, base: bytes | None = None,
                  max_output_bytes=MAX_RAW_BYTES) -> bytes:
    meta = inspect_packet(packet, max_output_bytes=max_output_bytes)["metadata"]
    if (set(meta) != {"kind", "layout", "count", "base_sha256", "records_sha256"}
            or meta["kind"] != "splats" or meta["layout"] != SPLAT_LAYOUT):
        raise ValueError("unsupported splat metadata or layout")
    count = _integer(meta["count"], "splat count", 0, MAX_RAW_BYTES // SPLAT.size)
    expected = count * SPLAT.size
    if meta["base_sha256"] is None:
        if base is not None:
            raise ValueError("static packet must not have a base")
    else:
        if base is None:
            raise ValueError("temporal splat packet requires its base")
        _validate_splats(base)
        if len(base) != expected or hashlib.sha256(base).hexdigest() != meta["base_sha256"]:
            raise ValueError("base fingerprint or count mismatch")
    data, _ = unpack(packet, kind="splats", max_output_bytes=max_output_bytes)
    if len(data) != expected:
        raise ValueError("decoded splat count mismatch")
    records = _interleave(data, SPLAT.size)
    if base is not None:
        records = _xor(records, base)
    _validate_splats(records)
    if hashlib.sha256(records).hexdigest() != meta["records_sha256"]:
        raise ValueError("splat reconstruction digest mismatch")
    return records
