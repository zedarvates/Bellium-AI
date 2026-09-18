"""Isolated lossless image research codec; not an automatic BLCP/1 candidate.

One fixed hypothesis from compression-independent-v1.md. This module does not
change the frozen byte predictors, normal image API or specialist registry.
"""

from .codecs import CHANNELS, ImagePixels, _image_size
from .container import MAX_PREDICTIVE_BYTES, _bytes, inspect_packet, pack, unpack

PREDICTORS = ("left", "up", "average", "gradient", "micro-nn")
KIND = "spatial-image-research/1"


def _transform(data, width, channels, predictor, *, decode=False):
    result = bytearray(len(data))
    plane_size = len(data) // channels
    for channel in range(channels):
        source = data[channel * plane_size:(channel + 1) * plane_size]
        known = bytearray(plane_size)
        encoded = bytearray(plane_size)
        weights = [2048, 2048, 0, 0]
        for i, value in enumerate(source):
            x = i % width
            left = known[i - 1] if x else 0
            up = known[i - width] if i >= width else 0
            corner = known[i - width - 1] if x and i >= width else 0
            if predictor == "left":
                predicted = left
            elif predictor == "up":
                predicted = up
            elif predictor == "average":
                predicted = (left + up) // 2
            elif predictor == "gradient":
                predicted = max(0, min(255, left + up - corner))
            else:
                features = (left - 128, up - 128, corner - 128, 128)
                predicted = max(0, min(255, 128 + sum(w * v for w, v in
                                                     zip(weights, features)) // 4096))
            actual = (value + predicted) & 255 if decode else value
            encoded[i] = actual if decode else (actual - predicted) & 255
            known[i] = actual
            if predictor == "micro-nn":
                error = actual - predicted
                denominator = 8 * (1 + sum(v * v for v in features))
                weights = [max(-32768, min(32768, w + error * v * 4096 // denominator))
                           for w, v in zip(weights, features)]
        result[channel * plane_size:(channel + 1) * plane_size] = encoded
    return bytes(result)


def encode_spatial_image(pixels: bytes, *, width: int, height: int, mode="RGB",
                         predictor="micro-nn") -> bytes:
    size = _image_size(width, height, mode)
    if size > MAX_PREDICTIVE_BYTES or len(_bytes(pixels, "pixels")) != size:
        raise ValueError("spatial experiment requires matching pixels, at most 65536 bytes")
    if predictor not in PREDICTORS:
        raise ValueError("unknown spatial predictor")
    channels = CHANNELS[mode]
    planes = b"".join(pixels[c::channels] for c in range(channels))
    residual = _transform(planes, width, channels, predictor)
    metadata = {"kind": KIND, "width": width, "height": height, "mode": mode,
                "predictor": predictor}
    return pack(residual, metadata, method="zlib")


def decode_spatial_image(packet: bytes) -> ImagePixels:
    report = inspect_packet(packet, max_output_bytes=MAX_PREDICTIVE_BYTES)
    metadata = report["metadata"]
    if (set(metadata) != {"kind", "width", "height", "mode", "predictor"}
            or metadata["kind"] != KIND or metadata["predictor"] not in PREDICTORS):
        raise ValueError("invalid spatial image metadata")
    width, height, mode = (metadata[k] for k in ("width", "height", "mode"))
    size = _image_size(width, height, mode)
    if size != report["raw_bytes"] or size > MAX_PREDICTIVE_BYTES:
        raise ValueError("spatial image dimensions exceed or differ from payload")
    residual, _ = unpack(packet, kind=KIND, max_output_bytes=MAX_PREDICTIVE_BYTES)
    channels = CHANNELS[mode]
    planes = _transform(residual, width, channels, metadata["predictor"], decode=True)
    pixels = bytearray(size)
    for channel in range(channels):
        pixels[channel::channels] = planes[channel * width * height:(channel + 1) * width * height]
    return ImagePixels(width, height, mode, bytes(pixels))
