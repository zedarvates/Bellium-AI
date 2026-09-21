"""A minimal deterministic PNG writer for atlas pages, and its own inspector.

Only what a reader needs: 8-bit RGB or RGBA, no interlacing, no palette, one
filter type (None) and a fixed zlib level, so the same page always encodes to the
same bytes. inspect_png re-parses the container instead of trusting the writer,
and decode_png decodes the same subset for a round-trip check. A tuned encoder
with per-scanline filter selection would produce smaller files.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

from bellium.atlas.raster import AtlasRaster, RasterPage, page_digest

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
COLOR_TYPES = {3: 2, 4: 6}
CHANNELS_BY_COLOR = {2: 3, 6: 4}
FILTER_NONE = 0
COMPRESS_LEVEL = 9
MAX_RAW_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class PngInfo:
    width: int
    height: int
    color_type: int
    channels: int
    bit_depth: int
    idat_bytes: int
    raw_bytes: int
    scanlines: bytes
    chunks: tuple[str, ...]
    violations: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.violations


def _chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return len(payload).to_bytes(4, "big") + body + zlib.crc32(body).to_bytes(4, "big")


def _scanlines(page: RasterPage) -> bytes:
    parts = []
    for row in page.pixels:
        parts.append(bytes([FILTER_NONE]))
        for pixel in row:
            parts.append(bytes(pixel))
    return b"".join(parts)


def encode_png(page: RasterPage) -> bytes:
    if not isinstance(page, RasterPage):
        raise ValueError("encode_png expects a RasterPage")
    if page.channels not in COLOR_TYPES:
        raise ValueError(f"channels must be one of {sorted(COLOR_TYPES)}")
    header = (
        page.width.to_bytes(4, "big")
        + page.height.to_bytes(4, "big")
        + bytes([8, COLOR_TYPES[page.channels], 0, 0, 0])
    )
    return (
        PNG_SIGNATURE
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(_scanlines(page), COMPRESS_LEVEL))
        + _chunk(b"IEND", b"")
    )


def encode_pages(raster: AtlasRaster) -> tuple[bytes, ...]:
    if not isinstance(raster, AtlasRaster):
        raise ValueError("encode_pages expects an AtlasRaster")
    return tuple(encode_png(page) for page in raster.pages)


def inspect_png(data: object) -> PngInfo:
    """Re-parse a PNG container. Violations are reported, never repaired."""
    violations: list[str] = []
    width = height = bit_depth = color_type = 0
    idat = bytearray()
    chunks: list[str] = []
    scanlines = b""
    seen_idat = False
    if not isinstance(data, (bytes, bytearray)):
        return PngInfo(0, 0, 0, 0, 0, 0, 0, b"", (), ("png data must be bytes",))
    raw = bytes(data)
    if not raw.startswith(PNG_SIGNATURE):
        return PngInfo(0, 0, 0, 0, 0, 0, 0, b"", (), ("png signature is missing",))
    offset = len(PNG_SIGNATURE)
    seen_iend = False
    previous = ""
    while offset < len(raw):
        if offset + 8 > len(raw):
            violations.append("truncated chunk header")
            break
        length = int.from_bytes(raw[offset:offset + 4], "big")
        kind = raw[offset + 4:offset + 8]
        body_start = offset + 8
        body_end = body_start + length
        if body_end + 4 > len(raw):
            violations.append(f"chunk {kind!r} runs past the end of the data")
            break
        payload = raw[body_start:body_end]
        stored = int.from_bytes(raw[body_end:body_end + 4], "big")
        computed = zlib.crc32(kind + payload)
        if stored != computed:
            violations.append(f"chunk {kind.decode('ascii', 'replace')} has a bad CRC")
        name = kind.decode("ascii", "replace")
        chunks.append(name)
        if kind == b"IHDR":
            if length != 13:
                violations.append("IHDR must be 13 bytes")
            else:
                width = int.from_bytes(payload[0:4], "big")
                height = int.from_bytes(payload[4:8], "big")
                bit_depth = payload[8]
                color_type = payload[9]
                if width <= 0 or height <= 0:
                    violations.append("IHDR declares an empty page")
                if bit_depth != 8:
                    violations.append(f"IHDR bit depth is {bit_depth}, expected 8")
                if color_type not in CHANNELS_BY_COLOR:
                    violations.append(f"IHDR colour type is {color_type}, expected 2 or 6")
                if payload[10] != 0 or payload[11] != 0 or payload[12] != 0:
                    violations.append("IHDR compression, filter or interlace is not 0")
        elif kind == b"IDAT":
            if seen_idat and previous != "IDAT":
                violations.append("IDAT chunks are not consecutive")
            seen_idat = True
            idat += payload
        elif kind == b"IEND":
            if length != 0:
                violations.append("IEND must be empty")
            seen_iend = True
            offset = body_end + 4
            if offset != len(raw):
                violations.append("data continues after IEND")
            break
        elif kind[:1].isupper():
            violations.append(f"unknown critical chunk {name}")
        previous = name
        offset = body_end + 4
    if not seen_iend:
        violations.append("IEND is missing")
    channels = CHANNELS_BY_COLOR.get(color_type, 0)
    raw_bytes = 0
    if not violations and width and height and channels:
        expected = height * (1 + width * channels)
        inflater = zlib.decompressobj()
        try:
            scanlines = inflater.decompress(bytes(idat), MAX_RAW_BYTES + 1)
        except zlib.error as error:
            violations.append(f"IDAT does not inflate: {error}")
            scanlines = b""
        if inflater.unconsumed_tail:
            violations.append("page is larger than the declared 64 MiB inspection limit")
            scanlines = b""
        elif scanlines and not inflater.eof:
            violations.append("IDAT stream ends before the image is complete")
            scanlines = b""
        if scanlines and len(scanlines) != expected:
            violations.append(f"scanlines are {len(scanlines)} bytes, expected {expected}")
        if scanlines and len(scanlines) == expected:
            stride = 1 + width * channels
            for row in range(height):
                if scanlines[row * stride] != FILTER_NONE:
                    violations.append(f"scanline {row} uses filter {scanlines[row * stride]}")
            raw_bytes = len(scanlines)
    elif width and height and channels:
        raw_bytes = height * (1 + width * channels)
    if raw_bytes > MAX_RAW_BYTES:
        violations.append("page is larger than the declared 64 MiB inspection limit")
    return PngInfo(width, height, color_type, channels, bit_depth, len(idat), raw_bytes,
                   scanlines, tuple(chunks), tuple(violations))


def decode_png(data: object) -> RasterPage:
    """Decode the subset this module writes. Anything else raises."""
    info = inspect_png(data)
    if not info.ok:
        raise ValueError("; ".join(info.violations))
    stride = 1 + info.width * info.channels
    rows = []
    for row in range(info.height):
        start = row * stride + 1
        line = info.scanlines[start:start + info.width * info.channels]
        rows.append(tuple(
            tuple(line[column * info.channels:(column + 1) * info.channels])
            for column in range(info.width)
        ))
    pixels = tuple(rows)
    return RasterPage(
        index=-1,
        width=info.width,
        height=info.height,
        channels=info.channels,
        pixels=pixels,
        digest=page_digest(pixels),
        raw_bytes=info.height * info.width * info.channels,
    )
