"""Archive complete, fixed-record binary PLY files without interpreting values.

The original header and every property byte survive. Byte-plane transposition
is reversible; there is no quantization, SH truncation or quaternion conversion.
This bounded subset accepts one vertex element with scalar properties only.
"""

from dataclasses import dataclass
import hashlib

from .codecs import _interleave, _planes
from .container import MAX_RAW_BYTES, _bytes, inspect_packet, pack, unpack

KIND = "ply-archive/1"
MAX_HEADER_BYTES = 16384
MAX_PROPERTIES = 256
SCALAR_SIZES = {
    b"char": 1, b"uchar": 1, b"short": 2, b"ushort": 2,
    b"int": 4, b"uint": 4, b"float": 4, b"double": 8,
    b"int8": 1, b"uint8": 1, b"int16": 2, b"uint16": 2,
    b"int32": 4, b"uint32": 4, b"float32": 4, b"float64": 8,
}


@dataclass(frozen=True)
class PlyLayout:
    encoding: str
    header_bytes: int
    vertex_count: int
    record_bytes: int
    properties: tuple[tuple[str, str], ...]


def inspect_ply(data: bytes) -> PlyLayout:
    """Validate structure and byte counts, not renderability or numeric values."""
    _bytes(data, "PLY data")
    if len(data) > MAX_RAW_BYTES:
        raise ValueError("PLY exceeds the 16 MiB byte limit")
    offset = 0
    lines = []
    while offset < min(len(data), MAX_HEADER_BYTES):
        end = data.find(b"\n", offset, MAX_HEADER_BYTES)
        if end < 0:
            break
        line = data[offset:end].removesuffix(b"\r")
        offset = end + 1
        if line == b"end_header":
            break
        lines.append(line)
    else:
        raise ValueError("missing bounded PLY header")
    if not lines or line != b"end_header":
        raise ValueError("missing newline-terminated bounded PLY header")
    if len(lines) < 3 or lines[0] != b"ply":
        raise ValueError("invalid PLY signature or header")
    format_words = lines[1].split()
    if (len(format_words) != 3 or format_words[0] != b"format"
            or format_words[1] not in (b"binary_little_endian", b"binary_big_endian")
            or format_words[2] != b"1.0"):
        raise ValueError("only binary PLY 1.0 is supported")
    count = None
    properties = []
    names = set()
    stride = 0
    for line in lines[2:]:
        words = line.split()
        if words and words[0] in (b"comment", b"obj_info"):
            continue
        if (len(words) == 3 and words[:2] == [b"element", b"vertex"]
                and count is None and not properties):
            token = words[2]
            if not token.isdigit() or len(token) > 8:
                raise ValueError("invalid vertex count")
            count = int(token)
            if count > MAX_RAW_BYTES:
                raise ValueError("vertex count exceeds the work limit")
        elif (len(words) == 3 and words[0] == b"property"
              and words[1] in SCALAR_SIZES and count is not None):
            name = words[2]
            if not name.isascii() or name in names or len(properties) >= MAX_PROPERTIES:
                raise ValueError("invalid, duplicate or excessive PLY properties")
            names.add(name)
            properties.append((words[1].decode("ascii"), name.decode("ascii")))
            stride += SCALAR_SIZES[words[1]]
        else:
            raise ValueError("requires a single vertex element with scalar properties")
    if count is None or not properties:
        raise ValueError("missing vertex element or scalar properties")
    if len(data) != offset + count * stride:
        raise ValueError("PLY body size does not match the declared vertex layout")
    return PlyLayout(format_words[1].decode("ascii"), offset, count, stride, tuple(properties))


def encode_ply(data: bytes, *, layout="auto", method="auto") -> bytes:
    """Compare complete archive sizes; ties prefer the original record layout."""
    info = inspect_ply(data)
    if layout not in ("auto", "original", "byte-planes"):
        raise ValueError("unknown PLY archive layout")
    source_digest = hashlib.sha256(data).hexdigest()
    choices = ("original", "byte-planes") if layout == "auto" else (layout,)
    best = None
    for choice in choices:
        transformed = data
        if choice == "byte-planes":
            transformed = data[:info.header_bytes] + _planes(data[info.header_bytes:], info.record_bytes)
        packet = pack(transformed, {"kind": KIND, "layout": choice,
                                    "source_sha256": source_digest}, method=method)
        if best is None or len(packet) < len(best):
            best = packet
    return best


def decode_ply(packet: bytes, *, max_output_bytes=MAX_RAW_BYTES) -> bytes:
    report = inspect_packet(packet, max_output_bytes=max_output_bytes)
    meta = report["metadata"]
    if (set(meta) != {"kind", "layout", "source_sha256"} or meta["kind"] != KIND
            or meta["layout"] not in ("original", "byte-planes")):
        raise ValueError("invalid PLY archive metadata")
    fingerprint = meta["source_sha256"]
    if (not isinstance(fingerprint, str) or len(fingerprint) != 64
            or any(c not in "0123456789abcdef" for c in fingerprint)):
        raise ValueError("invalid source fingerprint")
    transformed, _ = unpack(packet, kind=KIND, max_output_bytes=max_output_bytes)
    info = inspect_ply(transformed)
    data = transformed
    if meta["layout"] == "byte-planes":
        data = transformed[:info.header_bytes] + _interleave(
            transformed[info.header_bytes:], info.record_bytes)
    if hashlib.sha256(data).hexdigest() != fingerprint:
        raise ValueError("reconstructed PLY fingerprint mismatch")
    return data
