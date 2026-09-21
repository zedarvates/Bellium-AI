"""Archive complete PLY files without interpreting their values.

The header survives byte for byte, including comments, element order, property
order and unknown vendor columns. Binary bodies are optionally transposed into
byte planes, one element at a time; ASCII bodies are stored verbatim, or
transposed by column when they are canonical. Both transpositions are exactly
reversible, and neither quantizes, drops a spherical-harmonics coefficient nor
converts a quaternion.

List counts are read so every element range is known and bounds-checked, but the
item values themselves stay opaque. Bounded subset: at most 64 elements, at most
256 properties per element, a header of at most 16 KiB and a caller-declared
file limit. An ASCII body whose records do not match their declared fields is
refused rather than guessed at.
"""

from dataclasses import dataclass
import hashlib

from .codecs import _interleave, _planes
from .container import (
    ABSOLUTE_MAX_BYTES, MAX_RAW_BYTES, _bytes, _integer, inspect_packet, pack, unpack,
)

BINARY_KIND = "ply-archive/1"
ASCII_KIND = "ply-ascii/1"
# Kept for callers that imported the original name.
KIND = BINARY_KIND
BINARY_LAYOUTS = ("original", "byte-planes")
ASCII_LAYOUTS = ("opaque", "columns")
MAX_HEADER_BYTES = 16384
MAX_PROPERTIES = 256
MAX_ELEMENTS = 64
SCALAR_SIZES = {
    "char": 1, "uchar": 1, "short": 2, "ushort": 2,
    "int": 4, "uint": 4, "float": 4, "double": 8,
    "int8": 1, "uint8": 1, "int16": 2, "uint16": 2,
    "int32": 4, "uint32": 4, "float32": 4, "float64": 8,
}
INTEGER_TYPES = frozenset(name for name in SCALAR_SIZES if not name.startswith("float")
                          and name != "double")


@dataclass(frozen=True)
class PlyProperty:
    type: str
    name: str
    count_type: str | None = None
    item_type: str | None = None

    @property
    def is_list(self) -> bool:
        return self.count_type is not None

    @property
    def label(self) -> str:
        return f"list[{self.count_type}/{self.item_type}]" if self.is_list else self.type

    def as_pair(self) -> tuple[str, str]:
        return (self.label, self.name)


@dataclass(frozen=True)
class PlyElement:
    name: str
    count: int
    properties: tuple[PlyProperty, ...]


@dataclass(frozen=True)
class PlyElementSpan:
    name: str
    count: int
    properties: tuple[PlyProperty, ...]
    stride: int
    start: int
    end: int

    @property
    def planar(self) -> bool:
        """A stride of one byte cannot be improved by transposition."""
        return self.stride >= 2


@dataclass(frozen=True)
class PlyLayout:
    encoding: str
    header_bytes: int
    elements: tuple[PlyElementSpan, ...]
    unparsed_ascii: bool = False

    @property
    def ascii(self) -> bool:
        return self.encoding == "ascii"

    @property
    def vertex(self) -> PlyElementSpan | None:
        for element in self.elements:
            if element.name == "vertex":
                return element
        return None

    @property
    def vertex_count(self) -> int:
        element = self.vertex
        return element.count if element is not None else 0

    @property
    def properties(self) -> tuple[tuple[str, str], ...]:
        element = self.vertex
        return tuple(prop.as_pair() for prop in element.properties) if element else ()

    @property
    def record_bytes(self) -> int:
        element = self.vertex
        return element.stride if element is not None else 0


def _parse_header(data: bytes, *, limit: int):
    """Validate the bounded header. Body bytes are checked by the caller."""
    _bytes(data, "PLY data")
    _integer(limit, "limit", 1, ABSOLUTE_MAX_BYTES)
    if len(data) > limit:
        raise ValueError("PLY exceeds the declared byte limit")
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
    declared = lines[1].split()
    supported = (b"binary_little_endian", b"binary_big_endian", b"ascii")
    if (len(declared) != 3 or declared[0] != b"format"
            or declared[1] not in supported or declared[2] != b"1.0"):
        raise ValueError("only PLY 1.0 in binary little/big endian or ASCII is supported")
    elements: list[PlyElement] = []
    names: set[str] = set()
    properties: list[PlyProperty] = []
    count = None

    def close_element() -> None:
        if count is None:
            return
        elements.append(PlyElement(current, count, tuple(properties)))

    current = ""
    for line in lines[2:]:
        words = line.split()
        if words and words[0] in (b"comment", b"obj_info"):
            continue
        if len(words) == 3 and words[0] == b"element":
            close_element()
            if len(elements) >= MAX_ELEMENTS:
                raise ValueError("too many PLY elements")
            name = words[1]
            token = words[2]
            if (not name.isascii() or name in names or not token.isdigit()
                    or len(token) > 9):
                raise ValueError("invalid or duplicate PLY element")
            names.add(name.decode("ascii"))
            count = int(token)
            if count > limit:
                raise ValueError("element count exceeds the declared limit")
            current = name.decode("ascii")
            properties = []
            continue
        if count is None:
            raise ValueError("property declared before any element")
        if len(words) >= 2 and words[0] == b"property":
            if len(properties) >= MAX_PROPERTIES:
                raise ValueError("invalid, duplicate or excessive PLY properties")
            if words[1] == b"list":
                if len(words) != 5:
                    raise ValueError("invalid PLY list property")
                count_type, item_type, name = (word.decode("ascii", "replace")
                                               for word in words[2:5])
                if count_type not in INTEGER_TYPES or item_type not in SCALAR_SIZES:
                    raise ValueError("invalid PLY list property types")
            else:
                if len(words) != 3:
                    raise ValueError("invalid PLY property")
                count_type = item_type = None
                scalar_type, name = (word.decode("ascii", "replace") for word in words[1:3])
                if scalar_type not in SCALAR_SIZES:
                    raise ValueError("invalid PLY property type")
                item_type = scalar_type
                count_type = None
            if (not name.isascii() or name in {prop.name for prop in properties}
                    or not name):
                raise ValueError("invalid or duplicate PLY property name")
            properties.append(PlyProperty(item_type, name, count_type,
                                          item_type if count_type else None))
            continue
        raise ValueError("unsupported PLY header line")
    close_element()
    if count is None:
        raise ValueError("missing PLY element")
    return declared[1].decode("ascii"), offset, tuple(elements)


def _fixed_stride(element: PlyElement) -> int | None:
    if any(prop.is_list for prop in element.properties):
        return None
    return sum(SCALAR_SIZES[prop.type] for prop in element.properties)


def _spans(payload: bytes, offset: int, elements, encoding: str) -> tuple[PlyElementSpan, ...]:
    """Walk the body so every element range is known and bounds-checked."""
    order = "little" if encoding == "binary_little_endian" else "big"
    size = len(payload)
    position = offset
    spans = []
    for element in elements:
        start = position
        stride = _fixed_stride(element)
        if stride is not None:
            end = start + element.count * stride
            if end > size:
                raise ValueError("PLY body size does not match the declared elements")
            position = end
            spans.append(PlyElementSpan(element.name, element.count, element.properties,
                                        stride, start, end))
            continue
        widths = set()
        for _ in range(element.count):
            record_start = position
            for prop in element.properties:
                if not prop.is_list:
                    position += SCALAR_SIZES[prop.type]
                    continue
                count_size = SCALAR_SIZES[prop.count_type]
                if position + count_size > size:
                    raise ValueError("PLY list count runs past the end of the body")
                item_count = int.from_bytes(payload[position:position + count_size], order)
                position += count_size + item_count * SCALAR_SIZES[prop.item_type]
            if position > size:
                raise ValueError("PLY record runs past the end of the body")
            widths.add(position - record_start)
        stride = widths.pop() if len(widths) == 1 else 0
        spans.append(PlyElementSpan(element.name, element.count, element.properties,
                                    stride, start, position))
    if position != size:
        raise ValueError("PLY body has trailing bytes")
    return tuple(spans)


def _check_ascii_record(tokens: list[bytes], element: PlyElement) -> None:
    """Check one text record against the fields its element declares."""
    index = 0
    for prop in element.properties:
        if not prop.is_list:
            index += 1
            continue
        if index >= len(tokens):
            raise ValueError("ASCII list record is missing its arity")
        arity = tokens[index]
        index += 1
        if not arity.isdigit() or len(arity) > 9:
            raise ValueError("ASCII list arity is not a decimal integer")
        index += int(arity)
    if index != len(tokens):
        raise ValueError("ASCII record does not match its declared fields")


def _ascii_structure(body: bytes, elements, *, tolerant: bool = False) -> tuple[list, bool, list, bool]:
    """Return per-element token rows, canonicity, each element's row width, and had_unparsed.

    Canonicity means every line is exactly its tokens joined by one space and the
    body ends with a newline. Only then can a row be rebuilt from its tokens, so
    only then may the column layout be offered.
    """
    lines = body.split(b"\n")
    canonical = bool(lines) and lines[-1] == b""
    if canonical:
        lines.pop()
    if len(lines) != sum(element.count for element in elements):
        raise ValueError("ASCII body does not hold one line per declared record")
    rows = []
    widths: list[int | None] = []
    had_unparsed = False
    position = 0
    for element in elements:
        element_rows = []
        for _ in range(element.count):
            line = lines[position]
            position += 1
            tokens = line.split()
            try:
                _check_ascii_record(tokens, element)
            except ValueError:
                if not tolerant:
                    raise
                canonical = False
                had_unparsed = True
            if line != b" ".join(tokens):
                canonical = False
            element_rows.append(tokens)
        rows.append(element_rows)
        observed = {len(row) for row in element_rows}
        widths.append(observed.pop() if len(observed) == 1 else (0 if not element_rows else None))
    return rows, canonical, widths, had_unparsed


def _ascii_columns(rows: list[list[bytes]], columns: int) -> bytes:
    output = bytearray()
    for column in range(columns):
        for row in rows:
            output += row[column] + b"\n"
    return bytes(output)


def _ascii_from_column_stream(body: bytes, shape) -> bytes:
    tokens = body.split()
    if len(tokens) != sum(count * width for count, width in shape):
        raise ValueError("transposed ASCII body does not match its declared shape")
    output = bytearray()
    position = 0
    for count, width in shape:
        for index in range(count):
            output += b" ".join(tokens[position + column * count + index]
                                for column in range(width)) + b"\n"
        position += count * width
    return bytes(output)


def inspect_ply(data: bytes, *, limit=MAX_RAW_BYTES, tolerant: bool = False) -> PlyLayout:
    """Validate structure and byte counts, not renderability or numeric values."""
    encoding, header_bytes, elements = _parse_header(data, limit=limit)
    if encoding == "ascii":
        _, _, _, had_unparsed = _ascii_structure(data[header_bytes:], elements, tolerant=tolerant)
        spans = tuple(PlyElementSpan(element.name, element.count, element.properties,
                                     0, header_bytes, len(data)) for element in elements)
        return PlyLayout(encoding, header_bytes, spans, unparsed_ascii=had_unparsed)
    else:
        spans = _spans(data, header_bytes, elements, encoding)
    return PlyLayout(encoding, header_bytes, spans)


def encode_ply(data: bytes, *, layout="auto", method="auto", limit=MAX_RAW_BYTES, tolerant: bool = False) -> bytes:
    """Compare complete archive sizes; ties prefer the original record layout."""
    info = inspect_ply(data, limit=limit, tolerant=tolerant)
    source_digest = hashlib.sha256(data).hexdigest()
    header_bytes = info.header_bytes
    body = data[header_bytes:]
    if info.ascii:
        if layout not in ("auto", *ASCII_LAYOUTS):
            raise ValueError("ASCII PLY accepts the opaque or columns layout")
        rows, canonical, widths, had_unparsed = _ascii_structure(body, info.elements, tolerant=tolerant)
        shape = [[element.count, widths[index] or 0]
                 for index, element in enumerate(info.elements)]
        transposable = (canonical and not had_unparsed
                        and all(element.count == 0 or widths[index]
                                for index, element in enumerate(info.elements)))
        choices = ASCII_LAYOUTS if layout == "auto" else (layout,)
        candidates = []
        for choice in choices:
            if choice == "columns":
                if not transposable:
                    if layout == "auto":
                        continue
                    raise ValueError("the ASCII column layout needs canonical rows of "
                                     "uniform width")
                payload = b"".join(_ascii_columns(rows[index], widths[index])
                                   for index in range(len(rows)))
            else:
                payload = body
            meta = {
                "kind": ASCII_KIND, "layout": choice, "source_sha256": source_digest,
                "shape": shape,
            }
            if had_unparsed:
                meta["tolerant"] = True
            candidates.append(pack(data[:header_bytes] + payload, meta, method=method, limit=limit))
        if not candidates:
            raise ValueError("the ASCII column layout needs canonical rows of uniform width")
        return min(candidates, key=len)
    if layout not in ("auto", *BINARY_LAYOUTS):
        raise ValueError("binary PLY accepts the original or byte-planes layout")
    choices = BINARY_LAYOUTS if layout == "auto" else (layout,)
    candidates = []
    for choice in choices:
        records: list[list[int]] = []
        payload = data
        if choice == "byte-planes":
            planes = bytearray(data)
            for element in info.elements:
                if not element.planar:
                    continue
                chunk = bytes(planes[element.start:element.end])
                planes[element.start:element.end] = _planes(chunk, element.stride)
                records.append([element.start, element.end, element.stride])
            payload = bytes(planes)
        candidates.append(pack(payload, {
            "kind": BINARY_KIND, "layout": choice, "source_sha256": source_digest,
            "planar": records}, method=method, limit=limit))
    return min(candidates, key=len)


def _fingerprint(metadata) -> str:
    value = metadata["source_sha256"]
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise ValueError("invalid source fingerprint")
    return value


def _decode_ascii(packet: bytes, metadata, max_output_bytes) -> bytes:
    layout = metadata["layout"]
    legacy = "shape" not in metadata
    tolerant = bool(metadata.get("tolerant", False))
    expected = ({"kind", "layout", "source_sha256", "count", "columns"} if legacy else
                ({"kind", "layout", "source_sha256", "shape", "tolerant"} if tolerant
                 else {"kind", "layout", "source_sha256", "shape"}))
    if set(metadata) != expected or layout not in ASCII_LAYOUTS:
        raise ValueError("invalid ASCII PLY metadata")
    fingerprint = _fingerprint(metadata)
    transformed, _ = unpack(packet, kind=ASCII_KIND, max_output_bytes=max_output_bytes)
    encoding, header_bytes, elements = _parse_header(transformed, limit=max_output_bytes)
    if encoding != "ascii":
        raise ValueError("ASCII header disagrees with the packet metadata")
    if legacy:
        count = _integer(metadata["count"], "ASCII record count", 0, ABSOLUTE_MAX_BYTES)
        columns = _integer(metadata["columns"], "ASCII column count", 0, MAX_PROPERTIES)
        if len(elements) != 1 or elements[0].count != count \
                or len(elements[0].properties) != columns:
            raise ValueError("ASCII header disagrees with the packet metadata")
        shape = [[count, columns]]
    else:
        shape = _shape(metadata["shape"], elements)
    if layout == "opaque":
        data = transformed
    else:
        data = transformed[:header_bytes] + _ascii_from_column_stream(
            transformed[header_bytes:], shape)
    # Re-parse the reconstructed body so a declared shape cannot disagree with
    # the bytes it claims to describe, even on a verbatim packet.
    _, _, observed, _ = _ascii_structure(data[header_bytes:], elements, tolerant=tolerant)
    for index, (_, width) in enumerate(shape):
        if (observed[index] or 0) != width:
            raise ValueError("ASCII shape disagrees with the reconstructed body")
    if hashlib.sha256(data).hexdigest() != fingerprint:
        raise ValueError("reconstructed PLY fingerprint mismatch")
    return data


def _shape(value, elements) -> list:
    if not isinstance(value, list) or len(value) != len(elements):
        raise ValueError("invalid ASCII shape")
    shape = []
    for entry, element in zip(value, elements):
        if (not isinstance(entry, list) or len(entry) != 2
                or any(type(item) is not int for item in entry)):
            raise ValueError("invalid ASCII shape")
        count, width = entry
        if count != element.count or not 0 <= width <= 1 << 20:
            raise ValueError("ASCII shape disagrees with the header")
        if count and width < 1:
            raise ValueError("ASCII shape cannot rebuild zero-width records")
        shape.append([count, width])
    return shape


def _records(value, size: int) -> list[list[int]]:
    if not isinstance(value, list):
        raise ValueError("invalid PLY planar ranges")
    records = []
    previous_end = -1
    for entry in value:
        if (not isinstance(entry, list) or len(entry) != 3
                or any(type(item) is not int for item in entry)):
            raise ValueError("invalid PLY planar range")
        start, end, stride = entry
        if not 0 <= start < end <= size or stride < 2 or (end - start) % stride:
            raise ValueError("invalid PLY planar range")
        if start < previous_end:
            raise ValueError("overlapping PLY planar ranges")
        previous_end = end
        records.append([start, end, stride])
    return records


def _legacy_records(payload: bytes, layout: str, limit: int) -> list[list[int]]:
    """Earlier packets transposed the whole body of a single-element file."""
    if layout == "original":
        return []
    encoding, header_bytes, elements = _parse_header(payload, limit=limit)
    if encoding == "ascii" or len(elements) != 1:
        raise ValueError("legacy PLY packet is not a single binary element")
    stride = _fixed_stride(elements[0])
    if not stride:
        raise ValueError("legacy PLY packet has no fixed record layout")
    return _records([[header_bytes, len(payload), stride]], len(payload))


def decode_ply(packet: bytes, *, max_output_bytes=MAX_RAW_BYTES) -> bytes:
    report = inspect_packet(packet, max_output_bytes=max_output_bytes)
    metadata = report["metadata"]
    if metadata.get("kind") == ASCII_KIND:
        return _decode_ascii(packet, metadata, max_output_bytes)
    legacy = "planar" not in metadata
    expected = {"kind", "layout", "source_sha256"} if legacy else {
        "kind", "layout", "source_sha256", "planar"}
    if set(metadata) != expected or metadata["kind"] != BINARY_KIND \
            or metadata["layout"] not in BINARY_LAYOUTS:
        raise ValueError("invalid PLY archive metadata")
    fingerprint = _fingerprint(metadata)
    transformed, _ = unpack(packet, kind=BINARY_KIND, max_output_bytes=max_output_bytes)
    records = (_legacy_records(transformed, metadata["layout"], max_output_bytes) if legacy
               else _records(metadata["planar"], len(transformed)))
    data = bytearray(transformed)
    for start, end, stride in records:
        data[start:end] = _interleave(bytes(transformed[start:end]), stride)
    data = bytes(data)
    inspect_ply(data, limit=max_output_bytes)
    if hashlib.sha256(data).hexdigest() != fingerprint:
        raise ValueError("reconstructed PLY fingerprint mismatch")
    return data
