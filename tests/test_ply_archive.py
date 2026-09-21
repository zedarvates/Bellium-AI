import hashlib
import random
import struct

import pytest

from bellium.compression.container import MAX_RAW_BYTES, METHODS, inspect_packet, pack, unpack
from bellium.compression.ply_archive import (
    ASCII_KIND, ASCII_LAYOUTS, KIND, decode_ply, encode_ply, inspect_ply,
)
from bellium.specialists.ply_archive import archive_ply

ASCII_TYPES = (b"float", b"float", b"float", b"uchar")
ASCII_NAMES = (b"x", b"y", b"f_rest_0", b"tag")


def ascii_fixture(*, count=3, columns=4, separator=b" ", newline=b"\n", trailing=True):
    header = (b"ply\nformat ascii 1.0\ncomment ascii fixture\nelement vertex "
              + str(count).encode() + b"\n"
              + b"".join(b"property " + kind + b" " + name + b"\n"
                         for kind, name in zip(ASCII_TYPES[:columns], ASCII_NAMES[:columns]))
              + b"end_header\n")
    rows = [separator.join(f"{index * columns + column}.5".encode() for column in range(columns))
            + newline for index in range(count)]
    body = b"".join(rows)
    if body and not trailing:
        body = body[: -len(newline)]
    return header + body


def mesh_fixture(*, count=40, faces=None, endian="little"):
    """A polygonal file: a vertex element plus a face element with a list."""
    faces = faces if faces is not None else [(index, index + 1, index + 2)
                                             for index in range(0, count - 2, 3)]
    order = "<" if endian == "little" else ">"
    header = (f"ply\nformat binary_{endian}_endian 1.0\nelement vertex {count}\n".encode()
              + b"property float x\nproperty float y\nproperty float z\n"
              + f"element face {len(faces)}\n".encode()
              + b"property list uchar int vertex_indices\nend_header\n")
    body = b"".join(struct.pack(order + "fff", index * 0.5, index * 0.25, -index * 0.125)
                    for index in range(count))
    for face in faces:
        body += struct.pack(order + "B", len(face)) + struct.pack(
            order + "i" * len(face), *face)
    return header + body


def fixture(*, count=7, crlf=False, endian="little", nonfinite=False):
    ending = b"\r\n" if crlf else b"\n"
    lines = [b"ply", f"format binary_{endian}_endian 1.0".encode(),
             b"comment exact header  \xc3\xa9", b"obj_info custom export",
             f"element vertex {count}".encode(), b"property float x",
             b"property float f_rest_0", b"property uchar custom_id",
             b"property double vendor_extra", b"property short material", b"end_header"]
    header = ending.join(lines) + ending
    order = "<" if endian == "little" else ">"
    rows = [struct.pack(order + "ffBdh", -0.0, i / 3, i, 1e-20 * i, -1) for i in range(count)]
    if nonfinite and rows:
        # Keep a specific NaN payload and infinity as opaque bits.
        rows[0] = struct.pack(order + "II", 0x7fc01234, 0x7f800000) + rows[0][8:]
    return header + b"".join(rows)


@pytest.mark.parametrize("method", (*METHODS, "auto"))
@pytest.mark.parametrize("layout", ("original", "byte-planes", "auto"))
@pytest.mark.parametrize("endian,crlf", (("little", False), ("big", True)))
def test_full_ply_preserves_all_bytes(method, layout, endian, crlf):
    source = fixture(endian=endian, crlf=crlf, nonfinite=True)
    packet = encode_ply(source, method=method, layout=layout)
    assert decode_ply(packet) == source
    assert inspect_ply(decode_ply(packet)).properties[-1] == ("short", "material")


@pytest.mark.parametrize("count", (0, 1))
def test_empty_and_single_vertex_files(count):
    source = fixture(count=count)
    assert decode_ply(encode_ply(source)) == source


def test_auto_never_exceeds_an_explicit_candidate():
    source = fixture()
    packets = [encode_ply(source, layout=layout, method=method)
               for layout in ("original", "byte-planes") for method in METHODS]
    assert len(encode_ply(source)) == min(map(len, packets))


@pytest.mark.parametrize("modify", (
    lambda data: data.replace(b"binary_little_endian", b"ascii"),
    lambda data: data.replace(b"property float x", b"property list float uchar x"),
    lambda data: data.replace(b"vertex 7", b"vertex 999999999999999999999"),
    lambda data: data.replace(b"vertex 7", b"vertex -1"),
    lambda data: data.replace(b"vertex 7", b"vertex 8"),
    lambda data: data.replace(b"property float f_rest_0", b"property float x"),
    lambda data: data.replace(b"property float x", b"property object x"),
    lambda data: data[:-1], lambda data: data + b"tail",
    lambda data: data.replace(b"end_header\n", b""),
))
def test_unsupported_or_malformed_layouts_refused(modify):
    with pytest.raises(ValueError):
        encode_ply(modify(fixture()))


def test_long_header_and_excess_properties_refused():
    with pytest.raises(ValueError, match="header"):
        inspect_ply(b"ply\n" + b"comment " + b"x" * 17000 + b"\nend_header\n")
    data = (b"ply\nformat binary_little_endian 1.0\nelement vertex 0\n"
            + b"".join(f"property uchar p{i}\n".encode() for i in range(257)) + b"end_header\n")
    with pytest.raises(ValueError, match="excessive"):
        inspect_ply(data)


def test_bad_packets_and_decoded_structure_are_rejected():
    source = fixture()
    packet = encode_ply(source)
    with pytest.raises(ValueError):
        decode_ply(packet[:-1])
    with pytest.raises(ValueError, match="limit"):
        decode_ply(packet, max_output_bytes=10)
    meta = {"kind": KIND, "layout": "original", "source_sha256": hashlib.sha256(source).hexdigest()}
    with pytest.raises(ValueError, match="body size"):
        decode_ply(pack(source[:-1], meta))
    with pytest.raises(ValueError, match="fingerprint"):
        decode_ply(pack(source[:-1] + bytes([source[-1] ^ 1]), meta))
    with pytest.raises(ValueError, match="metadata"):
        decode_ply(pack(source, {**meta, "layout": "other"}))


def test_all_scalar_widths_and_opaque_byte_patterns():
    names = ["char", "uchar", "short", "ushort", "int", "uint", "float", "double",
             "int8", "uint8", "int16", "uint16", "int32", "uint32", "float32", "float64"]
    header = ("ply\nformat binary_big_endian 1.0\nelement vertex 3\n"
              + "".join(f"property {name} p{i}\n" for i, name in enumerate(names))
              + "end_header\n").encode()
    rng = random.Random(917)
    source = header + bytes(rng.randrange(256) for _ in range(3 * 52))
    assert inspect_ply(source).record_bytes == 52
    assert decode_ply(encode_ply(source)) == source


def test_large_files_keep_the_predictive_work_limit():
    header = b"ply\nformat binary_little_endian 1.0\nelement vertex 70000\nproperty uchar x\nend_header\n"
    source = header + bytes(70000)
    with pytest.raises(ValueError, match="65536"):
        encode_ply(source, method="micro-nn")
    assert inspect_packet(encode_ply(source))["method"] in ("raw", "zlib", "delta")
    assert decode_ply(encode_ply(source)) == source


def test_specialist_is_consultative_and_discoverable():
    from bellium.contracts import AuthorityMode
    from bellium.registry.catalog import get_specialist
    source = fixture()
    result = archive_ply(source)
    assert result.authority_mode is AuthorityMode.CONSULTATIVE
    assert result.confidence is None and result.output["full_file_preserved"]
    assert decode_ply(result.output["packet"]) == source
    assert not get_specialist(result.specialist_id).decision_eligible


@pytest.mark.parametrize("method", (*METHODS, "auto"))
@pytest.mark.parametrize("layout", (*ASCII_LAYOUTS, "auto"))
def test_ascii_roundtrip_is_byte_exact(method, layout):
    source = ascii_fixture(count=5)
    packet = encode_ply(source, layout=layout, method=method)
    assert decode_ply(packet) == source
    info = inspect_ply(decode_ply(packet))
    assert info.ascii and info.record_bytes == 0
    assert info.vertex_count == 5 and len(info.properties) == 4


def test_ascii_columns_layout_is_really_column_major():
    source = ascii_fixture(count=3)
    packet = encode_ply(source, layout="columns", method="zlib")
    body, _ = unpack(packet, kind=ASCII_KIND)
    header_bytes = inspect_ply(source).header_bytes
    assert body[header_bytes:] == (
        b"0.5\n4.5\n8.5\n1.5\n5.5\n9.5\n2.5\n6.5\n10.5\n3.5\n7.5\n11.5\n")


def test_ascii_auto_never_exceeds_an_explicit_candidate():
    source = ascii_fixture(count=6)
    sizes = [len(encode_ply(source, layout=layout, method=method))
             for layout in ASCII_LAYOUTS for method in METHODS]
    assert len(encode_ply(source)) == min(sizes)


def test_header_only_ascii_file_is_accepted():
    source = b"ply\nformat ascii 1.0\nelement vertex 0\nend_header\n"
    assert inspect_ply(source).vertex_count == 0
    assert decode_ply(encode_ply(source)) == source


@pytest.mark.parametrize("kwargs", (
    {"separator": b"  "}, {"separator": b"\t"}, {"trailing": False}, {"newline": b"\r\n"},
))
def test_non_canonical_ascii_is_archived_verbatim_but_never_transposed(kwargs):
    source = ascii_fixture(count=2, **kwargs)
    assert decode_ply(encode_ply(source)) == source
    assert decode_ply(encode_ply(source, layout="opaque")) == source
    with pytest.raises(ValueError, match="canonical"):
        encode_ply(source, layout="columns")


def test_ascii_token_count_must_match_the_header():
    source = ascii_fixture(count=2)
    broken = source[: source.index(b"end_header")] + b"end_header\n1.5 2.5\n"
    with pytest.raises(ValueError, match="one line per declared record"):
        encode_ply(broken)


def test_ascii_undeclared_fields_refused_by_default_and_accepted_with_tolerant():
    # Header declares 2 float properties, but body lines have 3 tokens
    header = (b"ply\nformat ascii 1.0\nelement vertex 2\n"
              b"property float x\nproperty float y\nend_header\n")
    source = header + b"1.0 2.0 999.0\n3.0 4.0 888.0\n"
    with pytest.raises(ValueError, match="match its declared fields"):
        inspect_ply(source)
    with pytest.raises(ValueError, match="match its declared fields"):
        encode_ply(source)
    info = inspect_ply(source, tolerant=True)
    assert info.unparsed_ascii is True
    packet = encode_ply(source, tolerant=True)
    assert decode_ply(packet) == source
    # Transposition must not be allowed for unparsed schema
    with pytest.raises(ValueError, match="column layout"):
        encode_ply(source, layout="columns", tolerant=True)
    res = archive_ply(source, tolerant=True)
    assert res.output["tolerant"] is True
    assert decode_ply(res.output["packet"]) == source


def test_fuze_ascii_real_file_escape_hatch():
    from pathlib import Path
    path = Path("C:/BelliumAI/ply-real-2026-09-18/corpus/fuze_ascii.ply")
    if not path.is_file():
        pytest.skip("fuze_ascii.ply not in local corpus")
    data = path.read_bytes()
    with pytest.raises(ValueError, match="match its declared fields"):
        encode_ply(data)
    info = inspect_ply(data, tolerant=True)
    assert info.unparsed_ascii is True
    packet = encode_ply(data, tolerant=True)
    assert decode_ply(packet) == data


def test_layouts_do_not_leak_between_ascii_and_binary():
    with pytest.raises(ValueError, match="ASCII PLY accepts"):
        encode_ply(ascii_fixture(), layout="byte-planes")
    with pytest.raises(ValueError, match="binary PLY accepts"):
        encode_ply(fixture(), layout="columns")


def test_ascii_metadata_must_agree_with_the_header():
    source = ascii_fixture(count=3)
    digest = hashlib.sha256(source).hexdigest()
    for count, columns in ((4, 4), (3, 3)):
        packet = pack(source, {"kind": ASCII_KIND, "layout": "opaque",
                               "source_sha256": digest, "count": count, "columns": columns})
        with pytest.raises(ValueError, match="disagrees"):
            decode_ply(packet)
    with pytest.raises(ValueError, match="metadata"):
        decode_ply(pack(source, {"kind": ASCII_KIND, "layout": "other",
                                 "source_sha256": digest, "count": 3, "columns": 4}))


def test_declared_limit_gates_large_files_without_changing_the_default():
    count = MAX_RAW_BYTES + 1
    header = (b"ply\nformat binary_little_endian 1.0\nelement vertex "
              + str(count).encode() + b"\nproperty uchar x\nend_header\n")
    source = header + bytes(count)
    with pytest.raises(ValueError, match="declared byte limit"):
        inspect_ply(source)
    with pytest.raises(ValueError, match="declared byte limit"):
        encode_ply(source)
    limit = MAX_RAW_BYTES + 1024 * 1024
    packet = encode_ply(source, limit=limit)
    with pytest.raises(ValueError, match="limit"):
        decode_ply(packet)
    assert decode_ply(packet, max_output_bytes=limit) == source


@pytest.mark.parametrize("limit", (0, -1, 256 * 1024 * 1024 + 1))
def test_declared_limit_is_bounded(limit):
    with pytest.raises(ValueError, match="limit"):
        encode_ply(fixture(), limit=limit)


@pytest.mark.parametrize("method", (*METHODS, "auto"))
@pytest.mark.parametrize("layout", ("original", "byte-planes", "auto"))
@pytest.mark.parametrize("endian", ("little", "big"))
def test_polygonal_mesh_roundtrip_keeps_every_byte(method, layout, endian):
    source = mesh_fixture(endian=endian)
    decoded = decode_ply(encode_ply(source, layout=layout, method=method))
    assert decoded == source
    vertex, face = inspect_ply(decoded).elements
    assert (vertex.name, vertex.count) == ("vertex", 40)
    assert (face.name, face.count) == ("face", 13)


def test_face_element_is_planar_only_when_its_arity_is_uniform():
    uniform = mesh_fixture()
    assert inspect_ply(uniform).elements[1].planar
    packet = encode_ply(uniform, layout="byte-planes", method="zlib")
    body, meta = unpack(packet, kind=KIND)
    vertex, face = meta["planar"]
    assert (vertex[2], face[2]) == (12, 13)
    assert len(body) == len(uniform)
    assert decode_ply(packet) == uniform


def test_mixed_face_arity_is_archived_but_left_untouched():
    source = mesh_fixture(count=12, faces=[(0, 1, 2), (2, 3, 4, 5), (5, 6, 7)])
    element = inspect_ply(source).elements[1]
    assert element.stride == 0 and not element.planar
    _, meta = unpack(encode_ply(source, layout="byte-planes", method="zlib"), kind=KIND)
    assert len(meta["planar"]) == 1  # Only the vertex element can be transposed.
    assert decode_ply(encode_ply(source)) == source


def test_byte_planes_beat_verbatim_on_a_polygonal_mesh():
    source = mesh_fixture(count=300)
    verbatim = len(encode_ply(source, layout="original", method="zlib"))
    planes = len(encode_ply(source, layout="byte-planes", method="zlib"))
    assert planes < verbatim
    # Automatic selection also chooses among raw, delta and the learned methods,
    # so it can only ever be at least as small as the best forced zlib candidate.
    assert len(encode_ply(source)) <= min(planes, verbatim)


def test_zero_record_face_element_is_accepted():
    source = mesh_fixture(count=4, faces=[])
    assert inspect_ply(source).elements[1].count == 0
    assert decode_ply(encode_ply(source)) == source


@pytest.mark.parametrize("truncate", (1, 4, 9))
def test_truncated_face_records_are_refused(truncate):
    with pytest.raises(ValueError, match="record runs past|body size"):
        inspect_ply(mesh_fixture()[:-truncate])


def test_list_count_beyond_the_body_is_refused():
    header = (b"ply\nformat binary_little_endian 1.0\nelement vertex 0\n"
              b"element face 1\nproperty list uchar int vertex_indices\nend_header\n")
    with pytest.raises(ValueError, match="record runs past"):
        inspect_ply(header + struct.pack("<B", 5) + struct.pack("<iii", 1, 2, 3))


def test_unreasonable_list_count_is_refused_quickly():
    header = (b"ply\nformat binary_little_endian 1.0\nelement vertex 0\n"
              b"element face 1\nproperty list uint int idx\nend_header\n")
    with pytest.raises(ValueError, match="record runs past"):
        inspect_ply(header + struct.pack("<I", 4_000_000_000))


def test_planar_metadata_is_validated():
    source = mesh_fixture()
    base = {"kind": KIND, "layout": "byte-planes",
            "source_sha256": hashlib.sha256(source).hexdigest()}
    size = len(source)
    for planar in ([[1, 2, size]], [[0, 10, 3]], [[0, 12, 1]], [[0, 10, 2], [4, 20, 2]],
                   [["a", 4, 2]], [[0, 4]]):
        with pytest.raises(ValueError, match="planar range"):
            decode_ply(pack(source, {**base, "planar": planar}))
    with pytest.raises(ValueError, match="planar ranges"):
        decode_ply(pack(source, {**base, "planar": "ranges"}))


def test_cheating_planar_ranges_cannot_forge_a_different_file():
    source = mesh_fixture()
    header_bytes = inspect_ply(source).header_bytes
    packet = pack(source, {"kind": KIND, "layout": "byte-planes",
                           "source_sha256": hashlib.sha256(source).hexdigest(),
                           "planar": [[header_bytes, len(source), 3]]})
    with pytest.raises(ValueError, match="planar range|body|record|fingerprint"):
        decode_ply(packet)


def test_legacy_packets_without_planar_ranges_still_decode():
    from bellium.compression.codecs import _planes
    source = fixture()
    info = inspect_ply(source)
    body = source[info.header_bytes:]
    legacy = pack(source[:info.header_bytes] + _planes(body, info.record_bytes), {
        "kind": KIND, "layout": "byte-planes",
        "source_sha256": hashlib.sha256(source).hexdigest()})
    assert decode_ply(legacy) == source
    plain = pack(source, {"kind": KIND, "layout": "original",
                          "source_sha256": hashlib.sha256(source).hexdigest()})
    assert decode_ply(plain) == source


def test_multi_element_ascii_roundtrips_exactly():
    source = (b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\n"
              b"element face 1\nproperty list uchar int vertex_indices\nend_header\n"
              b"0.5\n3 0 0 0\n")
    info = inspect_ply(source)
    assert [element.name for element in info.elements] == ["vertex", "face"]
    for layout in ("opaque", "columns", "auto"):
        assert decode_ply(encode_ply(source, layout=layout)) == source
    _, meta = unpack(encode_ply(source, layout="columns", method="zlib"), kind=ASCII_KIND)
    assert meta["shape"] == [[1, 1], [1, 4]]


def test_ascii_list_arity_is_checked_against_the_tokens_present():
    header = (b"ply\nformat ascii 1.0\nelement vertex 0\n"
              b"element face 1\nproperty list uchar int vertex_indices\nend_header\n")
    assert decode_ply(encode_ply(header + b"3 0 1 2\n")) == header + b"3 0 1 2\n"
    for body in (b"4 0 1 2\n", b"x 0 1 2\n", b"2 0 1 2\n", b"\n"):
        with pytest.raises(ValueError, match="arity|declared fields|missing"):
            inspect_ply(header + body)


def test_ascii_shape_metadata_is_validated():
    source = ascii_fixture(count=3)
    digest = hashlib.sha256(source).hexdigest()
    base = {"kind": ASCII_KIND, "layout": "opaque", "source_sha256": digest}
    for shape in ([[2, 4]], [[3, 5]], [[3, 0]], [[3]], "shape", [[3, 1 << 21]]):
        with pytest.raises(ValueError, match="shape"):
            decode_ply(pack(source, {**base, "shape": shape}))


def test_legacy_single_element_ascii_packet_still_decodes():
    source = ascii_fixture(count=3)
    packet = pack(source, {"kind": ASCII_KIND, "layout": "opaque",
                           "source_sha256": hashlib.sha256(source).hexdigest(),
                           "count": 3, "columns": 4})
    assert decode_ply(packet) == source
    wrong = pack(source, {"kind": ASCII_KIND, "layout": "opaque",
                          "source_sha256": hashlib.sha256(source).hexdigest(),
                          "count": 4, "columns": 4})
    with pytest.raises(ValueError, match="disagrees"):
        decode_ply(wrong)


def test_element_count_is_bounded():
    header = b"ply\nformat binary_little_endian 1.0\n" + b"".join(
        f"element e{index} 0\n".encode() for index in range(65)) + b"end_header\n"
    with pytest.raises(ValueError, match="too many"):
        inspect_ply(header)
