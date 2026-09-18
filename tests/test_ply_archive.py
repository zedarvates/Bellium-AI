import hashlib
import random
import struct

import pytest

from bellium.compression.container import METHODS, inspect_packet, pack
from bellium.compression.ply_archive import KIND, decode_ply, encode_ply, inspect_ply
from bellium.specialists.ply_archive import archive_ply


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
    lambda data: data.replace(b"property float x", b"property list uchar float x"),
    lambda data: data.replace(b"end_header", b"element face 0\nend_header"),
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
