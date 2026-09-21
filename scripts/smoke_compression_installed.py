"""Exercise BLCP from an installed wheel with Python -I, outside the source tree."""

import json
from pathlib import Path
import struct

import bellium
from bellium.compression import (
    decode_image, decode_splats, decode_stream, encode_image, encode_splats, encode_stream,
)
from bellium.registry.catalog import get_specialist
from bellium.specialists.compression import compress


def main():
    package = Path(bellium.__file__).resolve()
    if "site-packages" not in package.parts:
        raise RuntimeError("run against an installed wheel, not a source checkout")
    streams = bytes(range(256)) * 17
    pixels = bytes([200, 25, 100, 0, 15, 220, 55, 255]) * 32
    base = struct.pack("<14f", 1, 2, -0.0, 0, 0, 0, 1, .1, .2, .3, .8, .2, .3, .4) * 32
    current = struct.pack("<14f", 1.125, 2, -0.0, 0, 0, 0, 1, .1, .2, .3, .8, .2, .3, .4) * 32
    roundtrips = 0
    for method in ("raw", "zlib", "delta", "knn", "micro-nn", "auto"):
        assert decode_stream(encode_stream(streams, method=method)) == streams
        packet = encode_image(pixels, width=8, height=8, mode="RGBA", method=method)
        assert decode_image(packet).pixels == pixels
        assert decode_splats(encode_splats(base, method=method)) == base
        packet = encode_splats(current, base=base, method=method)
        assert decode_splats(packet, base=base) == current
        roundtrips += 4
    result = compress(streams, kind="stream")
    assert decode_stream(result.output["packet"]) == streams
    for kind in ("image", "stream", "splats"):
        assert not get_specialist(f"bellium/hybrid/{kind}-compression:v0").decision_eligible
    from bellium.compression.spatial_experiment import (
        PREDICTORS, decode_spatial_image, encode_spatial_image,
    )
    for predictor in PREDICTORS:
        packet = encode_spatial_image(pixels, width=8, height=8, mode="RGBA", predictor=predictor)
        assert decode_spatial_image(packet).pixels == pixels
    from bellium.compression.ply_archive import decode_ply, encode_ply
    from bellium.specialists.ply_archive import archive_ply
    header = (b"ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
              b"property float x\nproperty float f_rest_0\nproperty uchar tag\nend_header\n")
    ply = header + struct.pack("<ffB", -0.0, 0.25, 7) + struct.pack("<ffB", 1.5, -2.0, 0)
    archive_roundtrips = 0
    for layout in ("original", "byte-planes", "auto"):
        assert decode_ply(encode_ply(ply, layout=layout)) == ply
        archive_roundtrips += 1
    assert decode_ply(archive_ply(ply).output["packet"]) == ply
    assert not get_specialist("bellium/hybrid/ply-archive:v0").decision_eligible
    ascii_ply = (b"ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\n"
                 b"property uchar tag\nend_header\n1.5 3\n2.5 4\n")
    for layout in ("opaque", "columns", "auto"):
        assert decode_ply(encode_ply(ascii_ply, layout=layout)) == ascii_ply
    header_only = b"ply\nformat ascii 1.0\nelement vertex 0\nend_header\n"
    assert decode_ply(encode_ply(header_only)) == header_only
    mesh_header = (b"ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
                   b"property float x\nproperty float y\nproperty float z\n"
                   b"element face 1\nproperty list uchar int vertex_indices\nend_header\n")
    mesh = (mesh_header + struct.pack("<ffffff", 0, 0, 0, 1, 0, 0)
            + struct.pack("<B", 3) + struct.pack("<iii", 0, 1, 0))
    from bellium.compression.ply_archive import inspect_ply
    assert [element.name for element in inspect_ply(mesh).elements] == ["vertex", "face"]
    for layout in ("original", "byte-planes", "auto"):
        assert decode_ply(encode_ply(mesh, layout=layout)) == mesh
    ascii_mesh = (b"ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\n"
                  b"element face 1\nproperty list uchar int vertex_indices\nend_header\n"
                  b"0.0 0.0\n1.0 0.0\n3 0 1 0\n")
    ragged = ascii_mesh.replace(b"0.0 0.0\n", b"0.0  0.0\n")
    for sample in (ascii_mesh, ragged):
        for layout in ("opaque", "auto"):
            assert decode_ply(encode_ply(sample, layout=layout)) == sample
    assert decode_ply(encode_ply(ascii_mesh, layout="columns")) == ascii_mesh
    print(json.dumps({"installed_package": str(package), "exact_roundtrips": roundtrips,
                      "research_spatial_roundtrips": len(PREDICTORS),
                      "full_ply_archive_roundtrips": archive_roundtrips,
                      "ascii_ply_roundtrips": 4,
                      "polygonal_ply_roundtrips": 3,
                      "multi_element_ascii_roundtrips": 5,
                      "specialist_wrapper": True, "passed": True}))


if __name__ == "__main__":
    main()
