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
    print(json.dumps({"installed_package": str(package), "exact_roundtrips": roundtrips,
                      "research_spatial_roundtrips": len(PREDICTORS),
                      "specialist_wrapper": True, "passed": True}))


if __name__ == "__main__":
    main()
