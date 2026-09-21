"""Consultative complete binary-PLY archive; never writes or transmits files."""

from bellium.compression.container import MAX_RAW_BYTES, inspect_packet
from bellium.compression.ply_archive import encode_ply, inspect_ply
from bellium.contracts import AuthorityMode, SpecialistResult

SPECIALIST_ID = "bellium/hybrid/ply-archive:v0"


def archive_ply(data: bytes, *, layout="auto", method="auto", limit=MAX_RAW_BYTES, tolerant: bool = False) -> SpecialistResult:
    info = inspect_ply(data, limit=limit, tolerant=tolerant)
    packet = encode_ply(data, layout=layout, method=method, limit=limit, tolerant=tolerant)
    return SpecialistResult(
        specialist_id=SPECIALIST_ID,
        output={"packet": packet, **inspect_packet(packet), "vertex_count": info.vertex_count,
                "property_count": len(info.properties), "encoding": info.encoding,
                "full_file_preserved": True, "tolerant": info.unparsed_ascii},
        confidence=None, abstained=False, authority_mode=AuthorityMode.CONSULTATIVE,
        evidence_ref="docs/model-cards/ply-archive-v0.md",
        notes=("Supported binary or canonical ASCII PLY bytes archived exactly, all declared fields.",
               "Not numeric validation, rendering proof, .fovea conversion or a neural gain claim."),
    )
