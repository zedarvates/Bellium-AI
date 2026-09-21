"""Consultative compression experiments. Callers own storage and transmission."""

from bellium.compression import encode_image, encode_splats, encode_stream, inspect_packet
from bellium.contracts import AuthorityMode, SpecialistResult


def compress(data: bytes, *, kind: str, method="auto", **options) -> SpecialistResult:
    encoders = {"image": encode_image, "stream": encode_stream, "splats": encode_splats}
    if kind not in encoders:
        raise ValueError("kind must be image, stream or splats")
    packet = encoders[kind](data, method=method, **options)
    report = inspect_packet(packet)
    return SpecialistResult(
        specialist_id=f"bellium/hybrid/{kind}-compression:v0",
        output={"packet": packet, **report}, confidence=None, abstained=False,
        authority_mode=AuthorityMode.CONSULTATIVE,
        evidence_ref="docs/model-cards/compression-v0.md",
        notes=("Local research codec; measured size, no estimated confidence.",
               "No production quality or runtime integration claim."),
    )
