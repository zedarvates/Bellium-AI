"""Every language result must keep its evidence class explicit."""

from __future__ import annotations

EVIDENCE_LABELS = ("attested", "reconstructed", "inferred", "speculative")
_RANK = {
    "attested": 3,
    "reconstructed": 2,
    "inferred": 1,
    "speculative": 0,
}


def parse_evidence(value: object) -> str:
    label = str(value).strip().casefold()
    if label not in _RANK:
        raise ValueError(
            "evidence must be one of: attested, reconstructed, inferred, speculative"
        )
    return label


def weakest_evidence(labels: list[str]) -> str:
    if not labels:
        raise ValueError("need at least one evidence label")
    parsed = [parse_evidence(label) for label in labels]
    return min(parsed, key=_RANK.__getitem__)


def can_certify(evidence: str) -> bool:
    return parse_evidence(evidence) == "attested"

