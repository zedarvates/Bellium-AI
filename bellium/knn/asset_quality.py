"""Family-isolated asset-quality k-NN. Deterministic gates first, shadow advice after.

This is a Bellium-native implementation of the published algorithm. It never stores
local asset paths and never activates an asset.
"""

from __future__ import annotations

import json
import math
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.ledger import append_record, read_tail

SPECIALIST_ID = "bellium/knn/asset-quality:v0"
SCHEMA = "bellium.asset-quality/v1"
FAMILIES = ("image", "texture", "mesh", "animation", "godot")
VERDICTS = ("FAIL", "UNCERTAIN", "PASS", "PASS_ROBUST")
MIN_NEIGHBORS = 3
MIN_SIMILARITY = 0.70
MAX_ENTRIES = 5_000

_VERDICT_SCORE = {"FAIL": 0.0, "UNCERTAIN": 0.4, "PASS": 0.8, "PASS_ROBUST": 1.0}
_COMMON_CHECKS = ("decodable", "license_verified", "manifest_verified")
_FAMILY_CHECKS = {
    "image": ("dimensions_valid",),
    "texture": ("dimensions_valid",),
    "mesh": ("finite_geometry", "nonempty_geometry"),
    "animation": ("valid_timeline",),
    "godot": ("importable",),
}
_FEATURES = {
    "image": ("aesthetic", "artifact_free", "composition", "prompt_alignment", "technical"),
    "texture": ("artifact_free", "prompt_alignment", "seamless", "technical", "tiling"),
    "mesh": ("manifold", "normals", "prompt_alignment", "scale", "topology", "uv"),
    "animation": ("continuity", "loop_quality", "prompt_alignment", "timing"),
    "godot": ("import_health", "performance", "prompt_alignment", "runtime_health"),
}
_VERIFIER_FAMILIES = {
    "benchmark", "ci", "deterministic", "harness", "human", "independent",
    "pytest", "replay", "schema", "tests",
}


@dataclass(frozen=True)
class AssetAdvice:
    status: str
    verdict: str
    reason: str
    family: str
    evidence_strength: float
    neighbor_count: int
    neighbors: list[dict]
    failed_checks: list[str]
    missing_checks: list[str]
    shadow_only: bool = True
    acted: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    def to_result(self) -> SpecialistResult:
        abstained = self.status in {"abstain", "incomplete"}
        confidence = 1.0 if self.status == "rule_fail" else self.evidence_strength
        return SpecialistResult(
            specialist_id=SPECIALIST_ID,
            output=self.to_dict(),
            confidence=confidence,
            abstained=abstained,
            authority_mode=AuthorityMode.SHADOW,
            notes=("Learned advice cannot publish, import or activate an asset.",),
        )


def _family(value: object) -> str:
    family = str(value).strip().casefold()
    if family not in FAMILIES:
        raise ValueError(f"family must be one of: {', '.join(FAMILIES)}")
    return family


def _sha256(value: object) -> str:
    digest = str(value).strip().casefold()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("sha256 must be a 64-character hexadecimal digest")
    return digest


def _features(report: dict, family: str) -> list[float]:
    source = report.get("features")
    if not isinstance(source, dict):
        raise ValueError("features must be an object")
    expected = _FEATURES[family]
    if set(source) != set(expected):
        raise ValueError(f"features for {family} must be exactly: {', '.join(expected)}")
    values: list[float] = []
    for name in expected:
        value = source[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"feature {name} must be numeric")
        number = float(value)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"feature {name} must be between 0 and 1")
        values.append(number)
    return values


def _checks(report: dict, family: str) -> tuple[list[str], list[str]]:
    source = report.get("checks")
    if not isinstance(source, dict):
        source = {}
    required = (*_COMMON_CHECKS, *_FAMILY_CHECKS[family])
    missing = [name for name in required if name not in source]
    invalid = [name for name in required if name in source and not isinstance(source[name], bool)]
    if invalid:
        raise ValueError(f"checks must be boolean: {', '.join(invalid)}")
    failed = [name for name in required if source.get(name) is False]
    return failed, missing


def _validate(report: object) -> tuple[dict, str, list[float], list[str], list[str]]:
    if not isinstance(report, dict):
        raise ValueError("asset report must be an object")
    family = _family(report.get("family"))
    digest = _sha256(report.get("sha256"))
    size = report.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("size_bytes must be a positive integer")
    values = _features(report, family)
    failed, missing = _checks(report, family)
    clean = dict(report)
    clean["family"] = family
    clean["sha256"] = digest
    return clean, family, values, failed, missing


def _similarity(left: list[float], right: list[float]) -> float:
    distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))
    return max(0.0, 1.0 - distance / math.sqrt(len(left)))


def _path(root: str | Path) -> Path:
    return Path(root).resolve() / "asset-quality.jsonl"


def _load(root: str | Path) -> list[dict]:
    lines = read_tail(_path(root), MAX_ENTRIES)
    records = []
    for line in lines[-MAX_ENTRIES:]:
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            if not isinstance(item, dict):
                continue
            family = _family(item.get("family"))
            if item.get("schema") != SCHEMA or item.get("verified") is not True:
                continue
            if item.get("verdict") not in VERDICTS or item.get("raw_asset_path_stored") is not False:
                continue
            values = item.get("features")
            if not isinstance(values, list) or len(values) != len(_FEATURES[family]):
                continue
            if any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) or not 0.0 <= v <= 1.0 for v in values):
                continue
            if item.get("feature_names") != list(_FEATURES[family]):
                continue
            size = item.get("size_bytes")
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                continue
            if not isinstance(item.get("id"), str) or not item["id"]:
                continue
            if item.get("quality_score") != _VERDICT_SCORE[item["verdict"]]:
                continue
            verifier = item.get("verified_by")
            if not isinstance(verifier, str) or not verifier:
                continue
            family_name = verifier.casefold().split(":", 1)[0].replace("-", "_")
            if family_name not in _VERIFIER_FAMILIES:
                continue
            _sha256(item.get("sha256"))
            records.append(item)
        except (TypeError, ValueError):
            continue
    return records


def _support(records: Iterable[dict]) -> list[dict]:
    latest: dict[tuple[str, str], dict] = {}
    for item in records:
        latest[(item["family"], item["sha256"])] = item
    return list(latest.values())


def record_verified(
    report: dict,
    *,
    verdict: str,
    verified_by: str,
    memory_root: str | Path,
    evidence_refs: Iterable[str] = (),
) -> dict:
    clean, family, values, failed, missing = _validate(report)
    verdict = str(verdict).strip().upper().replace("-", "_")
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of: {', '.join(VERDICTS)}")
    verifier = " ".join(str(verified_by).split())
    verifier_family = verifier.casefold().split(":", 1)[0].replace("-", "_")
    if not verifier or verifier_family not in _VERIFIER_FAMILIES:
        raise ValueError("verified_by must identify an external verifier")
    if failed and verdict != "FAIL":
        raise ValueError("a failed deterministic check can only be recorded as FAIL")
    if missing:
        raise ValueError("all deterministic checks are required before recording")
    refs = list(evidence_refs)
    if isinstance(evidence_refs, (str, bytes)) or len(refs) > 20:
        raise ValueError("evidence_refs must be a list of at most 20 strings")
    if any(not isinstance(ref, str) or not ref.strip() or len(ref) > 256 for ref in refs):
        raise ValueError("evidence references must be non-empty strings of at most 256 characters")
    record = {
        "schema": SCHEMA,
        "id": f"aq_{uuid.uuid4().hex[:12]}",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "verified": True,
        "verified_by": verifier,
        "family": family,
        "sha256": clean["sha256"],
        "size_bytes": clean["size_bytes"],
        "feature_names": list(_FEATURES[family]),
        "features": values,
        "verdict": verdict,
        "quality_score": _VERDICT_SCORE[verdict],
        "evidence_refs": [str(ref).strip() for ref in refs],
        "raw_asset_path_stored": False,
    }
    append_record(_path(memory_root), record)
    return record


def evaluate_asset(
    report: dict,
    *,
    memory_root: str | Path,
    k: int = 5,
    min_similarity: float = MIN_SIMILARITY,
) -> AssetAdvice:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    if isinstance(min_similarity, bool) or not isinstance(min_similarity, (int, float)) or not math.isfinite(min_similarity) or not 0 <= min_similarity <= 1:
        raise ValueError("min_similarity must be finite and between 0 and 1")
    _, family, values, failed, missing = _validate(report)
    if failed:
        return AssetAdvice("rule_fail", "FAIL", "A mandatory deterministic check failed.",
                           family, 1.0, 0, [], failed, missing)
    if missing:
        return AssetAdvice("incomplete", "UNCERTAIN", "Mandatory checks are missing.",
                           family, 0.0, 0, [], failed, missing)
    family_records = [item for item in _support(_load(memory_root)) if item["family"] == family]
    scored = sorted(
        ((_similarity(values, item["features"]), item) for item in family_records),
        key=lambda pair: pair[0], reverse=True,
    )
    nearest = [pair for pair in scored if pair[0] >= float(min_similarity)][:k]
    neighbors = [{"id": item["id"], "similarity": round(sim, 4),
                  "verdict": item["verdict"], "verified_by": item["verified_by"]}
                 for sim, item in nearest]
    if len(nearest) < MIN_NEIGHBORS:
        return AssetAdvice("abstain", "UNCERTAIN",
                           "Too few similar verified assets in this family.", family,
                           0.0, len(nearest), neighbors, [], [])
    weight = sum(max(sim, 0.001) for sim, _ in nearest)
    score = sum(max(sim, 0.001) * item["quality_score"] for sim, item in nearest) / weight
    if score < 0.25:
        verdict = "FAIL"
    elif score < 0.60:
        verdict = "UNCERTAIN"
    elif score < 0.90:
        verdict = "PASS"
    else:
        verdict = "PASS_ROBUST"
    strength = min(0.95, (sum(sim for sim, _ in nearest) / len(nearest)) * len(nearest) / k)
    return AssetAdvice("suggest", verdict,
                       f"Weighted verdict from {len(nearest)} verified {family} neighbor(s).",
                       family, round(strength, 4), len(nearest), neighbors, [], [])


def quality_status(memory_root: str | Path) -> dict:
    recorded = _load(memory_root)
    records = _support(recorded)
    counts = Counter(item["family"] for item in records)
    return {
        "schema": "bellium.asset-quality-status/v1",
        "mode": "shadow",
        "recorded_outcomes": len(recorded),
        "verified_assets": len(records),
        "by_family": {family: counts.get(family, 0) for family in FAMILIES},
        "families_ready": [family for family in FAMILIES if counts.get(family, 0) >= MIN_NEIGHBORS],
        "activation_allowed": False,
    }
