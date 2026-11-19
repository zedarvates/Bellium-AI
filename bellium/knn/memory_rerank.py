"""Lexical k-NN memory reranker. No payload bodies, domain-isolated."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from heapq import nsmallest
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.resources import model_path
from bellium.language.evidence import parse_evidence

SPECIALIST_ID = "bellium/knn/memory-reranker:v0"
SCHEMA = "bellium.memory-rerank/v1"
MIN_NEIGHBORS = 2
MIN_SIMILARITY = 0.34
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_EVIDENCE_WEIGHT = {
    "attested": 1.0,
    "reconstructed": 0.6,
    "inferred": 0.4,
    "speculative": 0.2,
}
_LEXICAL_PROFILES = MappingProxyType({
    "exact": frozenset(),
    "fr-en": frozenset(
        "a an the this that these those of to for in on at with by and "
        "i me my you your we our he his she her it its they their "
        "is are was were be been how what which can do does "
        "le la les l un une des du de d et en pour avec sur dans au aux à "
        "je j tu nous vous il elle ils elles ce cet cette ces est sont "
        "comment quel quelle quels quelles moi me mon ma mes ton ta tes son sa ses votre vos".split()
    ),
})
DEFAULT_MEMORY = (
    model_path()
    / "knn"
    / "memory"
    / "rerank-fixtures-v0.json"
)


def tokenize(text: object) -> set[str]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("terms must be a non-empty string")
    tokens = set(_TOKEN.findall(unicodedata.normalize("NFC", text.casefold())))
    if not tokens:
        raise ValueError("terms produced no tokens")
    return tokens


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def load_memory(path: str | Path | None = None) -> list[dict[str, Any]]:
    payload = json.loads(Path(path or DEFAULT_MEMORY).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("memory rerank schema mismatch")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("memory needs items")
    return _validated_memory(items)


def _validated_memory(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Apply the same contract to files and caller-supplied exemplar lists."""
    if not isinstance(items, list):
        raise ValueError("memory must be a list")
    loaded = []
    seen = set()
    for raw in items:
        if not isinstance(raw, dict):
            raise ValueError("memory item must be an object")
        if "body" in raw or "content" in raw or "prompt" in raw:
            raise ValueError("memory items cannot store body/content/prompt")
        item_id = str(raw.get("id") or "").strip()
        domain = str(raw.get("domain") or "").strip()
        title = str(raw.get("title") or "").strip()
        if not item_id or not domain or not title:
            raise ValueError("memory item needs id, domain and title")
        if item_id in seen:
            raise ValueError(f"duplicate memory id: {item_id}")
        seen.add(item_id)
        raw_terms = raw.get("terms")
        if isinstance(raw_terms, str):
            terms = tokenize(raw_terms)
        elif isinstance(raw_terms, (set, frozenset)) and raw_terms:
            terms = set()
            for token in raw_terms:
                parsed = tokenize(token)
                if len(parsed) != 1:
                    raise ValueError("memory terms must contain individual tokens")
                terms.update(parsed)
        else:
            raise ValueError("memory terms must be a non-empty string or token set")
        loaded.append({
            "id": item_id,
            "domain": domain,
            "title": title,
            "terms": terms,
            "evidence": parse_evidence(raw.get("evidence")),
        })
    return loaded


def _validate_profile(profile: str) -> frozenset[str]:
    if not isinstance(profile, str) or profile not in _LEXICAL_PROFILES:
        raise ValueError("lexical profile must be exact or fr-en")
    return _LEXICAL_PROFILES[profile]


def _profiled_records(records, profile):
    stopwords = _validate_profile(profile)
    if not stopwords:
        return records
    filtered = []
    for item in records:
        terms = item["terms"] - stopwords
        if not terms:
            raise ValueError(f"memory item {item['id']} has no content terms under {profile}")
        filtered.append({**item, "terms": terms})
    return filtered


@dataclass(frozen=True, slots=True)
class _DomainIndex:
    records: tuple[Mapping[str, Any], ...]
    postings: Mapping[str, tuple[int, ...]]


@dataclass(frozen=True, slots=True, init=False)
class PreparedMemory:
    """Validated immutable snapshot for repeated queries.

    Construction copies input records and terms. Edits to the source list or
    file do not update this snapshot: build a new instance when memory changes.
    No global cache or file watching is involved.
    """

    _domains: Mapping[str, _DomainIndex]
    _size: int
    profile: str

    def __init__(self, memory: list[dict[str, Any]] | None = None, *, profile: str = "exact"):
        _validate_profile(profile)
        records = _validated_memory(memory) if memory is not None else load_memory()
        records = _profiled_records(records, profile)
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        tokens: dict[str, str] = {}
        for item in records:
            frozen = MappingProxyType({
                **item,
                "terms": frozenset(tokens.setdefault(term, term) for term in item["terms"]),
            })
            grouped.setdefault(item["domain"], []).append(frozen)
        domains = {}
        for domain, items in grouped.items():
            postings: dict[str, list[int]] = {}
            for index, item in enumerate(items):
                for term in item["terms"]:
                    postings.setdefault(term, []).append(index)
            domains[domain] = _DomainIndex(
                tuple(items), MappingProxyType({term: tuple(indices)
                                               for term, indices in postings.items()}))
        object.__setattr__(self, "_domains", MappingProxyType(domains))
        object.__setattr__(self, "_size", len(records))
        object.__setattr__(self, "profile", profile)

    def __len__(self) -> int:
        return self._size

    def _matching(self, domain: str, terms: set[str]) -> Iterator[Mapping[str, Any]]:
        index = self._domains.get(domain)
        if index is None:
            return
        matched = set()
        for term in terms:
            matched.update(index.postings.get(term, ()))
        for position in matched:
            yield index.records[position]


def rerank_memory(
    query: dict[str, Any],
    *,
    memory: list[dict[str, Any]] | PreparedMemory | None = None,
    k: int = 5,
    profile: str | None = None,
) -> SpecialistResult:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    if not isinstance(query, dict):
        raise ValueError("query must be an object")
    domain = str(query.get("domain") or "").strip()
    if not domain:
        raise ValueError("query needs domain")
    terms = tokenize(query.get("terms"))
    if isinstance(memory, PreparedMemory):
        if profile is not None and profile != memory.profile:
            raise ValueError("query lexical profile does not match prepared memory")
        effective_profile = memory.profile
        terms -= _validate_profile(effective_profile)
        records = memory._matching(domain, terms)
    else:
        effective_profile = "exact" if profile is None else profile
        terms -= _validate_profile(effective_profile)
        records = _validated_memory(memory) if memory is not None else load_memory()
        records = _profiled_records(records, effective_profile)
    metadata = {"lexical_profile": effective_profile} if effective_profile != "exact" else {}
    if not terms:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "no_content_terms", "neighbors": [],
             "domain": domain, **metadata},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("Lexical filtering left no content terms; no memory is proposed.",),
        )

    def candidates():
        for item in records:
            if item["domain"] != domain:
                continue
            sim = _jaccard(terms, item["terms"])
            if sim >= MIN_SIMILARITY:
                yield sim * _EVIDENCE_WEIGHT[item["evidence"]], sim, item

    # Keep only k candidates during selection; ties do not depend on file order.
    nearest = nsmallest(k, candidates(), key=lambda pair: (-pair[0], -pair[1], pair[2]["id"]))
    neighbors = [
        {
            "id": item["id"],
            "title": item["title"],
            "similarity": round(sim, 4),
            "score": round(score, 4),
            "evidence": item["evidence"],
        }
        for score, sim, item in nearest
    ]
    if len(nearest) < MIN_NEIGHBORS:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "too_few_similar_memories", "neighbors": neighbors,
             "domain": domain, **metadata},
            0.0,
            True,
            AuthorityMode.CONSULTATIVE,
            notes=("Reranker never invents a memory hit.",),
        )
    strength = min(0.95, nearest[0][0])
    return SpecialistResult(
        SPECIALIST_ID,
        {
            "status": "suggest",
            "domain": domain,
            "neighbors": neighbors,
            "top_id": neighbors[0]["id"],
            **metadata,
        },
        round(strength, 4),
        False,
        AuthorityMode.CONSULTATIVE,
        notes=("Titles and ids only. Bodies are not stored or returned.",),
    )

