"""Lexical k-NN memory reranker. No payload bodies, domain-isolated."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.resources import model_path
from bellium.language.evidence import parse_evidence

SPECIALIST_ID = "bellium/knn/memory-reranker:v0"
SCHEMA = "bellium.memory-rerank/v1"
MIN_NEIGHBORS = 2
MIN_SIMILARITY = 0.34
_TOKEN = re.compile(r"[a-z0-9]+")
_EVIDENCE_WEIGHT = {
    "attested": 1.0,
    "reconstructed": 0.6,
    "inferred": 0.4,
    "speculative": 0.2,
}
DEFAULT_MEMORY = (
    model_path()
    / "knn"
    / "memory"
    / "rerank-fixtures-v0.json"
)


def tokenize(text: object) -> set[str]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("terms must be a non-empty string")
    tokens = set(_TOKEN.findall(text.casefold()))
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
        loaded.append({
            "id": item_id,
            "domain": domain,
            "title": title,
            "terms": tokenize(raw.get("terms")),
            "evidence": parse_evidence(raw.get("evidence")),
        })
    return loaded


def rerank_memory(
    query: dict[str, Any],
    *,
    memory: list[dict[str, Any]] | None = None,
    k: int = 5,
) -> SpecialistResult:
    domain = str(query.get("domain") or "").strip()
    if not domain:
        raise ValueError("query needs domain")
    terms = tokenize(query.get("terms"))
    records = [item for item in (memory if memory is not None else load_memory()) if item["domain"] == domain]
    scored = []
    for item in records:
        sim = _jaccard(terms, item["terms"])
        if sim < MIN_SIMILARITY:
            continue
        score = sim * _EVIDENCE_WEIGHT[item["evidence"]]
        scored.append((score, sim, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    nearest = scored[:k]
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
             "domain": domain},
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
        },
        round(strength, 4),
        False,
        AuthorityMode.CONSULTATIVE,
        notes=("Titles and ids only. Bodies are not stored or returned.",),
    )

