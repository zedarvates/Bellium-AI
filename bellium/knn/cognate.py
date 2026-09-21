"""Family-aware cognate retrieval. Labelled sets first; similarity never invents a proto-form."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.resources import model_path
from bellium.language.evidence import parse_evidence, weakest_evidence

SPECIALIST_ID = "bellium/knn/cognate-retrieval:v0"
SCHEMA = "bellium.cognate-memory/v1"
MIN_NEIGHBORS = 3
MIN_SIMILARITY = 0.55
DEFAULT_MEMORY = (
    model_path()
    / "knn"
    / "language"
    / "cognates-v0.json"
)


def _levenshtein(left: list[str], right: list[str]) -> int:
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (0 if a == b else 1)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def _similarity(left: list[str], right: list[str]) -> float:
    span = max(len(left), len(right), 1)
    return max(0.0, 1.0 - _levenshtein(left, right) / span)


def _form(value: object) -> str:
    text = str(value or "").strip().casefold()
    if not text or any(ch.isspace() for ch in text) or not all(ch.isalpha() for ch in text):
        raise ValueError("form must be a single alphabetic word")
    return text


def _ipa(value: object) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError("ipa must be a non-empty list")
    tokens = [str(item).strip() for item in value]
    if any(not item for item in tokens):
        raise ValueError("ipa contains an empty phone")
    return tokens


def _pair_similarity(form: str, ipa: list[str] | None, item: dict[str, Any]) -> float:
    grapheme = _similarity(list(form), list(item["form"]))
    if ipa is None:
        return grapheme
    phone = _similarity(ipa, item["ipa"])
    return (grapheme + phone) / 2.0


def load_cognates(path: str | Path | None = None) -> list[dict[str, Any]]:
    payload = json.loads(Path(path or DEFAULT_MEMORY).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("cognate memory schema mismatch")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("cognate memory needs items")
    loaded = []
    seen = set()
    for raw in items:
        if not isinstance(raw, dict):
            raise ValueError("cognate item must be an object")
        if raw.get("raw_audio_stored") is True:
            raise ValueError("raw audio must not be stored")
        item_id = str(raw.get("id") or "").strip()
        family_id = str(raw.get("family_id") or "").strip()
        language_id = str(raw.get("language_id") or "").strip()
        source = str(raw.get("source") or "").strip()
        gloss = str(raw.get("gloss") or "").strip().casefold()
        form = _form(raw.get("form"))
        if not item_id or not family_id or not language_id or not source or not gloss:
            raise ValueError("cognate item needs id, family_id, language_id, gloss and source")
        key = (family_id, language_id, form)
        if key in seen:
            raise ValueError(f"duplicate cognate form: {form}")
        seen.add(key)
        set_id = raw.get("cognate_set")
        if set_id is not None:
            set_id = str(set_id).strip()
            if not set_id:
                raise ValueError("cognate_set must be non-empty when present")
        loaded.append({
            "id": item_id,
            "family_id": family_id,
            "language_id": language_id,
            "form": form,
            "ipa": _ipa(raw.get("ipa")),
            "gloss": gloss,
            "evidence": parse_evidence(raw.get("evidence")),
            "source": source,
            "cognate_set": set_id,
        })
    return loaded


def retrieve_cognates(
    query: dict[str, Any],
    *,
    memory: list[dict[str, Any]] | None = None,
    k: int = 5,
) -> SpecialistResult:
    family_id = str(query.get("family_id") or "").strip()
    language_id = str(query.get("language_id") or "").strip()
    if not family_id or not language_id:
        raise ValueError("query needs family_id and language_id")
    form = _form(query.get("form"))
    ipa = _ipa(query["ipa"]) if query.get("ipa") is not None else None
    records = memory if memory is not None else load_cognates()
    family = [item for item in records if item["family_id"] == family_id]
    others = [item for item in family if item["language_id"] != language_id]
    exact = [
        item for item in family
        if item["language_id"] == language_id and item["form"] == form
    ]
    if len(exact) > 1:
        return _abstain("conflicting_query_forms", family_id, language_id, form)
    query_item = exact[0] if exact else None
    if query_item and query_item.get("cognate_set"):
        members = [
            item for item in others
            if item.get("cognate_set") == query_item["cognate_set"]
        ]
        if members:
            evidence = weakest_evidence(
                [query_item["evidence"], *[item["evidence"] for item in members]]
            )
            glosses = {query_item["gloss"], *[item["gloss"] for item in members]}
            if len(glosses) > 1:
                return _abstain("mixed_gloss_in_labelled_set", family_id, language_id, form)
            return SpecialistResult(
                SPECIALIST_ID,
                {
                    "status": "linked",
                    "family_id": family_id,
                    "language_id": language_id,
                    "form": form,
                    "cognate_set": query_item["cognate_set"],
                    "gloss": query_item["gloss"],
                    "evidence": evidence,
                    "certified": False,
                    "link_attested": evidence == "attested",
                    "neighbors": [_neighbor(1.0, item) for item in members],
                    "proto_form": None,
                },
                0.9 if evidence == "attested" else 0.65,
                False,
                AuthorityMode.CONSULTATIVE,
                notes=(
                    "Labelled cognate sets are not proto-reconstructions.",
                    "certified stays false: relatedness is not a reconstructed etymon.",
                ),
            )
    scored = sorted(
        ((_pair_similarity(form, ipa, item), item) for item in others),
        key=lambda pair: pair[0],
        reverse=True,
    )
    nearest = [pair for pair in scored if pair[0] >= MIN_SIMILARITY][:k]
    neighbors = [_neighbor(sim, item) for sim, item in nearest]
    if len(nearest) < MIN_NEIGHBORS:
        return _abstain("too_few_cross_lect_neighbors", family_id, language_id, form, neighbors)
    glosses = {item["gloss"] for _, item in nearest}
    if len(glosses) > 1:
        return _abstain("mixed_neighbor_gloss", family_id, language_id, form, neighbors)
    evidence = weakest_evidence([item["evidence"] for _, item in nearest])
    if query_item:
        evidence = weakest_evidence([query_item["evidence"], evidence])
    if evidence == "attested":
        evidence = "inferred"
    strength = min(0.8, sum(sim for sim, _ in nearest) / len(nearest))
    return SpecialistResult(
        SPECIALIST_ID,
        {
            "status": "suggest",
            "family_id": family_id,
            "language_id": language_id,
            "form": form,
            "gloss": next(iter(glosses)),
            "evidence": evidence,
            "certified": False,
            "link_attested": False,
            "neighbors": neighbors,
            "proto_form": None,
        },
        round(strength, 4),
        False,
        AuthorityMode.CONSULTATIVE,
        notes=(
            "Similarity is not a reconstructed proto-form.",
            "Same-language lookalikes are not treated as cognates.",
        ),
    )


def _neighbor(sim: float, item: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": item["id"],
        "language_id": item["language_id"],
        "form": item["form"],
        "ipa": list(item["ipa"]),
        "gloss": item["gloss"],
        "evidence": item["evidence"],
        "similarity": round(sim, 4),
    }


def _abstain(
    reason: str,
    family_id: str,
    language_id: str,
    form: str,
    neighbors: list[dict[str, Any]] | None = None,
) -> SpecialistResult:
    return SpecialistResult(
        SPECIALIST_ID,
        {
            "status": "abstain",
            "reason": reason,
            "family_id": family_id,
            "language_id": language_id,
            "form": form,
            "certified": False,
            "proto_form": None,
            "neighbors": neighbors or [],
            "evidence": "speculative",
        },
        0.0,
        True,
        AuthorityMode.CONSULTATIVE,
        notes=("Cognate retrieval abstains instead of inventing a proto-form.",),
    )

