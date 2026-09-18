"""Language-isolated grapheme-phoneme k-NN. Exact attested mappings first."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.resources import model_path
from bellium.knn.phoneme import load_inventory
from bellium.language.evidence import can_certify, parse_evidence, weakest_evidence

SPECIALIST_ID = "bellium/knn/grapheme-phoneme:v0"
SCHEMA = "bellium.grapheme-phoneme/v1"
MIN_NEIGHBORS = 3
MIN_SIMILARITY = 0.60
DEFAULT_MEMORY = (
    model_path()
    / "knn"
    / "language"
    / "grapheme-phoneme-v0.json"
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


def _form_tokens(value: object) -> list[str]:
    text = str(value or "").strip().casefold()
    if not text or any(ch.isspace() for ch in text):
        raise ValueError("form must be a single word")
    if not all(ch.isalpha() for ch in text):
        raise ValueError("form may contain letters only")
    return list(text)


def _ipa_tokens(value: object) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError("ipa must be a non-empty list")
    tokens = [str(item).strip() for item in value]
    if any(not item for item in tokens):
        raise ValueError("ipa contains an empty phone")
    return tokens


def load_mappings(
    path: str | Path | None = None,
    *,
    inventory: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    payload = json.loads(Path(path or DEFAULT_MEMORY).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("grapheme-phoneme schema mismatch")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("mapping memory needs items")
    allowed = {
        (item["language_id"], item["ipa"])
        for item in (inventory if inventory is not None else load_inventory())
    }
    loaded = []
    seen = set()
    for raw in items:
        if not isinstance(raw, dict):
            raise ValueError("mapping item must be an object")
        if raw.get("raw_audio_stored") is True:
            raise ValueError("raw audio must not be stored")
        item_id = str(raw.get("id") or "").strip()
        language_id = str(raw.get("language_id") or "").strip()
        source = str(raw.get("source") or "").strip()
        form = "".join(_form_tokens(raw.get("form")))
        ipa = _ipa_tokens(raw.get("ipa"))
        if not item_id or not language_id or not source:
            raise ValueError("mapping needs id, language_id and source")
        if (item_id, language_id, form) in seen:
            raise ValueError(f"duplicate mapping: {form}")
        seen.add((item_id, language_id, form))
        unknown = [phone for phone in ipa if (language_id, phone) not in allowed]
        if unknown:
            raise ValueError(f"unknown phone in mapping {form}: {', '.join(unknown)}")
        loaded.append({
            "id": item_id,
            "language_id": language_id,
            "form": form,
            "ipa": ipa,
            "evidence": parse_evidence(raw.get("evidence")),
            "source": source,
        })
    return loaded


def _result(output: dict[str, Any], confidence: float, abstained: bool) -> SpecialistResult:
    return SpecialistResult(
        SPECIALIST_ID,
        output,
        round(confidence, 4),
        abstained,
        AuthorityMode.CONSULTATIVE,
        notes=(
            "Exact attested mappings can be certified.",
            "Neighbor-only mappings stay uncertified.",
        ),
    )


def map_form(
    query: dict[str, Any],
    *,
    memory: list[dict[str, Any]] | None = None,
    inventory: list[dict[str, Any]] | None = None,
    k: int = 5,
) -> SpecialistResult:
    language_id = str(query.get("language_id") or "").strip()
    direction = str(query.get("direction") or "g2p").strip().casefold()
    if not language_id:
        raise ValueError("query needs language_id")
    if direction not in {"g2p", "p2g"}:
        raise ValueError("direction must be g2p or p2g")
    records = [
        item for item in (memory if memory is not None else load_mappings(inventory=inventory))
        if item["language_id"] == language_id
    ]
    if direction == "g2p":
        form = "".join(_form_tokens(query.get("form")))
        exact = [item for item in records if item["form"] == form]
        if len(exact) == 1:
            item = exact[0]
            return _result({
                "status": "exact",
                "direction": direction,
                "language_id": language_id,
                "form": form,
                "ipa": list(item["ipa"]),
                "evidence": item["evidence"],
                "certified": can_certify(item["evidence"]),
                "source": item["source"],
                "neighbors": [],
            }, 0.95 if can_certify(item["evidence"]) else 0.7, False)
        if len(exact) > 1:
            return _result({
                "status": "abstain",
                "reason": "conflicting_exact_mappings",
                "language_id": language_id,
                "form": form,
                "certified": False,
                "evidence": "speculative",
            }, 0.0, True)
        scored = sorted(
            ((_similarity(list(form), list(item["form"])), item) for item in records),
            key=lambda pair: pair[0],
            reverse=True,
        )
        nearest = [pair for pair in scored if pair[0] >= MIN_SIMILARITY][:k]
        return _fuzzy(direction, language_id, nearest, form=form)

    ipa = _ipa_tokens(query.get("ipa"))
    exact = [item for item in records if item["ipa"] == ipa]
    if len(exact) == 1:
        item = exact[0]
        return _result({
            "status": "exact",
            "direction": direction,
            "language_id": language_id,
            "form": item["form"],
            "ipa": list(ipa),
            "evidence": item["evidence"],
            "certified": can_certify(item["evidence"]),
            "source": item["source"],
            "neighbors": [],
        }, 0.95 if can_certify(item["evidence"]) else 0.7, False)
    if len(exact) > 1:
        forms = sorted({item["form"] for item in exact})
        evidence = weakest_evidence([item["evidence"] for item in exact])
        return _result({
            "status": "ambiguous",
            "direction": direction,
            "language_id": language_id,
            "ipa": list(ipa),
            "forms": forms,
            "evidence": evidence,
            "certified": False,
            "neighbors": [{"id": item["id"], "form": item["form"], "evidence": item["evidence"]} for item in exact],
        }, 0.6, False)
    scored = sorted(
        ((_similarity(ipa, item["ipa"]), item) for item in records),
        key=lambda pair: pair[0],
        reverse=True,
    )
    nearest = [pair for pair in scored if pair[0] >= MIN_SIMILARITY][:k]
    return _fuzzy(direction, language_id, nearest, ipa=ipa)


def _fuzzy(
    direction: str,
    language_id: str,
    nearest: list[tuple[float, dict[str, Any]]],
    *,
    form: str | None = None,
    ipa: list[str] | None = None,
) -> SpecialistResult:
    neighbors = [
        {
            "id": item["id"],
            "form": item["form"],
            "ipa": list(item["ipa"]),
            "similarity": round(sim, 4),
            "evidence": item["evidence"],
        }
        for sim, item in nearest
    ]
    if len(nearest) < MIN_NEIGHBORS:
        return _result({
            "status": "abstain",
            "reason": "too_few_similar_mappings",
            "direction": direction,
            "language_id": language_id,
            "form": form,
            "ipa": ipa,
            "certified": False,
            "evidence": "speculative",
            "neighbors": neighbors,
        }, 0.0, True)
    if direction == "g2p":
        values = {tuple(item["ipa"]) for _, item in nearest}
        if len(values) > 1:
            return _result({
                "status": "abstain",
                "reason": "mixed_neighbor_ipa",
                "direction": direction,
                "language_id": language_id,
                "form": form,
                "certified": False,
                "evidence": "speculative",
                "neighbors": neighbors,
            }, 0.0, True)
        proposal = list(nearest[0][1]["ipa"])
        field = "ipa"
    else:
        values = {item["form"] for _, item in nearest}
        if len(values) > 1:
            return _result({
                "status": "abstain",
                "reason": "mixed_neighbor_forms",
                "direction": direction,
                "language_id": language_id,
                "ipa": ipa,
                "certified": False,
                "evidence": "speculative",
                "neighbors": neighbors,
            }, 0.0, True)
        proposal = nearest[0][1]["form"]
        field = "form"
    evidence = weakest_evidence([item["evidence"] for _, item in nearest])
    strength = min(0.8, sum(sim for sim, _ in nearest) / len(nearest))
    return _result({
        "status": "suggest",
        "direction": direction,
        "language_id": language_id,
        "form": form if direction == "g2p" else proposal,
        "ipa": proposal if direction == "g2p" else ipa,
        field: proposal,
        "evidence": "inferred" if evidence == "attested" else evidence,
        "certified": False,
        "neighbors": neighbors,
    }, strength, False)
