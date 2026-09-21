"""Pronunciation similarity: DTW on phoneme features, then optional word k-NN."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.resources import model_path
from bellium.knn.phoneme import _similarity, load_inventory
from bellium.language.evidence import can_certify, parse_evidence

SPECIALIST_ID = "bellium/knn/pronunciation-similarity:v0"
SCHEMA = "bellium.pronunciation-memory/v1"
DEFAULT_MEMORY = (
    model_path()
    / "knn"
    / "language"
    / "pronunciation-memory-v0.json"
)


def _dtw(left: list[dict[str, float]], right: list[dict[str, float]]) -> tuple[float, list[tuple[int, int]]]:
    if not left or not right:
        raise ValueError("both pronunciations need at least one phone")
    rows, cols = len(left), len(right)
    table = [[float("inf")] * (cols + 1) for _ in range(rows + 1)]
    table[0][0] = 0.0
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            sim = _similarity(left[i - 1], right[j - 1])
            cost = 1.0 if sim is None else 1.0 - sim
            table[i][j] = cost + min(table[i - 1][j], table[i][j - 1], table[i - 1][j - 1])
    path: list[tuple[int, int]] = []
    i, j = rows, cols
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        options = (
            (table[i - 1][j - 1], i - 1, j - 1),
            (table[i - 1][j], i - 1, j),
            (table[i][j - 1], i, j - 1),
        )
        _, i, j = min(options, key=lambda item: item[0])
    path.reverse()
    norm = table[rows][cols] / max(rows, cols)
    return norm, path


def _lookup_phone(ipa: str, language_id: str, inventory: list[dict[str, Any]]) -> dict[str, float]:
    matches = [
        item for item in inventory
        if item["language_id"] == language_id and item["ipa"] == ipa
    ]
    if not matches:
        raise ValueError(f"unknown phone '{ipa}' in language {language_id}")
    return matches[0]["features"]


def _sequence_edits(left: list[str], right: list[str]) -> list[dict[str, str | None]]:
    """Symbol edits are distinct from DTW's repeat-tolerant acoustic similarity."""
    table = [list(range(len(right) + 1))]
    for i, phone in enumerate(left, 1):
        row = [i]
        for j, reference in enumerate(right, 1):
            row.append(min(row[-1] + 1, table[-1][j] + 1,
                           table[-1][j - 1] + (phone != reference)))
        table.append(row)
    edits = []
    i, j = len(left), len(right)
    while i or j:
        if i and j and table[i][j] == table[i - 1][j - 1] + (left[i - 1] != right[j - 1]):
            if left[i - 1] != right[j - 1]:
                edits.append({"learner": left[i - 1], "reference": right[j - 1]})
            i, j = i - 1, j - 1
        elif i and table[i][j] == table[i - 1][j] + 1:
            edits.append({"learner": left[i - 1], "reference": None})
            i -= 1
        else:
            edits.append({"learner": None, "reference": right[j - 1]})
            j -= 1
    return list(reversed(edits))


def _sequence(
    phones: object,
    language_id: str,
    inventory: list[dict[str, Any]],
) -> tuple[list[str], list[dict[str, float]]]:
    if not isinstance(phones, list) or not phones:
        raise ValueError("ipa sequence must be a non-empty list")
    symbols = [str(phone).strip() for phone in phones]
    if any(not symbol for symbol in symbols):
        raise ValueError("ipa sequence contains an empty phone")
    return symbols, [_lookup_phone(symbol, language_id, inventory) for symbol in symbols]


def load_pronunciations(path: str | Path | None = None) -> list[dict[str, Any]]:
    payload = json.loads(Path(path or DEFAULT_MEMORY).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("pronunciation memory schema mismatch")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("pronunciation memory needs items")
    loaded = []
    for raw in items:
        if not isinstance(raw, dict):
            raise ValueError("pronunciation item must be an object")
        form = str(raw.get("form") or "").strip()
        language_id = str(raw.get("language_id") or "").strip()
        item_id = str(raw.get("id") or "").strip()
        if not form or not language_id or not item_id:
            raise ValueError("pronunciation item needs id, language_id and form")
        if raw.get("raw_audio_stored") is True:
            raise ValueError("raw audio must not be stored in pronunciation memory")
        loaded.append({
            "id": item_id,
            "language_id": language_id,
            "form": form,
            "ipa": list(raw.get("ipa") or []),
            "evidence": parse_evidence(raw.get("evidence")),
            "speaker": str(raw.get("speaker") or "unspecified"),
        })
    return loaded


def compare_pronunciation(
    query: dict[str, Any],
    *,
    inventory: list[dict[str, Any]] | None = None,
    memory: list[dict[str, Any]] | None = None,
) -> SpecialistResult:
    language_id = str(query.get("language_id") or "").strip()
    form = str(query.get("form") or "").strip()
    if not language_id:
        raise ValueError("query needs language_id")
    phones_inventory = inventory if inventory is not None else load_inventory()
    learner_ipa, learner_vec = _sequence(query.get("ipa"), language_id, phones_inventory)

    reference = query.get("reference")
    memory_items = memory if memory is not None else load_pronunciations()
    if reference is None:
        if not form:
            raise ValueError("query needs form when no explicit reference is given")
        candidates = [
            item for item in memory_items
            if item["language_id"] == language_id and item["form"] == form
        ]
        attested = [item for item in candidates if item["evidence"] == "attested"]
        pool = attested or candidates
        if not pool:
            return SpecialistResult(
                SPECIALIST_ID,
                {
                    "status": "abstain",
                    "reason": "no_reference_pronunciation",
                    "language_id": language_id,
                    "form": form,
                    "evidence": "speculative",
                    "certified": False,
                },
                0.0,
                True,
                AuthorityMode.CONSULTATIVE,
                notes=("No labelled pronunciation exists for this word form.",),
            )
        best = None
        best_score = -1.0
        for item in pool:
            _, ref_vec = _sequence(item["ipa"], language_id, phones_inventory)
            dist, _ = _dtw(learner_vec, ref_vec)
            score = max(0.0, 1.0 - dist)
            if score > best_score:
                best = item
                best_score = score
        reference_item = best
        ref_ipa = list(best["ipa"])
        evidence = parse_evidence(best["evidence"])
        score = best_score
    else:
        if not isinstance(reference, dict):
            raise ValueError("reference must be an object")
        if reference.get("language_id", language_id) != language_id:
            raise ValueError("reference language differs from query language")
        ref_ipa, ref_vec = _sequence(reference.get("ipa"), language_id, phones_inventory)
        evidence = parse_evidence(reference.get("evidence"))
        dist, _ = _dtw(learner_vec, ref_vec)
        score = max(0.0, 1.0 - dist)
        reference_item = {
            "id": str(reference.get("id") or "explicit-reference"),
            "speaker": str(reference.get("speaker") or "unspecified"),
            "evidence": evidence,
        }

    mismatches = _sequence_edits(learner_ipa, ref_ipa)
    exact_match = learner_ipa == ref_ipa
    if exact_match:
        bucket = "match"
    elif score >= 0.75:
        bucket = "close"
    else:
        bucket = "mismatch"
    certified = can_certify(evidence) and exact_match
    return SpecialistResult(
        SPECIALIST_ID,
        {
            "status": "compare",
            "language_id": language_id,
            "form": form or None,
            "similarity": round(score, 4),
            "bucket": bucket,
            "evidence": evidence,
            "certified": certified,
            "exact_match": exact_match,
            "mismatches": mismatches,
            "reference_id": reference_item.get("id"),
            "speaker": reference_item.get("speaker"),
        },
        round(min(0.95, score), 4),
        False,
        AuthorityMode.CONSULTATIVE,
        notes=(
            "A reconstructed reference can be compared, never certified.",
            "No raw audio is stored or required.",
        ),
    )
