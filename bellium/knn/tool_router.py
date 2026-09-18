"""Exemplar k-NN for consultative tool routing. Closed catalog, no raw prompts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.resources import model_path


SPECIALIST_ID = "bellium/knn/tool-router:v0"
SCHEMA = "bellium.tool-router-memory/v1"
CATALOG = ("none", "inspect_files", "run_tests", "git_status", "escalate")
FEATURE_NAMES = (
    "has_code",
    "has_files",
    "has_error",
    "wants_mutation",
    "wants_external_state",
    "criticality",
)
MIN_NEIGHBORS = 3
MIN_SIMILARITY = 0.62
DEFAULT_MEMORY = (
    model_path()
    / "knn"
    / "tools"
    / "exemplars-v0.json"
)


def _bool_signal(source: dict, name: str) -> bool:
    value = source.get(name, False)
    if not isinstance(value, bool):
        raise ValueError(f"signal {name} must be boolean")
    return bool(value)


def _features(query: dict[str, Any]) -> dict[str, float]:
    source = query.get("features")
    if not isinstance(source, dict):
        raise ValueError("query needs a features object")
    values = {}
    for name in FEATURE_NAMES:
        if name not in source:
            raise ValueError(f"missing feature {name}")
        raw = source[name]
        if raw is None:
            raise ValueError(f"feature {name} is unknown; do not coerce to 0")
        if isinstance(raw, bool) and name != "criticality":
            values[name] = 1.0 if raw else 0.0
            continue
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"feature {name} must be numeric")
        number = float(raw)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"feature {name} must be between 0 and 1")
        values[name] = number
    return values


def deterministic_veto(query: dict[str, Any]) -> str | None:
    if "text" in query or "prompt" in query:
        raise ValueError("pass boolean signals, not raw text")
    signals = query.get("signals", {})
    if not isinstance(signals, dict):
        raise ValueError("signals must be an object")
    if _bool_signal(signals, "mentions_secret") or _bool_signal(signals, "mentions_deploy") or _bool_signal(signals, "mentions_destruction"):
        return "sensitive_or_destructive_task"
    features = _features(query)
    if features["wants_mutation"] >= 0.5:
        return "mutation_is_outside_the_closed_catalog"
    if features["criticality"] >= 0.85:
        return "critical_task_requires_escalation"
    requested = query.get("requested_tool")
    if requested is None or requested == "":
        return None
    tool = str(requested).strip()
    if tool not in CATALOG:
        return "unknown_requested_tool"
    if tool == "escalate":
        return "caller_requested_escalation"
    return None


def load_exemplars(path: str | Path | None = None) -> list[dict[str, Any]]:
    payload = json.loads(Path(path or DEFAULT_MEMORY).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("tool router memory schema mismatch")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("tool router memory needs items")
    return _validated_exemplars(items)


def _validated_exemplars(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    loaded = []
    seen = {}
    for raw in items:
        if not isinstance(raw, dict):
            raise ValueError("exemplar must be an object")
        if raw.get("raw_prompt_stored") is True:
            raise ValueError("raw prompts must not be stored")
        tool = str(raw.get("tool") or "").strip()
        if tool not in CATALOG:
            raise ValueError(f"exemplar tool not in catalog: {tool}")
        item = {
            "id": str(raw.get("id") or "").strip(),
            "tool": tool,
            "features": _features({"features": raw.get("features")}),
            "verified_by": str(raw.get("verified_by") or "").strip(),
        }
        if not item["id"] or not item["verified_by"]:
            raise ValueError("exemplar needs id and verified_by")
        if item["id"] in seen:
            if seen[item["id"]] != item:
                raise ValueError("conflicting duplicate exemplar id")
            continue
        seen[item["id"]] = item
        loaded.append(item)
    return loaded


def _similarity(left: dict[str, float], right: dict[str, float]) -> float:
    vec_l = [left[name] for name in FEATURE_NAMES]
    vec_r = [right[name] for name in FEATURE_NAMES]
    distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(vec_l, vec_r)))
    return max(0.0, 1.0 - distance / math.sqrt(len(FEATURE_NAMES)))


def retrieve_tool(
    query: dict[str, Any],
    *,
    memory: list[dict[str, Any]] | None = None,
    k: int = 5,
) -> SpecialistResult:
    veto = deterministic_veto(query)
    if veto:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "rule_escalate", "tool": "escalate", "reason": veto, "neighbors": []},
            1.0,
            False,
            AuthorityMode.CONSULTATIVE,
            notes=("Deterministic veto wins before k-NN.",),
        )
    features = _features(query)
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    records = _validated_exemplars(memory) if memory is not None else load_exemplars()
    scored = sorted(
        ((_similarity(features, item["features"]), item) for item in records),
        key=lambda pair: pair[0],
        reverse=True,
    )
    nearest = [pair for pair in scored if pair[0] >= MIN_SIMILARITY][:k]
    neighbors = [
        {"id": item["id"], "tool": item["tool"], "similarity": round(sim, 4),
         "verified_by": item["verified_by"]}
        for sim, item in nearest
    ]
    if len(nearest) < MIN_NEIGHBORS:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "tool": None, "reason": "too_few_similar_exemplars",
             "neighbors": neighbors},
            0.0,
            True,
            AuthorityMode.CONSULTATIVE,
        )
    tools = {item["tool"] for _, item in nearest}
    if len(tools) > 1:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "tool": None, "reason": "mixed_neighbor_tools",
             "neighbors": neighbors},
            0.0,
            True,
            AuthorityMode.CONSULTATIVE,
            notes=("Disagreeing neighbors do not elect a tool.",),
        )
    tool = nearest[0][1]["tool"]
    strength = min(0.95, sum(sim for sim, _ in nearest) / len(nearest))
    return SpecialistResult(
        SPECIALIST_ID,
        {"status": "suggest", "tool": tool, "neighbors": neighbors},
        round(strength, 4),
        False,
        AuthorityMode.CONSULTATIVE,
        notes=("Proposal only. No tool is executed.",),
    )
