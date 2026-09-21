"""Named feature contracts for Bellium micro-NNs.

Legacy Botte Secrete schemas are preserved so imported weights stay valid.
Unknown, missing or extra features fail closed.
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    lo: float
    hi: float
    desc: str


SCHEMAS: dict[str, list[FeatureSpec]] = {
    "binary_router": [
        FeatureSpec("complexity", 0.0, 1.0, "task complexity"),
        FeatureSpec("budget_ratio", 0.0, 1.0, "remaining budget / total"),
        FeatureSpec("has_local_model", 0.0, 1.0, "local backend reachable"),
    ],
    "effort_classifier": [
        FeatureSpec("file_size_ratio", 0.0, 1.0, "content size / 100KB"),
        FeatureSpec("token_ratio", 0.0, 1.0, "word count / 2000"),
        FeatureSpec("is_code", 0.0, 1.0, "contains code"),
        FeatureSpec("depth_ratio", 0.0, 1.0, "directory or reasoning depth / 10"),
    ],
    "anomaly_detector": [
        FeatureSpec("log_freq", 0.0, 1.0, "log entries per minute"),
        FeatureSpec("error_ratio", 0.0, 1.0, "fraction of error lines"),
        FeatureSpec("unique_errors", 0.0, 1.0, "distinct error types / 10"),
        FeatureSpec("avg_latency", 0.0, 1.0, "request latency / 1000ms"),
        FeatureSpec("retry_count", 0.0, 1.0, "retries / 5"),
    ],
    "error_classifier": [
        FeatureSpec("error_code_norm", 0.0, 1.0, "exit code / 255"),
        FeatureSpec("message_length_ratio", 0.0, 1.0, "message length / 2000"),
        FeatureSpec("has_traceback", 0.0, 1.0, "stack trace present"),
        FeatureSpec("kw_syntax", 0.0, 1.0, "syntax error keywords"),
        FeatureSpec("kw_runtime", 0.0, 1.0, "runtime error keywords"),
        FeatureSpec("kw_network", 0.0, 1.0, "network error keywords"),
        FeatureSpec("kw_permission", 0.0, 1.0, "permission error keywords"),
        FeatureSpec("kw_timeout", 0.0, 1.0, "timeout keywords"),
        FeatureSpec("kw_resource", 0.0, 1.0, "resource exhaustion keywords"),
        FeatureSpec("line_count_ratio", 0.0, 1.0, "error lines / 100"),
        FeatureSpec("has_suggestion", 0.0, 1.0, "did you mean present"),
        FeatureSpec("exit_code_nonzero", 0.0, 1.0, "process failed"),
    ],
    "compressibility_predictor": [
        FeatureSpec("raw_length", 0.0, 1.0, "content length / 100KB"),
        FeatureSpec("content_type", 0.0, 1.0, "0=text 0.25=json 0.5=log 0.75=code 1.0=tool"),
        FeatureSpec("repetition_ratio", 0.0, 1.0, "fraction of repeated lines"),
        FeatureSpec("entropy", 0.0, 1.0, "character entropy"),
        FeatureSpec("has_structure", 0.0, 1.0, "structured document present"),
        FeatureSpec("json_depth", 0.0, 1.0, "JSON nesting depth / 10"),
    ],
    "context_pruning_predictor": [
        FeatureSpec("total_size", 0.0, 1.0, "total context size / 100KB"),
        FeatureSpec("num_sections", 0.0, 1.0, "context sections / 20"),
        FeatureSpec("section_types", 0.0, 1.0, "0=code 0.33=doc 0.66=log 1.0=mixed"),
        FeatureSpec("usage_freq", 0.0, 1.0, "historical usage frequency"),
        FeatureSpec("query_similarity", 0.0, 1.0, "distance to query"),
        FeatureSpec("section_density", 0.0, 1.0, "non-whitespace ratio"),
    ],
    "skip_agent_predictor": [
        FeatureSpec("fingerprint_match", 0.0, 1.0, "code fingerprint matches cache"),
        FeatureSpec("project_hash_match", 0.0, 1.0, "project content unchanged"),
        FeatureSpec("agent_type", 0.0, 1.0, "0=audit 0.33=fix 0.66=optimize 1.0=analyze"),
        FeatureSpec("cache_history", 0.0, 1.0, "previous runs cached"),
        FeatureSpec("semantic_distance", 0.0, 1.0, "distance to previous queries"),
        FeatureSpec("criticality", 0.0, 1.0, "task criticality"),
        FeatureSpec("recent_skip_rate", 0.0, 1.0, "recent correct skips"),
    ],
    "cloud_escalation_predictor": [
        FeatureSpec("effort_score", 0.0, 1.0, "effort classifier output"),
        FeatureSpec("task_type", 0.0, 1.0, "0=audit 0.25=fix 0.5=analyze 0.75=design 1.0=research"),
        FeatureSpec("local_fail_history", 0.0, 1.0, "recent local failures"),
        FeatureSpec("criticality", 0.0, 1.0, "business criticality"),
        FeatureSpec("budget_remaining", 0.0, 1.0, "token budget remaining"),
        FeatureSpec("time_pressure", 0.0, 1.0, "time remaining inverted"),
        FeatureSpec("local_model_quality", 0.0, 1.0, "estimated local model quality"),
    ],
    "response_length_predictor": [
        FeatureSpec("query_type", 0.0, 1.0, "0=simple 0.33=explain 0.66=analyze 1.0=design"),
        FeatureSpec("agent_type", 0.0, 1.0, "0=audit 0.33=fix 0.66=report 1.0=analyze"),
        FeatureSpec("criticality", 0.0, 1.0, "task criticality"),
        FeatureSpec("history_avg_length", 0.0, 1.0, "average previous length / 4000"),
        FeatureSpec("user_pref", 0.0, 1.0, "0=short 0.5=medium 1.0=long"),
        FeatureSpec("query_complexity", 0.0, 1.0, "complexity score"),
    ],
    "tool_call_predictor": [
        FeatureSpec("has_code", 0.0, 1.0, "query contains code"),
        FeatureSpec("has_files", 0.0, 1.0, "query references files"),
        FeatureSpec("tool_history", 0.0, 1.0, "previous runs using tools"),
        FeatureSpec("query_type", 0.0, 1.0, "0=ask 0.33=fix 0.66=audit 1.0=deploy"),
        FeatureSpec("criticality", 0.0, 1.0, "task criticality"),
        FeatureSpec("budget_ratio", 0.0, 1.0, "budget remaining"),
        FeatureSpec("time_ratio", 0.0, 1.0, "time available / required"),
    ],
    "semantic_cache_hit_predictor": [
        FeatureSpec("cache_density", 0.0, 1.0, "cache entries / capacity"),
        FeatureSpec("query_embedding_norm", 0.0, 1.0, "query embedding norm"),
        FeatureSpec("avg_distance", 0.0, 1.0, "mean distance to cache"),
        FeatureSpec("agent_type", 0.0, 1.0, "0=audit 0.33=fix 0.66=report 1.0=analyze"),
        FeatureSpec("pattern_frequency", 0.0, 1.0, "query pattern frequency"),
        FeatureSpec("cache_hit_history", 0.0, 1.0, "recent cache hits"),
        FeatureSpec("query_length", 0.0, 1.0, "query length / 2000"),
    ],
    "inpaint_router": [
        FeatureSpec("mask_area_ratio", 0.0, 1.0, "masked pixels / image pixels"),
        FeatureSpec("bbox_fill", 0.0, 1.0, "mask area / bounding-box area"),
        FeatureSpec("border_touch", 0.0, 1.0, "mask touches image border"),
        FeatureSpec("neighbor_std", 0.0, 1.0, "color spread next to the hole"),
        FeatureSpec("hole_count_norm", 0.0, 1.0, "connected holes / 5"),
        FeatureSpec("max_span_ratio", 0.0, 1.0, "largest hole span / image span"),
    ],
    "tool_router": [
        FeatureSpec("has_code", 0.0, 1.0, "task mentions code"),
        FeatureSpec("has_files", 0.0, 1.0, "task mentions files"),
        FeatureSpec("has_error", 0.0, 1.0, "task mentions a failure"),
        FeatureSpec("wants_mutation", 0.0, 1.0, "task wants a write or deploy"),
        FeatureSpec("wants_external_state", 0.0, 1.0, "task needs live external state"),
        FeatureSpec("criticality", 0.0, 1.0, "caller-supplied criticality"),
    ],
    "npc_behavior_router": [
        FeatureSpec("threat_level", 0.0, 1.0, "perceived threat of the current situation"),
        FeatureSpec("distance_ratio", 0.0, 1.0, "distance to the stimulus / engagement range"),
        FeatureSpec("health_ratio", 0.0, 1.0, "current health / maximum health"),
        FeatureSpec("support_ratio", 0.0, 1.0, "nearby allies / expected allies"),
        FeatureSpec("visibility", 0.0, 1.0, "how clearly the stimulus is perceived"),
        FeatureSpec("alertness", 0.0, 1.0, "current alert level of the agent"),
        FeatureSpec("cover_ratio", 0.0, 1.0, "cover available on the current route"),
        FeatureSpec("cooldown_ready", 0.0, 1.0, "the agent can act right now"),
    ],
    "consequence_predictor": [
        FeatureSpec("reversible", 0.0, 1.0, "the action can be undone"),
        FeatureSpec("scope_ratio", 0.0, 1.0, "declared scope / largest allowed scope"),
        FeatureSpec("touches_protected", 0.0, 1.0, "touches protected, production or private data"),
        FeatureSpec("has_backup", 0.0, 1.0, "a verified backup or snapshot exists"),
        FeatureSpec("dry_run_available", 0.0, 1.0, "the action can be previewed first"),
        FeatureSpec("blast_radius", 0.0, 1.0, "declared blast radius if it goes wrong"),
        FeatureSpec("idempotent", 0.0, 1.0, "repeating the action changes nothing more"),
        FeatureSpec("recent_failures", 0.0, 1.0, "recent failures in this action family"),
    ],
}

_CODE_RE = re.compile(r"```|\bdef \b|\bclass \b|\bfunction\b|\bimport \b|=>|;\s*$", re.M)
_ERR_KW = {
    "kw_syntax": ("syntaxerror", "invalid syntax", "unexpected eof", "unexpected indent", "eol"),
    "kw_runtime": ("typeerror", "valueerror", "keyerror", "indexerror", "attributeerror", "nameerror"),
    "kw_network": ("connectionerror", "econnrefused", "dns", "socket", "httperror", "ssl", "unreachable"),
    "kw_permission": ("permission denied", "access denied", "eacces", "forbidden", "403", "not permitted"),
    "kw_timeout": ("timeout", "timed out", "timeouterror", "deadline"),
    "kw_resource": ("memoryerror", "oom", "out of memory", "disk full", "enospc", "no space left"),
}


def _clamp(value: float, lo: float, hi: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("features must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("features must be finite")
    if number < lo:
        return lo
    if number > hi:
        return hi
    return number


def feature_names(model: str) -> list[str]:
    return [spec.name for spec in SCHEMAS[model]]


def featurize(model: str, values: dict[str, float]) -> list[float]:
    if model not in SCHEMAS:
        known = ", ".join(sorted(SCHEMAS))
        raise ValueError(f"unknown model '{model}'; known: {known}")
    specs = SCHEMAS[model]
    allowed = {spec.name for spec in specs}
    missing = [spec.name for spec in specs if spec.name not in values]
    extra = [key for key in values if key not in allowed]
    if missing or extra:
        raise ValueError(f"{model}: feature mismatch missing={missing} extra={extra}")
    return [_clamp(values[spec.name], spec.lo, spec.hi) for spec in specs]


def binary_router_values(complexity: float, budget_ratio: float, has_local: bool) -> dict[str, float]:
    return {
        "complexity": complexity,
        "budget_ratio": budget_ratio,
        "has_local_model": 1.0 if has_local else 0.0,
    }


def effort_classifier_values(text: str, *, is_code: bool | None = None, depth: int = 0) -> dict[str, float]:
    code = bool(_CODE_RE.search(text)) if is_code is None else is_code
    return {
        "file_size_ratio": min(len(text) / 102_400, 1.0),
        "token_ratio": min(len(text.split()) / 2000, 1.0),
        "is_code": 1.0 if code else 0.0,
        "depth_ratio": min(depth / 10.0, 1.0),
    }


def error_classifier_values(error_text: str, *, exit_code: int = 1) -> dict[str, float]:
    low = error_text.lower()
    traceback_hit = bool(re.search(r"traceback|stack trace|at .+\(.+:\d+\)", low))
    return {
        "error_code_norm": min(abs(exit_code), 255) / 255.0,
        "message_length_ratio": min(len(error_text) / 2000, 1.0),
        "has_traceback": 1.0 if traceback_hit else 0.0,
        **{name: (1.0 if any(token in low for token in terms) else 0.0) for name, terms in _ERR_KW.items()},
        "line_count_ratio": min(error_text.count("\n") / 100.0, 1.0),
        "has_suggestion": 1.0 if "did you mean" in low else 0.0,
        "exit_code_nonzero": 1.0 if exit_code != 0 else 0.0,
    }
