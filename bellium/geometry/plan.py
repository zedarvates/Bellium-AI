"""Strict, dependency-free contract for primitive-to-complex mesh plans.

This module validates an operation graph only. It does not execute geometry.
Unknown fields fail closed so downstream tools cannot silently reinterpret plans.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

SCHEMA_VERSION = "primitive-mesh-plan-v0"
PROFILES = {
    "hard-surface-prop",
    "modular-environment",
    "stylized-organic",
    "humanoid-base-mesh",
    "creature-blockout",
    "printable-object",
}
PRIMITIVES = {
    "box", "cylinder", "sphere", "cone", "torus", "plane", "capsule",
    "polyline", "spline", "extruded-profile", "lathe-profile", "base-mesh",
}
OPERATORS = {
    "translate", "rotate", "scale", "extrude", "inset", "bevel", "bridge",
    "boolean-union", "boolean-difference", "boolean-intersection",
    "bend", "twist", "taper", "smooth", "subdivision",
    "shrinkwrap", "project", "mirror", "array", "sweep",
}
REPRESENTATIONS = {"texture", "normal-bump", "displacement", "geometry"}
ROOT_KEYS = {
    "schema_version", "asset_id", "profile", "concept_sources", "parts",
    "operations", "surface_patterns", "handoff", "provenance",
}
PART_KEYS = {
    "id", "primitive", "semantic_role", "transform", "dimensions", "symmetry",
    "editable_parameters",
}
TRANSFORM_KEYS = {"translation", "rotation_euler", "scale"}
OPERATION_KEYS = {
    "id", "operator", "input_ids", "parameters", "semantic_intent",
    "source_evidence", "confidence", "reversible", "output_ids",
}
PATTERN_KEYS = {
    "pattern", "representation", "target_ids", "scale", "orientation", "confidence",
}
HANDOFF_KEYS = {
    "run_approx_surface", "run_topology_grammar", "generate_uv",
    "generate_lod", "generate_collision",
}
PROVENANCE_KEYS = {"method", "tool_version", "source_hash", "notes"}
CONCEPT_KEYS = {"kind", "source_id", "confidence", "notes"}
CONCEPT_KINDS = {"image", "sketch", "text", "pattern-reference"}


@dataclass(frozen=True)
class Part:
    id: str
    primitive: str
    semantic_role: str
    transform: dict[str, tuple[float, float, float]]
    dimensions: tuple[float, ...] = ()
    symmetry: str | None = None
    editable_parameters: dict[str, Any] | None = None


@dataclass(frozen=True)
class Operation:
    id: str
    operator: str
    input_ids: tuple[str, ...]
    parameters: dict[str, Any]
    reversible: bool
    semantic_intent: str | None = None
    source_evidence: str | None = None
    confidence: float | None = None
    output_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class SurfacePattern:
    pattern: str
    representation: str
    target_ids: tuple[str, ...]
    scale: float | None = None
    orientation: str | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class ConstructionPlan:
    asset_id: str
    profile: str
    parts: tuple[Part, ...]
    operations: tuple[Operation, ...]
    surface_patterns: tuple[SurfacePattern, ...]
    concept_sources: tuple[dict[str, Any], ...]
    handoff: dict[str, bool]
    provenance: dict[str, Any]
    schema_version: str = SCHEMA_VERSION


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _unknown(payload: dict[str, Any], allowed: set[str], label: str) -> None:
    extra = sorted(set(payload) - allowed)
    if extra:
        raise ValueError(f"unknown {label} keys refuse to be ignored: {', '.join(extra)}")


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _confidence(value: object, name: str) -> float:
    result = _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return result


def _vec3(value: object, name: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three numbers")
    return tuple(_finite(item, f"{name}[{index}]") for index, item in enumerate(value))  # type: ignore[return-value]


def _string_list(value: object, name: str, *, nonempty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list")
    items = tuple(_text(item, f"{name}[]") for item in value)
    if nonempty and not items:
        raise ValueError(f"{name} must not be empty")
    return items


def _parse_part(value: object) -> Part:
    payload = _object(value, "part")
    _unknown(payload, PART_KEYS, "part")
    primitive = _text(payload.get("primitive"), "part.primitive")
    if primitive not in PRIMITIVES:
        raise ValueError(f"unsupported primitive: {primitive}")
    transform_payload = _object(payload.get("transform"), "part.transform")
    _unknown(transform_payload, TRANSFORM_KEYS, "transform")
    transform: dict[str, tuple[float, float, float]] = {}
    for key, default in (
        ("translation", (0.0, 0.0, 0.0)),
        ("rotation_euler", (0.0, 0.0, 0.0)),
        ("scale", (1.0, 1.0, 1.0)),
    ):
        transform[key] = _vec3(transform_payload.get(key, default), f"part.transform.{key}")
    dimensions_raw = payload.get("dimensions", [])
    if not isinstance(dimensions_raw, (list, tuple)):
        raise ValueError("part.dimensions must be a list")
    dimensions = tuple(_finite(item, "part.dimensions[]") for item in dimensions_raw)
    if any(item <= 0.0 for item in dimensions):
        raise ValueError("part.dimensions must be > 0")
    editable = payload.get("editable_parameters", {})
    if not isinstance(editable, dict):
        raise ValueError("part.editable_parameters must be an object")
    symmetry = payload.get("symmetry")
    if symmetry is not None:
        symmetry = _text(symmetry, "part.symmetry")
    return Part(
        id=_text(payload.get("id"), "part.id"),
        primitive=primitive,
        semantic_role=_text(payload.get("semantic_role"), "part.semantic_role"),
        transform=transform,
        dimensions=dimensions,
        symmetry=symmetry,
        editable_parameters=dict(editable),
    )


def _parse_operation(value: object) -> Operation:
    payload = _object(value, "operation")
    _unknown(payload, OPERATION_KEYS, "operation")
    operator = _text(payload.get("operator"), "operation.operator")
    if operator not in OPERATORS:
        raise ValueError(f"unsupported operator: {operator}")
    parameters = payload.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("operation.parameters must be an object")
    reversible = payload.get("reversible")
    if not isinstance(reversible, bool):
        raise ValueError("operation.reversible must be boolean")
    confidence = payload.get("confidence")
    return Operation(
        id=_text(payload.get("id"), "operation.id"),
        operator=operator,
        input_ids=_string_list(payload.get("input_ids"), "operation.input_ids", nonempty=True),
        parameters=dict(parameters),
        reversible=reversible,
        semantic_intent=(
            _text(payload["semantic_intent"], "operation.semantic_intent")
            if payload.get("semantic_intent") is not None else None
        ),
        source_evidence=(
            _text(payload["source_evidence"], "operation.source_evidence")
            if payload.get("source_evidence") is not None else None
        ),
        confidence=_confidence(confidence, "operation.confidence") if confidence is not None else None,
        output_ids=_string_list(payload.get("output_ids", []), "operation.output_ids"),
    )


def _parse_pattern(value: object) -> SurfacePattern:
    payload = _object(value, "surface pattern")
    _unknown(payload, PATTERN_KEYS, "surface pattern")
    representation = _text(payload.get("representation"), "surface_pattern.representation")
    if representation not in REPRESENTATIONS:
        raise ValueError(f"unsupported surface representation: {representation}")
    scale_raw = payload.get("scale")
    if scale_raw is not None:
        scale = _finite(scale_raw, "surface_pattern.scale")
        if scale < 0.0:
            raise ValueError("surface_pattern.scale must be >= 0")
    else:
        scale = None
    confidence = payload.get("confidence")
    orientation = payload.get("orientation")
    return SurfacePattern(
        pattern=_text(payload.get("pattern"), "surface_pattern.pattern"),
        representation=representation,
        target_ids=_string_list(payload.get("target_ids"), "surface_pattern.target_ids", nonempty=True),
        scale=scale,
        orientation=_text(orientation, "surface_pattern.orientation") if orientation is not None else None,
        confidence=_confidence(confidence, "surface_pattern.confidence") if confidence is not None else None,
    )


def plan_from_dict(value: object) -> ConstructionPlan:
    payload = _object(value, "construction plan")
    _unknown(payload, ROOT_KEYS, "construction plan")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    profile = _text(payload.get("profile"), "profile")
    if profile not in PROFILES:
        raise ValueError(f"unsupported construction profile: {profile}")

    raw_parts = payload.get("parts")
    raw_operations = payload.get("operations")
    raw_patterns = payload.get("surface_patterns", [])
    raw_sources = payload.get("concept_sources", [])
    if not isinstance(raw_parts, (list, tuple)) or not raw_parts:
        raise ValueError("parts must be a non-empty list")
    if not isinstance(raw_operations, (list, tuple)):
        raise ValueError("operations must be a list")
    if not isinstance(raw_patterns, (list, tuple)):
        raise ValueError("surface_patterns must be a list")
    if not isinstance(raw_sources, (list, tuple)):
        raise ValueError("concept_sources must be a list")

    sources: list[dict[str, Any]] = []
    for source in raw_sources:
        item = _object(source, "concept source")
        _unknown(item, CONCEPT_KEYS, "concept source")
        kind = _text(item.get("kind"), "concept_source.kind")
        if kind not in CONCEPT_KINDS:
            raise ValueError(f"unsupported concept source kind: {kind}")
        normalized = dict(item)
        normalized["kind"] = kind
        normalized["confidence"] = _confidence(item.get("confidence"), "concept_source.confidence")
        sources.append(normalized)

    handoff_payload = _object(payload.get("handoff", {}), "handoff")
    _unknown(handoff_payload, HANDOFF_KEYS, "handoff")
    handoff: dict[str, bool] = {}
    for key in HANDOFF_KEYS:
        flag = handoff_payload.get(key, False)
        if not isinstance(flag, bool):
            raise ValueError(f"handoff.{key} must be boolean")
        handoff[key] = flag

    provenance = _object(payload.get("provenance"), "provenance")
    _unknown(provenance, PROVENANCE_KEYS, "provenance")
    _text(provenance.get("method"), "provenance.method")

    plan = ConstructionPlan(
        asset_id=_text(payload.get("asset_id"), "asset_id"),
        profile=profile,
        parts=tuple(_parse_part(item) for item in raw_parts),
        operations=tuple(_parse_operation(item) for item in raw_operations),
        surface_patterns=tuple(_parse_pattern(item) for item in raw_patterns),
        concept_sources=tuple(sources),
        handoff=handoff,
        provenance=dict(provenance),
    )
    violations = check_plan(plan)
    if violations:
        raise ValueError("invalid construction plan: " + ", ".join(violations))
    return plan


def check_plan(plan: ConstructionPlan) -> list[str]:
    violations: list[str] = []
    part_ids = [part.id for part in plan.parts]
    if len(part_ids) != len(set(part_ids)):
        violations.append("duplicate_part_id")
    operation_ids = [operation.id for operation in plan.operations]
    if len(operation_ids) != len(set(operation_ids)):
        violations.append("duplicate_operation_id")
    known = set(part_ids)
    for index, operation in enumerate(plan.operations):
        missing = [item for item in operation.input_ids if item not in known]
        if missing:
            violations.append(f"operation_{index}_missing_input")
        for output in operation.output_ids:
            if output in known:
                violations.append(f"operation_{index}_duplicate_output")
            known.add(output)
    for index, pattern in enumerate(plan.surface_patterns):
        if any(target not in known for target in pattern.target_ids):
            violations.append(f"surface_pattern_{index}_missing_target")
    return violations


def plan_to_dict(plan: ConstructionPlan) -> dict[str, Any]:
    return {
        "schema_version": plan.schema_version,
        "asset_id": plan.asset_id,
        "profile": plan.profile,
        "concept_sources": [dict(source) for source in plan.concept_sources],
        "parts": [
            {
                "id": part.id,
                "primitive": part.primitive,
                "semantic_role": part.semantic_role,
                "transform": {key: list(value) for key, value in part.transform.items()},
                "dimensions": list(part.dimensions),
                **({"symmetry": part.symmetry} if part.symmetry is not None else {}),
                "editable_parameters": dict(part.editable_parameters or {}),
            }
            for part in plan.parts
        ],
        "operations": [
            {
                "id": op.id,
                "operator": op.operator,
                "input_ids": list(op.input_ids),
                "parameters": dict(op.parameters),
                "reversible": op.reversible,
                **({"semantic_intent": op.semantic_intent} if op.semantic_intent is not None else {}),
                **({"source_evidence": op.source_evidence} if op.source_evidence is not None else {}),
                **({"confidence": op.confidence} if op.confidence is not None else {}),
                "output_ids": list(op.output_ids),
            }
            for op in plan.operations
        ],
        "surface_patterns": [
            {
                "pattern": item.pattern,
                "representation": item.representation,
                "target_ids": list(item.target_ids),
                **({"scale": item.scale} if item.scale is not None else {}),
                **({"orientation": item.orientation} if item.orientation is not None else {}),
                **({"confidence": item.confidence} if item.confidence is not None else {}),
            }
            for item in plan.surface_patterns
        ],
        "handoff": dict(plan.handoff),
        "provenance": dict(plan.provenance),
    }
