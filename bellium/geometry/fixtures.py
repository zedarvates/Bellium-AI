"""Authored immutable fixtures for ShapeGrammar 3D contract tests."""

from __future__ import annotations

from copy import deepcopy

from bellium.geometry.plan import ConstructionPlan, plan_from_dict

_FIXTURES = {
    "sci-fi-hammer": {
        "schema_version": "primitive-mesh-plan-v0",
        "asset_id": "fixture-sci-fi-hammer-v0",
        "profile": "hard-surface-prop",
        "concept_sources": [
            {
                "kind": "text",
                "source_id": "authored-fixture",
                "confidence": 1.0,
                "notes": "Deterministic contract fixture; not an inferred concept.",
            }
        ],
        "parts": [
            {
                "id": "handle",
                "primitive": "cylinder",
                "semantic_role": "grip",
                "transform": {
                    "translation": [0.0, -1.6, 0.0],
                    "rotation_euler": [0.0, 0.0, 0.0],
                    "scale": [1.0, 1.0, 1.0],
                },
                "dimensions": [0.32, 3.4],
                "symmetry": "axial-y",
                "editable_parameters": {"radius": 0.16, "length": 3.4},
            },
            {
                "id": "head",
                "primitive": "box",
                "semantic_role": "hammer-head",
                "transform": {
                    "translation": [0.0, 0.35, 0.0],
                    "rotation_euler": [0.0, 0.0, 0.0],
                    "scale": [1.0, 1.0, 1.0],
                },
                "dimensions": [2.6, 0.9, 0.95],
                "symmetry": "mirror-x",
                "editable_parameters": {"width": 2.6, "height": 0.9, "depth": 0.95},
            },
            {
                "id": "core",
                "primitive": "cylinder",
                "semantic_role": "energy-core",
                "transform": {
                    "translation": [0.0, 0.35, 0.0],
                    "rotation_euler": [0.0, 0.0, 90.0],
                    "scale": [1.0, 1.0, 1.0],
                },
                "dimensions": [0.46, 1.05],
                "symmetry": "axial-x",
                "editable_parameters": {"radius": 0.23, "length": 1.05},
            },
        ],
        "operations": [
            {
                "id": "bevel-head",
                "operator": "bevel",
                "input_ids": ["head"],
                "parameters": {"width": 0.08, "segments": 2},
                "semantic_intent": "soften silhouette while preserving hard-surface form",
                "source_evidence": "fixture-rule",
                "confidence": 1.0,
                "reversible": True,
                "output_ids": ["head-beveled"],
            },
            {
                "id": "mirror-core-detail",
                "operator": "mirror",
                "input_ids": ["core"],
                "parameters": {"axis": "x", "offset": 0.72},
                "semantic_intent": "paired energy-core detail",
                "source_evidence": "fixture-rule",
                "confidence": 1.0,
                "reversible": True,
                "output_ids": ["core-pair"],
            },
        ],
        "surface_patterns": [
            {
                "pattern": "brushed-metal",
                "representation": "normal-bump",
                "target_ids": ["head-beveled"],
                "scale": 0.05,
                "orientation": "x",
                "confidence": 1.0,
            },
            {
                "pattern": "grip-ridges",
                "representation": "normal-bump",
                "target_ids": ["handle"],
                "scale": 0.08,
                "orientation": "around-axis",
                "confidence": 1.0,
            },
        ],
        "handoff": {
            "run_approx_surface": True,
            "run_topology_grammar": True,
            "generate_uv": True,
            "generate_lod": True,
            "generate_collision": True,
        },
        "provenance": {
            "method": "authored-deterministic-fixture",
            "tool_version": "bellium-shapegrammar-v0",
            "source_hash": "fixture:sci-fi-hammer:v0",
            "notes": "No geometry execution claim.",
        },
    }
}

FIXTURE_NAMES = tuple(sorted(_FIXTURES))


def fixture(name: str) -> ConstructionPlan:
    if name not in _FIXTURES:
        raise KeyError(name)
    return plan_from_dict(deepcopy(_FIXTURES[name]))
