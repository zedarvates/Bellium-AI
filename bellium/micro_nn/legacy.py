"""Load and run imported Botte Secrete micro-NNs. Observe/shadow only."""

from __future__ import annotations

import json
import math
import hashlib
from pathlib import Path

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.micro_nn.features import featurize
from bellium.micro_nn.mlp import load_mlp, predict_mlp
from bellium.resources import model_path
from legacy.micro_nn.calibration import apply_temperature, load_temperature

REPO_ROOT = Path(__file__).resolve().parents[2]
LEGACY_DIR = model_path("micro_nn", "legacy-botte")
PROVENANCE_PATH = LEGACY_DIR / "PROVENANCE.json"

LEGACY_META = {
    "anomaly_detector": {
        "id": "bellium/micro-nn/anomaly-detector:legacy-botte",
        "file": "anomaly_detector.json",
        "labels": ["normal", "anomaly"],
    },
    "binary_router": {
        "id": "bellium/micro-nn/binary-router:legacy-botte",
        "file": "binary_router.json",
        "labels": ["local", "cloud"],
    },
    "cloud_escalation_predictor": {
        "id": "bellium/micro-nn/cloud-escalation:legacy-botte",
        "file": "cloud_escalation_predictor.json",
        "labels": ["local_small", "local_big", "cloud"],
    },
    "compressibility_predictor": {
        "id": "bellium/micro-nn/compressibility:legacy-botte",
        "file": "compressibility_predictor.json",
        "labels": ["none", "delta", "heavy"],
    },
    "context_pruning_predictor": {
        "id": "bellium/micro-nn/context-pruning:legacy-botte",
        "file": "context_pruning_predictor.json",
        "labels": ["keep", "prune"],
    },
    "effort_classifier": {
        "id": "bellium/micro-nn/effort-classifier:legacy-botte",
        "file": "effort_classifier.json",
        "labels": ["easy (local)", "medium (hybrid)", "hard (cloud)"],
    },
    "error_classifier": {
        "id": "bellium/micro-nn/error-classifier:legacy-botte",
        "file": "error_classifier.json",
        "labels": ["syntax", "runtime", "network", "permission", "timeout", "resource"],
    },
    "response_length_predictor": {
        "id": "bellium/micro-nn/response-length:legacy-botte",
        "file": "response_length_predictor.json",
        "labels": ["short", "medium", "long"],
    },
    "semantic_cache_hit_predictor": {
        "id": "bellium/micro-nn/semantic-cache-hit:legacy-botte",
        "file": "semantic_cache_hit_predictor.json",
        "labels": ["miss", "hit"],
    },
    "skip_agent_predictor": {
        "id": "bellium/micro-nn/skip-agent:legacy-botte",
        "file": "skip_agent_predictor.json",
        "labels": ["execute", "skip"],
    },
    "tool_call_predictor": {
        "id": "bellium/micro-nn/tool-call:legacy-botte",
        "file": "tool_call_predictor.json",
        "labels": ["llm_only", "use_tool"],
    },
}

ABSTAIN_THRESHOLD = 0.45


def list_legacy_models() -> list[str]:
    return sorted(LEGACY_META)


def load_provenance() -> dict:
    if not PROVENANCE_PATH.exists():
        raise FileNotFoundError("legacy provenance is missing; run scripts/import_legacy_botte.py")
    return json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))


def load_legacy_model(name: str) -> dict:
    if name not in LEGACY_META:
        raise ValueError(f"unknown legacy model: {name}")
    path = LEGACY_DIR / LEGACY_META[name]["file"]
    if not path.exists():
        raise FileNotFoundError(f"missing imported weights: {path}")
    entries = load_provenance()["models"]
    expected = next((entry["sha256"] for entry in entries if entry["file"] == path.name), None)
    if expected is None or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"{name}: weights do not match the provenance manifest")
    return load_mlp(path)


def classify_legacy(
    name: str,
    values: dict[str, float],
    *,
    min_confidence: float = ABSTAIN_THRESHOLD,
) -> SpecialistResult:
    if isinstance(min_confidence, bool) or not isinstance(min_confidence, (int, float)) or not math.isfinite(min_confidence) or not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be finite and between 0 and 1")
    meta = LEGACY_META[name]
    vector = featurize(name, values)
    probs = predict_mlp(load_legacy_model(name), vector)
    probs = apply_temperature(probs, load_temperature(name))
    labels = meta["labels"]
    if len(probs) != len(labels):
        raise ValueError(f"{name}: output size {len(probs)} does not match labels {len(labels)}")
    best = max(range(len(probs)), key=probs.__getitem__)
    confidence = float(probs[best])
    abstained = confidence < min_confidence
    return SpecialistResult(
        specialist_id=meta["id"],
        output={
            "label": None if abstained else labels[best],
            "probabilities": {labels[i]: float(probs[i]) for i in range(len(labels))},
            "source": "legacy-botte",
        },
        confidence=confidence,
        abstained=abstained,
        authority_mode=AuthorityMode.OBSERVE,
        evidence_ref=str(LEGACY_META[name]["file"]),
        notes=("Imported weights are not production-authoritative.",),
    )
