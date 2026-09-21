"""Consultative tool router: deterministic veto, then k-NN, then micro-NN."""

from __future__ import annotations

from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.resources import model_path
from bellium.knn.tool_router import retrieve_tool, _features
from bellium.micro_nn.features import featurize
from bellium.micro_nn.mlp import load_mlp, predict_mlp

SPECIALIST_ID = "bellium/hybrid/tool-router:v0"
NN_ID = "bellium/micro-nn/tool-router:v0"
MODEL_PATH = (
    model_path()
    / "micro_nn"
    / "tool-router"
    / "v0.json"
)
LABELS = ("none", "use_tool", "escalate")


def classify_tool_need(query: dict[str, Any]) -> SpecialistResult:
    values = _features(query)
    vector = featurize("tool_router", values)
    probs = predict_mlp(load_mlp(MODEL_PATH), vector)
    best = max(range(len(probs)), key=probs.__getitem__)
    confidence = float(probs[best])
    abstained = confidence < 0.55
    return SpecialistResult(
        NN_ID,
        {
            "label": None if abstained else LABELS[best],
            "probabilities": {LABELS[i]: round(float(probs[i]), 4) for i in range(len(LABELS))},
        },
        round(confidence, 4),
        abstained,
        AuthorityMode.CONSULTATIVE,
        evidence_ref=MODEL_PATH.name,
        notes=("Micro-NN chooses none/use_tool/escalate, never a concrete tool.",),
    )


def route_tool(query: dict[str, Any]) -> SpecialistResult:
    knn = retrieve_tool(query)
    if knn.output.get("status") == "rule_escalate":
        return SpecialistResult(
            SPECIALIST_ID,
            {"tool": "escalate", "status": "escalate", "knn": knn.output, "micro_nn": None},
            1.0, False, AuthorityMode.CONSULTATIVE,
            notes=("Deterministic veto returned before consulting any model.",),
        )
    nn = classify_tool_need(query)
    if nn.abstained:
        return SpecialistResult(
            SPECIALIST_ID,
            {"tool": None, "status": "abstain", "reason": "micro_nn_abstained",
             "knn": knn.output, "micro_nn": nn.output},
            0.0, True, AuthorityMode.CONSULTATIVE,
        )
    nn_label = None if nn.abstained else nn.output["label"]
    if nn_label == "escalate":
        return SpecialistResult(
            SPECIALIST_ID,
            {
                "tool": "escalate",
                "status": "escalate",
                "knn": knn.output,
                "micro_nn": nn.output,
            },
            nn.confidence,
            False,
            AuthorityMode.CONSULTATIVE,
            notes=("Escalate is consultative. No tool is launched.",),
        )
    if knn.abstained and nn_label == "none":
        return SpecialistResult(
            SPECIALIST_ID,
            {"tool": "none", "status": "suggest", "knn": knn.output, "micro_nn": nn.output},
            nn.confidence,
            False,
            AuthorityMode.CONSULTATIVE,
            notes=("No similar exemplar; micro-NN only supports doing nothing.",),
        )
    if knn.abstained:
        return SpecialistResult(
            SPECIALIST_ID,
            {"tool": None, "status": "abstain", "knn": knn.output, "micro_nn": nn.output,
             "reason": "knn_abstained"},
            0.0,
            True,
            AuthorityMode.CONSULTATIVE,
        )
    tool = knn.output.get("tool")
    if (nn_label == "none" and tool not in (None, "none")) or (nn_label == "use_tool" and tool == "none"):
        return SpecialistResult(
            SPECIALIST_ID,
            {"tool": None, "status": "abstain", "reason": "knn_and_micro_nn_disagree",
             "knn": knn.output, "micro_nn": nn.output},
            0.0,
            True,
            AuthorityMode.CONSULTATIVE,
            notes=("Disagreement fails closed to abstention.",),
        )
    return SpecialistResult(
        SPECIALIST_ID,
        {"tool": tool, "status": "suggest", "knn": knn.output, "micro_nn": nn.output},
        knn.confidence,
        False,
        AuthorityMode.CONSULTATIVE,
        notes=("Proposal only. Deterministic policy still owns execution.",),
    )
