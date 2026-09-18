"""Hybrid Specialist Router.

Matches user intent / task specifications against registered tools following the
Bellium priority order:
1. Deterministic algorithm / rule / classical CV / DSP
2. k-NN / nearest exemplars
3. Micro-NN (specialist classifier)
4. Specialist micro-LLM / compact local model
5. Large local or remote model

Zero external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Dict, Set, Optional, Tuple


class EscalationTier(str, Enum):
    DETERMINISTIC = "tier_0_deterministic"
    KNN_EXEMPLAR = "tier_1_knn"
    MICRO_NN = "tier_2_micro_nn"
    MICRO_LLM = "tier_3_micro_llm"
    GENERAL_LARGE = "tier_4_large_model"


@dataclass
class ToolCapability:
    tool_id: str
    tier: EscalationTier
    modalities: Set[str]  # e.g. {'image', 'text', 'code'}
    tags: Set[str]  # e.g. {'cutout', 'inpaint', 'routing', 'formatting'}
    safety_critical: bool = False
    latency_budget_ms: float = 50.0
    success_rate: float = 0.95


@dataclass
class TaskRequirement:
    task_id: str
    modality: str
    tags: Set[str]
    text_sample: str = ""
    hard_constraints: Set[str] = field(default_factory=set)  # e.g. {'offline_only', 'zero_tokens'}
    max_latency_ms: Optional[float] = None


@dataclass
class RouteVerdict:
    selected_tool_id: Optional[str]
    selected_tier: EscalationTier
    confidence: float
    reason: str
    fallback_tier: Optional[EscalationTier] = None


class HybridRouter:
    """Deterministic priority router with capability matching and graceful abstention."""

    def __init__(self):
        self._registry: Dict[str, ToolCapability] = {}

    def register_tool(self, tool: ToolCapability) -> None:
        if not isinstance(tool.tier, EscalationTier) or not tool.tool_id:
            raise ValueError("tool needs an id and a valid tier")
        if not math.isfinite(tool.success_rate) or not 0 <= tool.success_rate <= 1:
            raise ValueError("success_rate must be finite and in [0, 1]")
        if not math.isfinite(tool.latency_budget_ms) or tool.latency_budget_ms < 0:
            raise ValueError("latency budget must be finite and non-negative")
        self._registry[tool.tool_id] = tool

    def route_task(self, req: TaskRequirement) -> RouteVerdict:
        if req.hard_constraints - {"zero_tokens", "offline_only"}:
            return RouteVerdict(
                None, EscalationTier.DETERMINISTIC, 0.0, "Unknown hard constraint; abstaining."
            )
        if req.max_latency_ms is not None and (
            not math.isfinite(req.max_latency_ms) or req.max_latency_ms < 0
        ):
            raise ValueError("maximum latency must be finite and non-negative")
        # 1. Filter by modality and hard constraints
        candidates = []
        for tool in self._registry.values():
            if req.modality not in tool.modalities:
                continue
            if req.max_latency_ms is not None and tool.latency_budget_ms > req.max_latency_ms:
                continue
            if "zero_tokens" in req.hard_constraints and tool.tier in (
                EscalationTier.MICRO_LLM,
                EscalationTier.GENERAL_LARGE,
            ):
                continue
            if "offline_only" in req.hard_constraints and tool.tier == EscalationTier.GENERAL_LARGE:
                continue
            candidates.append(tool)

        if not candidates:
            return RouteVerdict(
                selected_tool_id=None,
                selected_tier=EscalationTier.GENERAL_LARGE,
                confidence=0.0,
                reason="No candidate satisfied task modality/hard constraints. Abstaining.",
            )

        # 2. Score candidate tools by tag overlap and lowest competence tier
        # Preference order ranking
        tier_priority = {
            EscalationTier.DETERMINISTIC: 0,
            EscalationTier.KNN_EXEMPLAR: 1,
            EscalationTier.MICRO_NN: 2,
            EscalationTier.MICRO_LLM: 3,
            EscalationTier.GENERAL_LARGE: 4,
        }

        def score_candidate(t: ToolCapability) -> Tuple[int, int, float]:
            overlap = len(t.tags.intersection(req.tags))
            tier_rank = tier_priority[t.tier]
            # Return tuple to sort: max overlap (-overlap), min tier_rank, max success (-t.success_rate)
            return (-overlap, tier_rank, -t.success_rate)

        candidates.sort(key=score_candidate)
        best = candidates[0]
        overlap_count = len(best.tags.intersection(req.tags))

        if overlap_count == 0 and req.tags:
            # No tags matched, escalate to general fallback
            return RouteVerdict(
                selected_tool_id=None,
                selected_tier=best.tier,
                confidence=0.0,
                reason="No candidate matches the requested task; abstaining.",
            )

        conf = min(1.0, 0.5 + 0.25 * overlap_count) * best.success_rate
        fallback = next(
            (
                tool.tier
                for tool in candidates[1:]
                if tool.tags.intersection(req.tags) and tool.tier != best.tier
            ),
            None,
        )

        return RouteVerdict(
            selected_tool_id=best.tool_id,
            selected_tier=best.tier,
            confidence=round(conf, 3),
            reason=f"Matched {overlap_count} tags on {best.tool_id} at {best.tier.value}",
            fallback_tier=fallback,
        )
