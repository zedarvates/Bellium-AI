"""Route declarative capabilities without executing any tool."""

from dataclasses import asdict
import json

from bellium.routing import EscalationTier, HybridRouter, TaskRequirement, ToolCapability


def propose_tool(tags: set[str], modality: str = "image") -> dict:
    router = HybridRouter()
    for tool_id, tool_tags, tier in (
        ("image.grayscale", {"grayscale", "filter"}, EscalationTier.DETERMINISTIC),
        ("image.sepia", {"sepia", "filter"}, EscalationTier.DETERMINISTIC),
        ("image.inpaint", {"inpaint", "fill"}, EscalationTier.KNN_EXEMPLAR),
    ):
        router.register_tool(ToolCapability(
            tool_id=tool_id, tier=tier, modalities={"image"}, tags=tool_tags,
            # These values are illustrative catalog metadata, not benchmark results.
            latency_budget_ms=50.0, success_rate=0.8,
        ))
    verdict = router.route_task(TaskRequirement(
        task_id="example", modality=modality, tags=tags,
        hard_constraints={"offline_only", "zero_tokens"},
    ))
    # The published router may return a low-confidence fallback with no tag overlap.
    # This consumer treats that as abstention, never as authorization to execute.
    proposal = verdict.selected_tool_id if tags and verdict.confidence >= 0.5 else None
    return {"proposal": proposal, "executed": False, "verdict": asdict(verdict)}


def main() -> None:
    print(json.dumps({
        "matched": propose_tool({"sepia", "filter"}),
        "unsupported": propose_tool({"earth_gravity"}),
        "unsupported_modality": propose_tool({"transcribe"}, modality="audio"),
        "limits": "Heuristic routing score; catalog metadata is not measured performance.",
    }, indent=2))


if __name__ == "__main__":
    main()
