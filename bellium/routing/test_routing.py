import os
import sys

sys.path.insert(0, os.path.abspath("temp_bellium_repo"))
from bellium.routing import HybridRouter, ToolCapability, TaskRequirement, EscalationTier


def run_tests():
    router = HybridRouter()

    # Register tools across tiers
    # Tier 0: Deterministic cutout
    router.register_tool(
        ToolCapability(
            tool_id="bellium/cutout",
            tier=EscalationTier.DETERMINISTIC,
            modalities={"image"},
            tags={"cutout", "alpha", "segmentation", "background"},
            latency_budget_ms=10.0,
            success_rate=0.98,
        )
    )

    # Tier 1: Patch k-NN inpaint
    router.register_tool(
        ToolCapability(
            tool_id="bellium/inpaint_knn",
            tier=EscalationTier.KNN_EXEMPLAR,
            modalities={"image"},
            tags={"inpaint", "fill", "texture"},
            latency_budget_ms=25.0,
            success_rate=0.92,
        )
    )

    # Tier 4: General large vision model
    router.register_tool(
        ToolCapability(
            tool_id="large/vision_diffusion",
            tier=EscalationTier.GENERAL_LARGE,
            modalities={"image"},
            tags={"generative", "inpaint", "complex_scene"},
            latency_budget_ms=1500.0,
            success_rate=0.95,
        )
    )

    # Test 1: Route cutout task -> should pick tier 0 deterministic
    req1 = TaskRequirement(
        task_id="req_01",
        modality="image",
        tags={"cutout", "background"},
    )
    v1 = router.route_task(req1)
    print("Test 1 - Cutout routing:", v1.selected_tool_id, v1.selected_tier, v1.confidence)
    assert v1.selected_tool_id == "bellium/cutout"
    assert v1.selected_tier == EscalationTier.DETERMINISTIC
    assert v1.confidence > 0.9

    # Test 2: Route inpaint with zero_tokens constraint -> should select tier 1 knn, NOT tier 4
    req2 = TaskRequirement(
        task_id="req_02",
        modality="image",
        tags={"inpaint"},
        hard_constraints={"zero_tokens"},
    )
    v2 = router.route_task(req2)
    print("Test 2 - Inpaint zero tokens:", v2.selected_tool_id, v2.selected_tier)
    assert v2.selected_tool_id == "bellium/inpaint_knn"
    assert v2.selected_tier == EscalationTier.KNN_EXEMPLAR

    # Test 3: Route task with strict latency budget 15ms -> inpaint_knn (25ms) and diffusion (1500ms) dropped
    req3 = TaskRequirement(
        task_id="req_03",
        modality="image",
        tags={"inpaint"},
        max_latency_ms=15.0,
    )
    v3 = router.route_task(req3)
    print("Test 3 - Strict latency budget:", v3.selected_tool_id, v3.selected_tier)
    # A fast cutout cannot perform inpainting. The corrected contract abstains.
    assert v3.selected_tool_id is None

    # Test 4: Modality mismatch -> abstention
    req4 = TaskRequirement(
        task_id="req_04",
        modality="audio",
        tags={"speech"},
    )
    v4 = router.route_task(req4)
    print("Test 4 - Audio task:", v4.selected_tool_id, v4.reason)
    assert v4.selected_tool_id is None
    assert v4.confidence == 0.0

    print("\nAll Bellium Routing tests PASSED successfully!")


if __name__ == "__main__":
    run_tests()
