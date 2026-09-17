"""Bellium AI unified command-line interface."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bellium",
        description="Bellium AI: Specialist micro-intelligence laboratory (micro-NN, k-NN, hybrid)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available specialist commands")
    
    # test command
    subparsers.add_parser("test", help="Run all 12 validation gates")
    
    # cutout command
    cutout_p = subparsers.add_parser("cutout", help="Extract foreground or normalize background")
    cutout_p.add_argument("input", help="Input image path")
    cutout_p.add_argument("-o", "--output", help="Output image path")
    cutout_p.add_argument("--bg", choices=["transparent", "white"], default="transparent")
    cutout_p.add_argument("--tolerance", type=float, default=30.0)
    
    # inspect-frame command
    vision_p = subparsers.add_parser("inspect-frame", help="Scan camera frame for visual anomalies and blackouts")
    vision_p.add_argument("input", help="Input frame image path")
    
    # route command
    route_p = subparsers.add_parser("route", help="Query the tiered hybrid router for optimal tool tier")
    route_p.add_argument("--modality", default="image", choices=["image", "text", "audio"])
    route_p.add_argument("--tags", nargs="+", required=True, help="Task tags (e.g. cutout, inpaint)")
    route_p.add_argument("--zero-tokens", action="store_true", help="Enforce zero-token execution")
    
    parsed = parser.parse_args(args)
    
    if parsed.command == "test":
        import subprocess
        root = Path(__file__).resolve().parent.parent
        runner = root / "run_all_tests.py"
        if runner.exists():
            return subprocess.run([sys.executable, str(runner)]).returncode
        else:
            print("run_all_tests.py not found in repo root.")
            return 1
            
    elif parsed.command == "cutout":
        from PIL import Image
        from bellium.cutout import extract_foreground, normalize_background
        src = Image.open(parsed.input)
        if parsed.bg == "transparent":
            res = extract_foreground(src, tolerance=parsed.tolerance)
            out = res.image
            m = res.metrics
            print(f"[Cutout] recommendation: {m.recommendation}, coverage: {m.coverage_ratio:.1%}, conf: {m.confidence}")
        else:
            out = normalize_background(src)
            print("[Cutout] Normalized background to solid white")
        if parsed.output:
            out.save(parsed.output)
            print(f"Saved result to {parsed.output}")
        return 0
        
    elif parsed.command == "inspect-frame":
        from PIL import Image
        from bellium.vision import detect_visual_anomalies
        src = Image.open(parsed.input)
        rep = detect_visual_anomalies(src)
        print(f"[Vision] Status: {rep.severity} (anomaly_score: {rep.anomaly_score})")
        if rep.reasons:
            for r in rep.reasons:
                print(f"  - {r}")
        return 0 if rep.severity != 'critical_emergency' else 2
        
    elif parsed.command == "route":
        from bellium.routing import HybridRouter, ToolCapability, TaskRequirement, EscalationTier
        router = HybridRouter()
        router.register_tool(ToolCapability('bellium/cutout', EscalationTier.DETERMINISTIC, {'image'}, {'cutout', 'alpha'}))
        router.register_tool(ToolCapability('bellium/inpaint_knn', EscalationTier.KNN_EXEMPLAR, {'image'}, {'inpaint', 'fill'}))
        router.register_tool(ToolCapability('large/vision_diffusion', EscalationTier.GENERAL_LARGE, {'image'}, {'generative', 'inpaint'}))
        
        constraints = set()
        if parsed.zero_tokens:
            constraints.add('zero_tokens')
        req = TaskRequirement('cli_query', parsed.modality, set(parsed.tags), hard_constraints=constraints)
        v = router.route_task(req)
        print(f"[Route] selected_tool: {v.selected_tool_id}")
        print(f"[Route] selected_tier: {v.selected_tier.value} (conf: {v.confidence})")
        print(f"[Route] reason: {v.reason}")
        return 0
        
    else:
        parser.print_help()
        return 0


if __name__ == '__main__':
    sys.exit(main())
