"""Bellium AI unified command-line interface."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(args: list[str] | None = None) -> int:
    from bellium.filters import FILTER_NAMES

    parser = argparse.ArgumentParser(
        prog="bellium",
        description="Bellium AI: Specialist micro-intelligence laboratory (micro-NN, k-NN, hybrid)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available specialist commands")
    
    # test command
    subparsers.add_parser("test", help="Run all validation gates")
    
    # cutout command
    cutout_p = subparsers.add_parser("cutout", help="Extract foreground or normalize background")
    cutout_p.add_argument("input", help="Input image path")
    cutout_p.add_argument("-o", "--output", help="Output image path")
    cutout_p.add_argument("--bg", choices=["transparent", "white"], default="transparent")
    cutout_p.add_argument("--tolerance", type=float, default=30.0)

    # filter command
    filter_p = subparsers.add_parser(
        "filter", help="Apply a deterministic filter: grayscale, sepia, binary, invert, brightness, contrast, saturation or tint"
    )
    filter_p.add_argument("name", choices=list(FILTER_NAMES))
    filter_p.add_argument("input", help="Input image path")
    filter_p.add_argument("-o", "--output", required=True, help="Output image path")
    filter_p.add_argument("--strength", type=float, default=1.0, help="Blend 0.0-1.0 (default 1.0)")
    filter_p.add_argument("--mask", help="Optional mode L/1 mask with identical dimensions")
    filter_p.add_argument("--threshold", type=int, default=128, help="Binary threshold 0-255")
    filter_p.add_argument("--factor", type=float, default=1.0, help="Factor 0-16 for brightness, contrast, saturation")
    filter_p.add_argument("--color", help="Tint color such as '#88ccff' (tint only)")

    # texture command
    texture_p = subparsers.add_parser(
        "texture", help="Detect X/Y texture repetition with confidence and abstention"
    )
    texture_p.add_argument("input", help="Input image path")
    texture_p.add_argument("--axis", choices=["both", "x", "y"], default="both")
    texture_p.add_argument("--min-period", type=int, default=2)
    texture_p.add_argument("--max-period", type=int)
    texture_p.add_argument("--min-match", type=float, default=0.8)
    texture_p.add_argument("--min-contrast", type=float, default=0.05)
    texture_p.add_argument("--max-analysis", type=int, default=256)
    
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
        
    elif parsed.command == "filter":
        from PIL import Image
        from bellium.filters import apply_filter
        source = Path(parsed.input)
        target = Path(parsed.output)
        if source.resolve() == target.resolve():
            print("[Filter] Refusing to overwrite the source image; choose another output path.")
            return 2
        try:
            with Image.open(source) as opened:
                image = opened.copy()
            mask = None
            if parsed.mask:
                with Image.open(parsed.mask) as opened_mask:
                    mask = opened_mask.copy()
            options = {"strength": parsed.strength, "mask": mask}
            if parsed.name == "binary":
                options["threshold"] = parsed.threshold
            elif parsed.name in ("brightness", "contrast", "saturation"):
                options["factor"] = parsed.factor
            elif parsed.name == "tint":
                if not parsed.color:
                    print("[Filter] tint requires --color, for example --color '#88ccff'.")
                    return 2
                from PIL import ImageColor
                options["color"] = ImageColor.getrgb(parsed.color)
            if parsed.name != "binary" and parsed.threshold != 128:
                print("[Filter] --threshold applies only to binary.")
                return 2
            if parsed.name not in ("brightness", "contrast", "saturation") and parsed.factor != 1.0:
                print("[Filter] --factor applies only to brightness, contrast and saturation.")
                return 2
            if parsed.name != "tint" and parsed.color:
                print("[Filter] --color applies only to tint.")
                return 2
            result = apply_filter(parsed.name, image, **options)
        except (OSError, ValueError) as exc:
            print(f"[Filter] {exc}")
            return 2
        target.parent.mkdir(parents=True, exist_ok=True)
        result.save(target)
        print(f"[Filter] {parsed.name} applied (strength={parsed.strength}); saved to {target}")
        return 0

    elif parsed.command == "texture":
        from PIL import Image
        from bellium.texture import detect_repeat
        try:
            with Image.open(parsed.input) as opened:
                image = opened.copy()
            report = detect_repeat(
                image,
                axis=parsed.axis,
                min_period=parsed.min_period,
                max_period=parsed.max_period,
                min_match=parsed.min_match,
                min_contrast=parsed.min_contrast,
                max_analysis=parsed.max_analysis,
            )
        except (OSError, ValueError) as exc:
            print(f"[Texture] {exc}")
            return 1
        for item in (report.x, report.y):
            if item.period_px is None:
                print(f"[Texture] {item.axis}: no reliable period (match={item.match}, margin={item.margin})")
            else:
                print(
                    f"[Texture] {item.axis}: period={item.period_px} px "
                    f"(analysis {item.period_analysis}, conf={item.confidence}, "
                    f"match={item.match}, margin={item.margin})"
                )
        print(
            f"[Texture] edge difference x={report.edge_difference_x} "
            f"y={report.edge_difference_y} (0 identical, 1 maximally different)"
        )
        return 0 if report.reliable else 2

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
