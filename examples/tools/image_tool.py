"""Local image-tool example: Bellium filters and bounded previews."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from io import BytesIO
import json
import math
from pathlib import Path

from PIL import Image, ImageColor, ImageDraw

from bellium.cutout import extract_foreground
from bellium.filters import apply_filter
from bellium.inpaint import inpaint_patch_knn, route_inpaint_request


OPERATIONS = (
    "grayscale",
    "sepia",
    "binary",
    "invert",
    "brightness",
    "contrast",
    "saturation",
    "tint",
    "cutout",
    "inpaint",
)
FILTER_OPERATIONS = frozenset(
    {"grayscale", "sepia", "binary", "invert", "brightness", "contrast", "saturation", "tint"}
)
FACTOR_OPERATIONS = frozenset({"brightness", "contrast", "saturation"})
MAX_PIXELS = 1_048_576
MAX_INPAINT_PIXELS = 128 * 128
MAX_MASK_RATIO = 0.05


def load_image(path: str | Path, mode: str) -> Image.Image:
    with Image.open(path) as image:
        if image.width * image.height > MAX_PIXELS:
            raise ValueError("This example accepts at most 1,048,576 pixels.")
        return image.convert(mode)


def save_new_png(image: Image.Image, path: str | Path) -> None:
    """Exclusive creation prevents overwriting a source or an earlier result."""
    target = Path(path)
    if target.suffix.lower() != ".png":
        raise ValueError("Choose a new .png output path.")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(buffer.getvalue())


def edit_image(
    operation: str,
    input_path: str,
    output_path: str,
    *,
    mask_path: str | None = None,
    strength: float = 1.0,
    threshold: int = 128,
    factor: float = 1.0,
    color: str | None = None,
) -> dict:
    """Callable tool adapter. Returns metadata; never starts a remote model."""
    if operation not in OPERATIONS:
        raise ValueError(f"Unknown operation: {operation}")
    if not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("strength must be finite and between 0 and 1.")
    target = Path(output_path)
    if target.suffix.lower() != ".png":
        raise ValueError("Choose a new .png output path.")
    if target.exists():
        raise FileExistsError(f"Output already exists: {target}")
    if operation not in FILTER_OPERATIONS and strength != 1:
        raise ValueError("strength is supported only for filter operations.")
    if operation not in FACTOR_OPERATIONS and factor != 1.0:
        raise ValueError("factor is supported only for brightness, contrast and saturation.")
    if operation != "tint" and color is not None:
        raise ValueError("color is supported only for tint.")
    if operation == "tint" and color is None:
        raise ValueError("tint requires a color such as '#88ccff'.")
    if operation == "cutout" and mask_path is not None:
        raise ValueError("cutout estimates its own mask; omit mask_path.")
    if operation == "inpaint" and mask_path is None:
        raise ValueError("inpaint requires a mask_path.")

    original = load_image(input_path, "RGBA")
    mask = load_image(mask_path, "L") if mask_path is not None else None
    if mask is not None and mask.size != original.size:
        raise ValueError("The mask and input image must have identical dimensions.")
    report = {
        "operation": operation,
        "status": "completed",
        "output": None,
        "review_required": False,
    }

    if operation in FILTER_OPERATIONS:
        options = {"strength": strength, "mask": mask}
        if operation == "binary":
            options["threshold"] = threshold
        if operation in FACTOR_OPERATIONS:
            options["factor"] = factor
        if operation == "tint":
            options["color"] = ImageColor.getrgb(color)
        output = apply_filter(operation, original, **options)
        report["method"] = "bellium_filters"
    elif operation == "cutout":
        result = extract_foreground(original)
        report.update(method="bellium_cutout", metrics=asdict(result.metrics),
                      review_required=True)
        if result.metrics.recommendation != "confident":
            report.update(status="abstained", reason="Cutout needs review or escalation.")
            return report
        # The published cutout replaces alpha: combine it with existing transparency.
        alpha = Image.composite(original.getchannel("A"), Image.new("L", original.size),
                                result.mask)
        output = result.image.copy()
        output.putalpha(alpha)
        report["status"] = "candidate_written"
    else:
        if original.width * original.height > MAX_INPAINT_PIXELS:
            raise ValueError("The inpaint preview is limited to 16,384 pixels.")
        binary_mask = mask.point(lambda value: 255 if value > 128 else 0)
        verdict = route_inpaint_request(binary_mask, max_local_ratio=MAX_MASK_RATIO)
        report.update(method="bellium_inpaint_preview", route=asdict(verdict),
                      review_required=True)
        # Routing is advisory in the published core. This adapter enforces it BEFORE filling.
        if verdict.method != "patch_knn" or verdict.mask_ratio > MAX_MASK_RATIO:
            report.update(status="abstained", reason="Mask exceeds the 5% preview limit.")
            return report
        result = inpaint_patch_knn(original, binary_mask, patch_size=3, search_radius=8)
        output = Image.composite(result.image.convert("RGBA"), original, binary_mask)
        output.putalpha(original.getchannel("A"))
        report.update(status="candidate_written", metrics=asdict(result.metrics))

    save_new_png(output, output_path)
    report["output"] = str(target)
    return report


def create_demo(output_dir: str) -> dict:
    """Create original synthetic fixtures without downloading any assets."""
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=False)
    source = Image.new("RGB", (64, 64), "white")
    ImageDraw.Draw(source).rectangle((16, 12, 47, 51), fill=(30, 100, 180))
    mask = Image.new("L", source.size, 0)
    ImageDraw.Draw(mask).rectangle((28, 28, 30, 30), fill=255)
    damaged = source.copy()
    damaged.paste((220, 30, 90), (28, 28, 31, 31))
    for name, image in (("source", source), ("damaged", damaged), ("mask", mask)):
        save_new_png(image, target / f"{name}.png")
    reports = [edit_image(op, str(target / "source.png"), str(target / f"{op}.png"))
               for op in ("grayscale", "sepia", "binary", "invert", "brightness",
                          "contrast", "saturation")]
    reports.append(edit_image("tint", str(target / "source.png"), str(target / "tint.png"),
                              color="#88ccff", strength=0.5))
    reports.append(edit_image("cutout", str(target / "source.png"), str(target / "cutout.png")))
    reports.append(edit_image("inpaint", str(target / "damaged.png"),
                              str(target / "inpaint.png"), mask_path=str(target / "mask.png")))
    return {"fixture": "synthetic_demo_not_a_quality_benchmark", "results": reports}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    demo = sub.add_parser("demo", help="Generate synthetic images and run all operations")
    demo.add_argument("--output-dir", required=True)
    for operation in OPERATIONS:
        command = sub.add_parser(operation)
        command.add_argument("--input", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--mask")
        command.add_argument("--strength", type=float, default=1.0)
        command.add_argument("--threshold", type=int, default=128)
        command.add_argument("--factor", type=float, default=1.0)
        command.add_argument("--color")
    args = parser.parse_args(argv)
    try:
        if args.operation == "demo":
            report = create_demo(args.output_dir)
        else:
            report = edit_image(args.operation, args.input, args.output,
                                mask_path=args.mask, strength=args.strength,
                                threshold=args.threshold, factor=args.factor,
                                color=args.color)
    except (OSError, ValueError) as exc:
        print(json.dumps({"status": "error", "reason": str(exc)}))
        return 1
    print(json.dumps(report, indent=2))
    return 2 if report.get("status") == "abstained" else 0


if __name__ == "__main__":
    raise SystemExit(main())
