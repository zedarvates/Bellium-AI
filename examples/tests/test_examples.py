"""Integration checks for runnable examples and their observable boundaries."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image

from examples.micro_nn_triage import suggest_error_label
from examples.tool_routing import propose_tool
from examples.tools.image_tool import edit_image


class ImageToolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.png"
        self.output = self.root / "output.png"
        self.mask = self.root / "mask.png"
        image = Image.new("RGBA", (16, 16), (60, 100, 180, 128))
        image.putpixel((0, 0), (40, 50, 60, 0))
        image.save(self.source)
        self.before = self.source.read_bytes()

    def test_grayscale_preserves_source_alpha_and_dimensions(self):
        report = edit_image("grayscale", str(self.source), str(self.output))
        self.assertEqual(report["status"], "completed")
        self.assertEqual(self.source.read_bytes(), self.before)
        with Image.open(self.source) as source, Image.open(self.output) as output:
            self.assertEqual(source.size, output.size)
            self.assertEqual(source.getchannel("A").tobytes(), output.getchannel("A").tobytes())
            self.assertEqual(output.getchannel("R").tobytes(), output.getchannel("G").tobytes())
            self.assertEqual(output.getchannel("G").tobytes(), output.getchannel("B").tobytes())

    def test_binary_threshold_produces_two_levels_and_preserves_alpha(self):
        edit_image("binary", str(self.source), str(self.output), threshold=90)
        with Image.open(self.output) as output:
            self.assertEqual(output.getpixel((8, 8)), (255, 255, 255, 128))
            self.assertEqual(output.getpixel((0, 0)), (0, 0, 0, 0))
        self.output.unlink()
        edit_image("binary", str(self.source), str(self.output), threshold=110)
        with Image.open(self.output) as output:
            self.assertEqual(output.getpixel((8, 8)), (0, 0, 0, 128))

    def test_zero_strength_is_identity(self):
        edit_image("sepia", str(self.source), str(self.output), strength=0)
        with Image.open(self.source) as source, Image.open(self.output) as output:
            self.assertEqual(source.tobytes(), output.tobytes())

    def test_extended_filters_and_misuse_guard(self):
        edit_image("invert", str(self.source), str(self.output))
        with Image.open(self.output) as output:
            self.assertEqual(output.getpixel((8, 8)), (195, 155, 75, 128))
        self.output.unlink()
        edit_image("tint", str(self.source), str(self.output), color="#ff0000")
        with Image.open(self.output) as output:
            self.assertEqual(output.getpixel((8, 8)), (60, 0, 0, 128))
        self.output.unlink()
        with self.assertRaises(ValueError):
            edit_image("sepia", str(self.source), str(self.output), factor=2.0)
        with self.assertRaises(ValueError):
            edit_image("tint", str(self.source), str(self.output))
        self.assertFalse(self.output.exists())

    def test_filter_changes_only_selected_pixels(self):
        mask = Image.new("L", (16, 16), 0)
        mask.putpixel((8, 8), 255)
        mask.save(self.mask)
        edit_image("sepia", str(self.source), str(self.output), mask_path=str(self.mask))
        with Image.open(self.source) as source, Image.open(self.output) as output:
            self.assertNotEqual(source.getpixel((8, 8)), output.getpixel((8, 8)))
            for y in range(16):
                for x in range(16):
                    if (x, y) != (8, 8):
                        self.assertEqual(source.getpixel((x, y)), output.getpixel((x, y)))

    def test_refuses_overwrite(self):
        with self.assertRaises(FileExistsError):
            edit_image("sepia", str(self.source), str(self.source))
        self.assertEqual(self.source.read_bytes(), self.before)

    def test_invalid_mask_and_strength_write_nothing(self):
        Image.new("L", (2, 2), 255).save(self.mask)
        with self.assertRaises(ValueError):
            edit_image("inpaint", str(self.source), str(self.output), mask_path=str(self.mask))
        for strength in (float("nan"), float("inf"), -0.1, 1.1):
            with self.assertRaises(ValueError):
                edit_image("sepia", str(self.source), str(self.output), strength=strength)
        self.assertFalse(self.output.exists())

    def test_small_fill_preserves_outside_mask_and_alpha(self):
        mask = Image.new("L", (16, 16), 0)
        mask.putpixel((8, 8), 255)
        mask.save(self.mask)
        with Image.open(self.source) as source:
            damaged = source.copy()
        damaged.putpixel((8, 8), (255, 0, 0, 128))
        damaged.save(self.source)
        report = edit_image("inpaint", str(self.source), str(self.output),
                            mask_path=str(self.mask))
        self.assertTrue(report["review_required"])
        with Image.open(self.output) as output:
            self.assertEqual(output.getpixel((8, 8)), (60, 100, 180, 128))
            for y in range(16):
                for x in range(16):
                    if (x, y) != (8, 8):
                        self.assertEqual(damaged.getpixel((x, y)), output.getpixel((x, y)))

    def test_cli_abstention_does_not_write(self):
        Image.new("L", (16, 16), 255).save(self.mask)
        result = subprocess.run([
            sys.executable, "-m", "examples.tools.image_tool", "inpaint",
            "--input", str(self.source), "--mask", str(self.mask),
            "--output", str(self.output),
        ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "abstained")
        self.assertFalse(self.output.exists())
        self.assertEqual(self.source.read_bytes(), self.before)

    def test_cli_demo_runs_all_operations(self):
        target = self.root / "demo"
        result = subprocess.run([
            sys.executable, "-m", "examples.tools.image_tool", "demo",
            "--output-dir", str(target),
        ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        reports = json.loads(result.stdout)["results"]
        self.assertEqual(len(reports), 10)
        for report in reports:
            self.assertTrue(Path(report["output"]).is_file())
        with Image.open(target / "source.png") as source, Image.open(target / "inpaint.png") as fill:
            self.assertEqual(source.tobytes(), fill.convert("RGB").tobytes())


class AdvisoryTests(unittest.TestCase):
    def test_routing_handles_known_and_unknown_requests(self):
        self.assertEqual(propose_tool({"sepia", "filter"})["proposal"], "image.sepia")
        for tags, modality in (({"earth_gravity"}, "image"), ({"transcribe"}, "audio"),
                               (set(), "image")):
            report = propose_tool(tags, modality)
            self.assertIsNone(report["proposal"])
            self.assertFalse(report["executed"])

    def test_micro_nn_is_consultative_and_threshold_is_honored(self):
        report = suggest_error_label("FileNotFoundError: demo-input.txt does not exist")
        self.assertFalse(report["action_executed"])
        self.assertEqual(report["mode"], "consultative")
        self.assertAlmostEqual(sum(report["probabilities"]), 1.0, places=5)
        self.assertTrue(report["abstained"])
        self.assertIsNone(report["accepted_label"])


if __name__ == "__main__":
    unittest.main()
