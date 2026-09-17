"""Interpolation quality, routing boundaries and irreversible-information limits."""
import unittest

from PIL import Image, ImageDraw

from bellium.inpaint import inpaint_patch_knn, inpaint_preview


def ramp(size=(32, 32), mode="RGB"):
    image = Image.new(mode, size)
    for y in range(size[1]):
        for x in range(size[0]):
            rgb = (20 + 3 * x + y, 180 - 2 * x + y, 40 + x + 2 * y)
            image.putpixel((x, y), rgb if mode == "RGB" else (*rgb, 128))
    return image


class PreviewTests(unittest.TestCase):
    def test_affine_rgb_and_rgba_with_irregular_mask_are_exact(self):
        for mode in ("RGB", "RGBA"):
            with self.subTest(mode=mode):
                reference = ramp(mode=mode)
                damaged = reference.copy()
                mask = Image.new("L", reference.size)
                for xy in ((10, 10), (11, 10), (10, 11), (12, 12)):
                    mask.putpixel(xy, 129)
                    damaged.putpixel(xy, (255, 0, 255) if mode == "RGB" else (255, 0, 255, 128))
                mask.putpixel((11, 11), 128)  # Threshold is strictly greater than 128.
                before, mask_before = damaged.tobytes(), mask.tobytes()
                result = inpaint_preview(damaged, mask)
                self.assertEqual(result.image.tobytes(), reference.tobytes())
                self.assertEqual(result.image.mode, mode)
                self.assertEqual(result.metrics.method, "bilinear_rgb_v1")
                self.assertEqual(result.metrics.filled_pixels, 4)
                self.assertEqual(result.metrics.interpolation.max_rgb_residual, 0)
                self.assertEqual(result.metrics.verdict.confidence, 0)
                self.assertEqual(damaged.tobytes(), before)
                self.assertEqual(mask.tobytes(), mask_before)

    def test_bilinear_field_and_masked_color_independence(self):
        reference = Image.new("RGB", (16, 16))
        for y in range(16):
            for x in range(16):
                reference.putpixel((x, y), (10 + x * y, 30 + 2 * x + y, 90 - x - y))
        mask = Image.new("L", reference.size)
        ImageDraw.Draw(mask).rectangle((6, 6, 8, 8), fill=255)
        for color in ((0, 255, 0), (255, 0, 255)):
            damaged = reference.copy()
            damaged.paste(color, (6, 6, 9, 9))
            result = inpaint_preview(damaged, mask)
            self.assertTrue(result.metrics.interpolation.accepted)
            self.assertEqual(result.image.tobytes(), reference.tobytes())

    def test_validation_checks_noncorner_pixels_and_fallback_is_unchanged(self):
        image = ramp()
        mask = Image.new("L", image.size)
        mask.putpixel((12, 12), 255)
        image.putpixel((12, 12), (255, 0, 255))
        image.putpixel((12, 11), (255, 255, 255))
        result = inpaint_preview(image, mask, search_radius=6)
        baseline = inpaint_patch_knn(image, mask, search_radius=6)
        self.assertEqual(result.metrics.interpolation.reason, "context_mismatch")
        self.assertGreater(result.metrics.interpolation.max_rgb_residual, 1)
        self.assertEqual(result.image.tobytes(), baseline.image.tobytes())
        self.assertEqual(result.metrics.verdict, baseline.metrics.verdict)
        self.assertEqual(result.metrics.comparisons, baseline.metrics.comparisons)

    def test_residual_gate_includes_one_code_value_but_rejects_two(self):
        for delta, accepted in ((1, True), (2, False)):
            with self.subTest(delta=delta):
                image = ramp()
                mask = Image.new("L", image.size)
                mask.putpixel((12, 12), 255)
                rgb = image.getpixel((12, 11))
                image.putpixel((12, 11), (rgb[0] + delta, rgb[1], rgb[2]))
                result = inpaint_preview(image, mask, max_comparisons=1)
                self.assertEqual(result.metrics.interpolation.accepted, accepted)
                self.assertEqual(result.metrics.interpolation.max_rgb_residual, delta)

    def test_border_spread_and_alpha_are_not_interpolation_support(self):
        for reason in ("missing_enclosing_context", "selection_span", "nonuniform_or_zero_alpha"):
            with self.subTest(reason=reason):
                image = ramp(mode="RGBA")
                mask = Image.new("L", image.size)
                if reason == "missing_enclosing_context":
                    mask.putpixel((0, 12), 255)
                elif reason == "selection_span":
                    mask.putpixel((4, 12), 255)
                    mask.putpixel((20, 12), 255)
                else:
                    mask.putpixel((12, 12), 255)
                    image.putpixel((12, 11), (90, 80, 70, 0))
                result = inpaint_preview(image, mask, max_comparisons=1)
                baseline = inpaint_patch_knn(image, mask, max_comparisons=1)
                self.assertEqual(result.metrics.interpolation.reason, reason)
                self.assertEqual(result.image.tobytes(), baseline.image.tobytes())
                self.assertEqual(result.metrics.filled_pixels, baseline.metrics.filled_pixels)

    def test_maximum_interpolation_rectangle_and_selection_gate(self):
        reference = ramp()
        mask = Image.new("L", reference.size)
        ImageDraw.Draw(mask).rectangle((8, 8, 23, 23), fill=255)
        damaged = reference.copy()
        damaged.paste((0, 0, 0), (8, 8, 24, 24))
        result = inpaint_preview(damaged, mask)
        self.assertEqual(result.image.tobytes(), reference.tobytes())
        self.assertEqual(result.metrics.filled_pixels, 256)
        self.assertEqual(result.metrics.interpolation.support_pixels, 144)
        self.assertEqual(result.metrics.interpolation.validation_pixels, 140)
        full = Image.new("L", reference.size, 255)
        declined = inpaint_preview(damaged, full)
        self.assertEqual(declined.metrics.interpolation.reason, "selection_limit")
        self.assertEqual(declined.metrics.filled_pixels, 0)
        self.assertEqual(declined.image.tobytes(), damaged.tobytes())

    def test_fallback_budget_still_rolls_back_the_entire_preview(self):
        image = Image.new("RGB", (20, 20))
        for y in range(20):
            for x in range(20):
                image.putpixel((x, y), (220, 100, 20) if (x + y) % 2 else (20, 80, 220))
        mask = Image.new("L", image.size)
        mask.putpixel((10, 10), 255)
        image.putpixel((10, 10), (255, 0, 255))
        result = inpaint_preview(image, mask, max_comparisons=1)
        self.assertFalse(result.metrics.interpolation.accepted)
        self.assertEqual(result.metrics.verdict.method, "escalate_diffusion")
        self.assertEqual(result.metrics.filled_pixels, 0)
        self.assertEqual(result.image.tobytes(), image.tobytes())

    def test_hidden_detail_is_not_provable_from_matching_context(self):
        image = Image.new("RGB", (20, 20), (50, 80, 100))
        mask = Image.new("L", image.size)
        mask.putpixel((10, 10), 255)
        # This unique detail is completely selected; the algorithm cannot infer it.
        image.putpixel((10, 10), (240, 30, 10))
        result = inpaint_preview(image, mask)
        self.assertTrue(result.metrics.interpolation.accepted)
        self.assertNotEqual(result.image.getpixel((10, 10)), image.getpixel((10, 10)))
        self.assertEqual(result.metrics.verdict.confidence, 0)
        self.assertIn("review required", result.metrics.verdict.reason)

    def test_identity_and_invisible_selection(self):
        image = Image.new("RGBA", (20, 20), (30, 60, 90, 0))
        mask = Image.new("L", image.size)
        identity = inpaint_preview(image, mask)
        self.assertEqual(identity.image.tobytes(), image.tobytes())
        self.assertEqual(identity.metrics.filled_pixels, 0)
        mask.putpixel((10, 10), 255)
        result = inpaint_preview(image, mask)
        self.assertEqual(result.metrics.interpolation.reason, "nonuniform_or_zero_alpha")
        self.assertEqual(result.metrics.filled_pixels, 0)
        self.assertEqual(result.image.tobytes(), image.tobytes())

    def test_invalid_options_cannot_bypass_validation_via_interpolation(self):
        image = ramp()
        mask = Image.new("L", image.size)
        mask.putpixel((10, 10), 255)
        for options in ({"patch_size": 2}, {"search_radius": 0}, {"k_neighbors": True},
                        {"max_comparisons": 0}, {"max_context_error": float("nan")},
                        {"max_context_error": True}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                inpaint_preview(image, mask, **options)
        with self.assertRaises(ValueError):
            inpaint_preview(image, Image.new("L", (3, 3)))
        with self.assertRaises(ValueError):
            inpaint_preview(image.convert("L"), mask)


if __name__ == "__main__":
    unittest.main()
