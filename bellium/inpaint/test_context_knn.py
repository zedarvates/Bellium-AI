"""Observable synthesis quality, support boundaries and pixel preservation."""
import random
import unittest

from PIL import Image, ImageDraw

from bellium.inpaint import inpaint_patch_knn


class ContextKNNTests(unittest.TestCase):
    def test_checkerboard_recovers_context_instead_of_nearest_pixel(self):
        image = Image.new("RGB", (24, 24))
        for y in range(24):
            for x in range(24):
                image.putpixel((x, y), (220, 180, 80) if (x + y) % 2 else (20, 50, 100))
        expected = image.getpixel((12, 12))
        image.putpixel((12, 12), (255, 0, 255))
        mask = Image.new("L", image.size)
        mask.putpixel((12, 12), 255)
        result = inpaint_patch_knn(image, mask, search_radius=7)
        self.assertEqual(result.metrics.filled_pixels, 1)
        self.assertEqual(result.image.getpixel((12, 12)), expected)
        self.assertNotEqual(result.image.getpixel((12, 12)), image.getpixel((12, 11)))
        for y in range(24):
            for x in range(24):
                if (x, y) != (12, 12):
                    self.assertEqual(result.image.getpixel((x, y)), image.getpixel((x, y)))

    def test_k_controls_donor_aggregation_and_alpha_is_preserved(self):
        image = Image.new("RGBA", (9, 11))
        draw = ImageDraw.Draw(image)
        for y, color in ((1, 10), (5, 60), (9, 240)):
            draw.rectangle((0, y - 1, 2, y + 1), fill=(100, 100, 100, 180))
            image.putpixel((1, y), (color, color, color, 180))
        draw.rectangle((6, 4, 8, 6), fill=(100, 100, 100, 180))
        image.putpixel((7, 5), (255, 0, 255, 180))
        mask = Image.new("L", image.size)
        mask.putpixel((7, 5), 255)
        one = inpaint_patch_knn(image, mask, patch_size=3, search_radius=8, k_neighbors=1, max_context_error=0)
        three = inpaint_patch_knn(image, mask, patch_size=3, search_radius=8, k_neighbors=3, max_context_error=0)
        self.assertEqual(one.image.getpixel((7, 5)), (60, 60, 60, 180))
        self.assertEqual(three.image.getpixel((7, 5)), (103, 103, 103, 180))
        self.assertEqual(three.image.getchannel("A").tobytes(), image.getchannel("A").tobytes())

    def test_budget_rolls_back_a_partially_computed_repair(self):
        image = Image.new("RGBA", (20, 20), (30, 60, 100, 128))
        mask = Image.new("L", image.size)
        for xy in ((5, 5), (14, 14)):
            mask.putpixel(xy, 255)
            image.putpixel(xy, (255, 0, 0, 128))
        result = inpaint_patch_knn(image, mask, patch_size=3, search_radius=3, max_comparisons=4)
        self.assertEqual(result.metrics.filled_pixels, 0)
        self.assertEqual(result.metrics.discarded_pixels, 1)
        self.assertEqual(result.metrics.comparisons, 4)
        self.assertEqual(result.metrics.verdict.method, "escalate_diffusion")
        self.assertEqual(result.image.tobytes(), image.tobytes())

    def test_masked_color_cannot_become_donor_or_matching_evidence(self):
        image = Image.new("RGB", (18, 18), (20, 90, 170))
        mask = Image.new("L", image.size)
        ImageDraw.Draw(mask).rectangle((7, 7, 10, 10), fill=255)
        other = image.copy()
        image.paste((255, 0, 0), (7, 7, 11, 11))
        other.paste((0, 255, 0), (7, 7, 11, 11))
        first = inpaint_patch_knn(image, mask, search_radius=7)
        second = inpaint_patch_knn(other, mask, search_radius=7)
        self.assertEqual(first.metrics.filled_pixels, 16)
        self.assertEqual(first.image.tobytes(), second.image.tobytes())

    def test_context_rejection_returns_unchanged_image(self):
        rng = random.Random(17)
        image = Image.new("RGB", (20, 20))
        image.putdata([tuple(rng.randrange(256) for _ in range(3)) for _ in range(400)])
        mask = Image.new("L", image.size)
        mask.putpixel((10, 10), 255)
        result = inpaint_patch_knn(image, mask, max_context_error=0)
        self.assertEqual(result.metrics.verdict.method, "escalate_diffusion")
        self.assertEqual(result.metrics.filled_pixels, 0)
        self.assertEqual(result.image.tobytes(), image.tobytes())

    def test_no_local_support_never_uses_random_global_fallback(self):
        image = Image.new("RGB", (20, 20), "blue")
        mask = Image.new("L", image.size)
        mask.putpixel((10, 10), 255)
        result = inpaint_patch_knn(image, mask, patch_size=5, search_radius=1)
        self.assertEqual(result.metrics.filled_pixels, 0)
        self.assertEqual(result.metrics.verdict.method, "escalate_diffusion")

    def test_zero_mask_and_transparent_selection(self):
        image = Image.new("RGBA", (10, 10), (30, 70, 100, 0))
        mask = Image.new("L", image.size)
        self.assertEqual(inpaint_patch_knn(image, mask).image.tobytes(), image.tobytes())
        mask.putpixel((5, 5), 255)
        result = inpaint_patch_knn(image, mask)
        self.assertEqual(result.metrics.filled_pixels, 0)
        self.assertEqual(result.image.tobytes(), image.tobytes())
        self.assertEqual(result.metrics.verdict.method, "escalate_diffusion")

    def test_invalid_work_budget_and_scores_are_rejected(self):
        image = Image.new("RGB", (10, 10))
        mask = Image.new("L", image.size)
        for options in ({"max_comparisons": 0}, {"max_comparisons": True},
                        {"max_context_error": float("nan")}, {"patch_size": 1},
                        {"search_radius": 65}, {"k_neighbors": 17}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                inpaint_patch_knn(image, mask, **options)


if __name__ == "__main__":
    unittest.main()
