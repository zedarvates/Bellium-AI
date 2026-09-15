"""Regression tests for observed false-success paths in asset preparation."""
import unittest

from PIL import Image, ImageDraw

from bellium.inpaint import inpaint_patch_knn
from bellium.pipeline import AssetPrepPipeline


class InpaintBoundaries(unittest.TestCase):
    def test_fully_masked_image_is_not_claimed_repaired(self):
        image = Image.new("RGBA", (20, 20), (30, 60, 90, 100))
        result = inpaint_patch_knn(image, Image.new("L", image.size, 255))
        self.assertEqual(result.metrics.filled_pixels, 0)
        self.assertEqual(result.metrics.verdict.method, "escalate_diffusion")
        self.assertEqual(result.image.tobytes(), image.tobytes())
        self.assertEqual(result.image.mode, image.mode)

    def test_no_exemplar_and_zero_mask_preserve_original(self):
        image = Image.new("RGBA", (3, 3), (30, 60, 90, 100))
        mask = Image.new("L", image.size, 0)
        identity = inpaint_patch_knn(image, mask)
        self.assertEqual(identity.image.tobytes(), image.tobytes())
        mask.putpixel((1, 1), 255)
        result = inpaint_patch_knn(image, mask, patch_size=5)
        self.assertEqual(result.metrics.filled_pixels, 0)
        self.assertAlmostEqual(result.metrics.mask_ratio, 1 / 9)
        self.assertEqual(result.metrics.verdict.method, "escalate_diffusion")

    def test_mismatched_masks_and_bad_parameters_are_rejected(self):
        image = Image.new("RGB", (10, 10))
        with self.assertRaises(ValueError):
            inpaint_patch_knn(image, Image.new("L", (9, 10)))
        for options in ({"patch_size": 2}, {"search_radius": 0}, {"k_neighbors": True}):
            with self.assertRaises(ValueError):
                inpaint_patch_knn(image, Image.new("L", image.size), **options)

    def test_pipeline_propagates_escalation_even_with_confident_cutout(self):
        image = Image.new("RGB", (50, 50), "white")
        ImageDraw.Draw(image).rectangle((10, 10, 40, 40), fill="blue")
        result = AssetPrepPipeline().process_asset(image, Image.new("L", image.size, 255))
        self.assertEqual(result.verdict, "escalate")
        self.assertFalse(result.inpaint_applied)
        self.assertEqual(result.inpaint_pixels_filled, 0)
        self.assertEqual(result.confidence, 0)


if __name__ == "__main__":
    unittest.main()
