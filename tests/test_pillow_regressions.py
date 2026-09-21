import pytest
import random

pytest.importorskip("PIL")
from PIL import Image, ImageDraw

from bellium.cutout import extract_foreground, normalize_background, UnsafeCutoutError
from bellium.inpaint import inpaint_patch_knn


@pytest.mark.parametrize("alpha", [0, 100])
def test_hidden_or_translucent_pixels_do_not_become_opaque(alpha):
    image = Image.new("RGBA", (20, 20), (255, 255, 255, 0))
    ImageDraw.Draw(image).rectangle((6, 6, 13, 13), fill=(255, 0, 0, alpha))
    result = extract_foreground(image)
    assert result.image.getpixel((10, 10))[3] == alpha
    assert result.image.getpixel((0, 0))[3] == 0


def test_feather_and_padding_parameters_change_the_result():
    image = Image.new("RGB", (100, 100), "white")
    ImageDraw.Draw(image).rectangle((25, 25, 74, 74), fill="red")
    assert extract_foreground(image, feather_radius=0).mask.tobytes() != extract_foreground(image, feather_radius=3).mask.tobytes()
    small_margin = normalize_background(image, padding_ratio=0.05)
    large_margin = normalize_background(image, padding_ratio=0.30)
    assert small_margin.tobytes() != large_margin.tobytes()
    assert small_margin.getpixel((20, 50))[0:2] == (255, 0)
    assert large_margin.getpixel((20, 50)) == (255, 255, 255)


def test_normalization_propagates_an_unsafe_extraction():
    with pytest.raises(UnsafeCutoutError) as exc:
        normalize_background(Image.new("RGB", (20, 20), (120, 120, 120)))
    assert exc.value.metrics.recommendation == "escalate"


def test_full_mask_reports_zero_applied_pixels():
    image = Image.new("RGB", (20, 20), "red")
    result = inpaint_patch_knn(image, Image.new("L", (20, 20), 255))
    assert result.metrics.filled_pixels == 0
    assert result.metrics.verdict.method == "escalate_diffusion"
    assert result.image.tobytes() == image.tobytes()


def test_pillow_inpaint_rejects_wrong_mask_dimensions():
    with pytest.raises(ValueError, match="mask"):
        inpaint_patch_knn(Image.new("RGB", (20, 20)), Image.new("L", (21, 20)))


def test_pillow_adapter_keeps_image_when_context_checks_fail():
    rng = random.Random(831)
    image = Image.new("RGB", (32, 32))
    image.putdata([tuple(rng.randrange(256) for _ in range(3)) for _ in range(32*32)])
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rectangle((15, 15, 17, 17), fill=255)
    result = inpaint_patch_knn(image, mask, patch_size=3, search_radius=6, k_neighbors=5)
    assert result.image.tobytes() == image.tobytes()
    assert result.metrics.filled_pixels == 0
    assert result.metrics.verdict.method == "escalate_diffusion"
    assert result.metrics.quality["passed"] is False
