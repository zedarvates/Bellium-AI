"""Deterministic image filter tests: strength, masks, alpha and thresholds."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from PIL import Image

from bellium.filters import (
    FilterError,
    apply_filter,
    brightness,
    contrast,
    invert,
    saturation,
    tint,
    to_binary,
    to_grayscale,
    to_sepia,
)


def _expect_error(expected, function, label):
    try:
        function()
    except expected:
        return
    raise AssertionError(f"{label} did not raise {expected.__name__}")


def run_tests():
    # 1. Grayscale: equal channels, alpha preserved, source untouched.
    source = Image.new("RGBA", (2, 1))
    source.putdata([(255, 0, 0, 255), (0, 128, 255, 64)])
    before = source.tobytes()
    gray = to_grayscale(source)
    assert gray.mode == "RGBA"
    assert gray.size == source.size
    assert source.tobytes() == before
    assert gray.getchannel("A").tobytes() == source.getchannel("A").tobytes()
    assert gray.getchannel("R").tobytes() == gray.getchannel("G").tobytes()
    assert gray.getchannel("G").tobytes() == gray.getchannel("B").tobytes()
    assert gray.getpixel((0, 0))[0] < 100  # pure red luma is dark
    print("Grayscale channels and alpha: OK")

    # 2. Strength 0 is exact identity; intermediate strength stays between.
    identity = to_sepia(source, strength=0.0)
    assert identity.tobytes() == before
    half = to_grayscale(source, strength=0.5)
    full = to_grayscale(source, strength=1.0)
    for point in ((0, 0), (1, 0)):
        ordered = sorted((source.getpixel(point)[0], half.getpixel(point)[0], full.getpixel(point)[0]))
        assert ordered[0] <= ordered[1] <= ordered[2]
    print("Strength blending: OK")

    # 3. Sepia duotone endpoints and custom tone.
    ramp = Image.new("RGB", (2, 1))
    ramp.putdata([(0, 0, 0), (255, 255, 255)])
    sepia = to_sepia(ramp)
    assert sepia.getpixel((0, 0)) == (32, 16, 8)
    assert sepia.getpixel((1, 0)) == (244, 223, 181)
    custom = to_sepia(ramp, light=(10, 200, 90))
    assert custom.getpixel((1, 0)) == (10, 200, 90)
    assert custom.getpixel((1, 0)) != sepia.getpixel((1, 0))
    print("Sepia tone endpoints: OK")

    # 4. Binary threshold: values at or above threshold become white.
    ramp = Image.new("RGB", (3, 1))
    ramp.putdata([(90, 90, 90), (120, 120, 120), (130, 130, 130)])
    binary = to_binary(ramp, threshold=120)
    assert binary.getpixel((0, 0)) == (0, 0, 0)
    assert binary.getpixel((1, 0)) == (255, 255, 255)
    assert binary.getpixel((2, 0)) == (255, 255, 255)
    assert to_binary(ramp, threshold=0).getpixel((0, 0)) == (255, 255, 255)
    print("Binary threshold: OK")

    # 4b. Invert, brightness, contrast, saturation and tint.
    rgb = Image.new("RGB", (1, 1), (60, 100, 180))
    assert invert(rgb).getpixel((0, 0)) == (195, 155, 75)
    assert invert(rgb, strength=0.0).getpixel((0, 0)) == (60, 100, 180)
    assert brightness(rgb, factor=0.0).getpixel((0, 0)) == (0, 0, 0)
    assert brightness(rgb, factor=1.0).getpixel((0, 0)) == (60, 100, 180)
    assert brightness(rgb, factor=2.0).getpixel((0, 0)) == (120, 200, 255)
    assert contrast(rgb, factor=0.0).getpixel((0, 0)) == (97, 97, 97)
    assert contrast(rgb, factor=1.0).getpixel((0, 0)) == (60, 100, 180)
    assert saturation(rgb, factor=0.0).getpixel((0, 0)) == (97, 97, 97)
    assert saturation(rgb, factor=1.0).getpixel((0, 0)) == (60, 100, 180)
    assert tint(rgb, color=(255, 0, 0)).getpixel((0, 0)) == (60, 0, 0)
    assert tint(rgb, color=(255, 255, 255)).getpixel((0, 0)) == (60, 100, 180)
    half_tint = tint(rgb, color=(255, 0, 0), strength=0.5).getpixel((0, 0))
    assert half_tint == (60, 50, 90)
    alpha_source = Image.new("RGBA", (1, 1), (60, 100, 180, 77))
    assert invert(alpha_source).getpixel((0, 0)) == (195, 155, 75, 77)
    print("Invert, brightness, contrast, saturation, tint: OK")

    # 5. Mask restricts the change and blends intermediate values.
    source = Image.new("RGBA", (3, 1))
    source.putdata([(200, 40, 40, 255), (200, 40, 40, 200), (200, 40, 40, 100)])
    full = to_grayscale(source)
    mask = Image.new("L", (3, 1))
    mask.putdata([0, 128, 255])
    masked = to_grayscale(source, mask=mask)
    assert masked.getpixel((0, 0)) == source.getpixel((0, 0))
    assert masked.getpixel((2, 0)) == full.getpixel((2, 0))
    left, middle, right = (masked.getpixel((x, 0))[0] for x in range(3))
    assert left >= middle >= right
    assert masked.getpixel((2, 0))[3] == 100
    print("Mask blending: OK")

    # 6. Dispatch works for every advertised name.
    assert apply_filter("grayscale", source).size == source.size
    assert apply_filter("sepia", source).size == source.size
    assert apply_filter("binary", source).size == source.size
    assert apply_filter("invert", source).size == source.size
    for name in ("brightness", "contrast", "saturation"):
        assert apply_filter(name, source, factor=1.0).size == source.size
    assert apply_filter("tint", source, color=(255, 255, 255)).size == source.size
    print("Filter dispatch: OK")

    # 7. Invalid requests fail closed without mutating the source.
    snapshot = source.tobytes()
    _expect_error(FilterError, lambda: to_grayscale(source, strength=float("nan")), "nan strength")
    _expect_error(FilterError, lambda: to_grayscale(source, strength=1.5), "strength above 1")
    _expect_error(FilterError, lambda: to_grayscale(source, strength=True), "boolean strength")
    _expect_error(FilterError, lambda: to_binary(source, threshold=256), "threshold 256")
    _expect_error(FilterError, lambda: to_binary(source, threshold=-1), "negative threshold")
    _expect_error(
        FilterError,
        lambda: to_grayscale(source, mask=Image.new("L", (5, 5))),
        "mask size mismatch",
    )
    _expect_error(
        FilterError,
        lambda: to_grayscale(source, mask=Image.new("RGB", source.size)),
        "rgb mask",
    )
    _expect_error(FilterError, lambda: apply_filter("posterize", source), "unknown filter")
    _expect_error(FilterError, lambda: to_sepia(source, light=(256, 0, 0)), "light out of range")
    _expect_error(FilterError, lambda: brightness(source, factor=-1.0), "negative factor")
    _expect_error(FilterError, lambda: contrast(source, factor=float("inf")), "infinite factor")
    _expect_error(FilterError, lambda: saturation(source, factor=200.0), "factor above 16")
    _expect_error(FilterError, lambda: tint(source, color=None), "tint without color")
    _expect_error(FilterError, lambda: tint(source, color=(0, 300, 0)), "tint out of range")
    assert source.tobytes() == snapshot
    print("Validation and fail-closed behavior: OK")

    print()
    print("All Bellium Filter tests PASSED!")


if __name__ == "__main__":
    run_tests()
