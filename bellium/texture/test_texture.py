"""Texture repetition tests: periods, abstention, scaling and validation."""
import os
import math
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from PIL import Image

from bellium.texture import TextureError, detect_repeat


def _checkerboard(cell=8, size=64):
    image = Image.new("L", (size, size), 0)
    for y in range(size):
        for x in range(size):
            if ((x // cell) + (y // cell)) % 2 == 0:
                image.putpixel((x, y), 255)
    return image


def _stripes(period=10, size=(60, 40)):
    """Sine stripes: the image function has the exact requested period."""
    image = Image.new("L", size)
    image.putdata([
        int(round(127.5 + 127.5 * math.sin(2.0 * math.pi * x / period)))
        for _y in range(size[1])
        for x in range(size[0])
    ])
    return image


def _noise(size=(64, 64), seed=1234):
    rng = random.Random(seed)
    image = Image.new("L", size)
    image.putdata([rng.randrange(256) for _ in range(size[0] * size[1])])
    return image


def _gradient(size=(64, 64)):
    image = Image.new("L", size)
    image.putdata([int(255 * x / (size[0] - 1)) for _ in range(size[1]) for x in range(size[0])])
    return image


def _expect_error(function, label):
    try:
        function()
    except TextureError:
        return
    raise AssertionError(f"{label} did not raise TextureError")


def run_tests():
    # 1. Checkerboard: a single-axis shift by one cell inverts the pattern, so
    # the fundamental single-axis period is two cells (16 px for 8 px cells).
    board = _checkerboard(cell=8, size=64)
    before = board.tobytes()
    report = detect_repeat(board)
    assert report.reliable is True
    assert report.x.period_px == 16, report.x
    assert report.y.period_px == 16, report.y
    assert report.x.period_analysis == 16
    assert report.x.confidence >= 0.9
    assert report.analysis_size == (64, 64)
    assert report.scale_x == 1.0
    assert board.tobytes() == before
    print("Checkerboard periods: OK")

    # 2. Sine stripes repeat on x only; y abstains (rows are identical).
    stripes = _stripes(period=10, size=(60, 40))
    report = detect_repeat(stripes)
    assert report.reliable is True
    assert report.x.period_px == 10, report.x
    assert report.y.period_px is None, report.y
    assert report.y.reason == "no_reliable_period"
    assert report.edge_difference_y == 0.0
    print("Axis isolation: OK")

    # 3. Requesting only y on the same stripes reports no reliable period.
    report_y = detect_repeat(stripes, axis="y")
    assert report_y.reliable is False
    report_x = detect_repeat(stripes, axis="x")
    assert report_x.reliable is True and report_x.x.period_px == 10
    print("Requested axis semantics: OK")

    # 3b. A perfect period at the scan boundary is still accepted.
    tiny = _checkerboard(cell=1, size=32)
    tiny_report = detect_repeat(tiny)
    assert tiny_report.x.period_px == 2, tiny_report.x
    assert tiny_report.y.period_px == 2, tiny_report.y
    print("Boundary period: OK")

    # 4. Noise and smooth gradients abstain.
    noise = _noise()
    noise_report = detect_repeat(noise)
    assert noise_report.reliable is False, noise_report.to_dict()
    assert noise_report.x.period_px is None
    gradient = _gradient()
    gradient_report = detect_repeat(gradient)
    assert gradient_report.reliable is False, gradient_report.to_dict()
    flat = Image.new("L", (32, 32), 128)
    flat_report = detect_repeat(flat)
    assert flat_report.reliable is False
    assert flat_report.x.period_px is None and flat_report.y.period_px is None
    print("Abstention on noise and gradients: OK")

    # 5. A period outside the scanned range is not reported.
    wide = _stripes(period=24, size=(96, 40))
    limited = detect_repeat(wide, max_period=16)
    assert limited.reliable is False, limited.to_dict()
    assert limited.x.period_px is None
    print("Scan range bound: OK")

    # 6. Downscaled analysis maps the period back to source pixels.
    large = _checkerboard(cell=16, size=512)
    scaled = detect_repeat(large, max_analysis=256)
    assert scaled.analysis_size == (256, 256)
    assert scaled.scale_x == 0.5
    assert scaled.x.period_px is not None
    assert 30 <= scaled.x.period_px <= 34, scaled.x
    print("Downscaled analysis and pixel mapping: OK")

    # 7. Identical wrapped edges measure zero difference.
    edge_image = _noise(size=(32, 32), seed=99)
    pixels = edge_image.load()
    for y in range(32):
        pixels[31, y] = pixels[0, y]
    assert detect_repeat(edge_image).edge_difference_x == 0.0
    print("Wrapped edge difference: OK")

    # 8. Results are deterministic.
    assert detect_repeat(board).to_dict() == detect_repeat(board).to_dict()
    print("Determinism: OK")

    # 9. Invalid requests fail closed.
    _expect_error(lambda: detect_repeat(board, axis="z"), "unknown axis")
    _expect_error(lambda: detect_repeat(board, min_period=1), "min_period below 2")
    _expect_error(lambda: detect_repeat(board, min_period=8, max_period=4), "inverted range")
    _expect_error(lambda: detect_repeat(board, min_match=1.0), "min_match of 1.0")
    _expect_error(lambda: detect_repeat(board, min_contrast=1.5), "min_contrast above 1")
    _expect_error(lambda: detect_repeat(board, max_analysis=8), "max_analysis below 16")
    _expect_error(lambda: detect_repeat("not an image"), "non-image input")
    _expect_error(lambda: detect_repeat(Image.new("L", (2, 2))), "image below 4x4")
    print("Validation and fail-closed behavior: OK")

    print()
    print("All Bellium Texture tests PASSED!")


if __name__ == "__main__":
    run_tests()
