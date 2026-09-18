from bellium.knn.color_cutout import cutout
from bellium.knn.patch_inpaint import inpaint
from bellium.specialists.inpaint_router import route_inpaint
from bellium.specialists.white_background import normalize_white_background


def _solid(h: int, w: int, color: tuple[int, int, int]):
    return [[color for _ in range(w)] for _ in range(h)]


def test_patch_knn_fills_small_uniform_hole() -> None:
    image = _solid(12, 12, (40, 80, 120))
    mask = [[0] * 12 for _ in range(12)]
    for r in range(5, 7):
        for c in range(5, 7):
            image[r][c] = (0, 0, 0)
            mask[r][c] = 1
    result = inpaint(image, mask)
    assert result.abstained is False
    filled = result.output["image"]
    assert filled[5][5] == (40, 80, 120)
    assert filled[6][6] == (40, 80, 120)


def test_patch_knn_abstains_on_huge_hole() -> None:
    image = _solid(8, 8, (10, 10, 10))
    mask = [[1] * 8 for _ in range(8)]
    result = inpaint(image, mask)
    assert result.abstained is True
    assert result.output["reason"] == "hole_too_large"


def test_inpaint_router_prefers_patch_for_tiny_clean_hole() -> None:
    image = _solid(16, 16, (200, 200, 200))
    mask = [[0] * 16 for _ in range(16)]
    mask[8][8] = 1
    result = route_inpaint(image, mask)
    assert result.abstained is False
    assert result.output["label"] == "patch_knn"


def test_inpaint_router_escalates_large_busy_hole() -> None:
    image = _solid(16, 16, (0, 0, 0))
    for r in range(16):
        for c in range(16):
            image[r][c] = ((r * 17) % 255, (c * 13) % 255, 40)
    mask = [[1 if r < 10 and c < 10 else 0 for c in range(16)] for r in range(16)]
    result = route_inpaint(image, mask)
    assert result.output["label"] == "escalate"


def test_color_cutout_and_white_background() -> None:
    image = _solid(10, 10, (12, 24, 200))
    for r in range(3, 8):
        for c in range(3, 8):
            image[r][c] = (220, 40, 40)
    cut = cutout(image)
    assert cut.abstained is False
    assert cut.output["alpha"][5][5] == 1.0
    assert cut.output["alpha"][0][0] == 0.0
    white = normalize_white_background(image)
    assert white.abstained is False
    painted = white.output["image"]
    assert painted[0][0] == (255, 255, 255)
    # cropped object remains red
    reds = [pixel for row in painted for pixel in row if pixel[0] > 180]
    assert reds

