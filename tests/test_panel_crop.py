from bellium.knn.panel_crop import load_layouts, propose_margin
from bellium.specialists.panel_crop import crop_panel


def _panel():
    image = [[(20, 40, 180) for _ in range(16)] for _ in range(16)]
    for r in range(4, 12):
        for c in range(4, 12):
            image[r][c] = (200, 40, 40)
    return image


def test_centered_panel_crops_without_clipping_content() -> None:
    result = crop_panel({"family": "storycore-panel", "image": _panel()})
    assert result.abstained is False
    assert result.output["status"] == "safe"
    assert result.output["certified"] is False
    crop = result.output["crop"]
    assert crop["x"] <= 4 and crop["y"] <= 4
    assert crop["x"] + crop["w"] >= 12 and crop["y"] + crop["h"] >= 12
    sliced = result.output["image"]
    reds = [pixel for row in sliced for pixel in row if pixel[0] > 180]
    assert reds


def test_required_box_on_the_edge_is_unsafe() -> None:
    result = crop_panel({
        "family": "storycore-panel",
        "image": _panel(),
        "content_boxes": [{"x": 0, "y": 4, "w": 4, "h": 4}],
    })
    assert result.output["status"] == "unsafe"
    assert result.output["reason"] == "required_content_already_clipped"


def test_explicit_inner_box_is_kept() -> None:
    result = crop_panel({
        "family": "storycore-panel",
        "image": _panel(),
        "content_boxes": [{"x": 4, "y": 4, "w": 8, "h": 8}],
    })
    assert result.output["status"] == "safe"
    crop = result.output["crop"]
    assert crop["x"] <= 4 <= crop["x"] + crop["w"] - 8
    assert crop["y"] <= 4 <= crop["y"] + crop["h"] - 8


def test_unknown_family_abstains() -> None:
    result = crop_panel({"family": "manga-unknown", "image": _panel()})
    assert result.abstained is True


def test_missing_box_field_is_not_zeroed() -> None:
    blocked = False
    try:
        crop_panel({
            "family": "storycore-panel",
            "image": _panel(),
            "content_boxes": [{"x": 4, "y": 4, "w": 8}],
        })
    except ValueError as exc:
        blocked = "h" in str(exc)
    assert blocked


def test_layout_knn_is_family_isolated() -> None:
    features = {"fill": 0.30, "aspect": 0.95, "touch_left": 0, "touch_right": 0, "touch_top": 0, "touch_bottom": 0}
    sprite = propose_margin({"family": "sprite", "features": features})
    panel = propose_margin({"family": "storycore-panel", "features": features})
    assert sprite.output["margin"] < panel.output["margin"]


def test_raw_pixels_cannot_be_kept(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        '{"schema":"bellium.panel-crop-memory/v1","items":[{'
        '"id":"x","family":"sprite","verdict":"safe","margin":0.05,'
        '"source":"x","raw_image_stored":true,'
        '"features":{"fill":0.3,"aspect":1,"touch_left":0,"touch_right":0,"touch_top":0,"touch_bottom":0}}]}',
        encoding="utf-8",
    )
    blocked = False
    try:
        load_layouts(path)
    except ValueError:
        blocked = True
    assert blocked

