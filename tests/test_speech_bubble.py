from bellium.knn.speech_bubble import load_bubbles
from bellium.specialists.speech_bubble import locate_bubbles


def _panel_with_bubble():
    image = [[(30, 50, 80) for _ in range(16)] for _ in range(16)]
    for r in range(2, 8):
        for c in range(4, 12):
            image[r][c] = (250, 250, 245)
    return image


def _dark_panel():
    return [[(20, 24, 30) for _ in range(16)] for _ in range(16)]


def test_bright_compact_region_is_a_bubble() -> None:
    result = locate_bubbles({"image": _panel_with_bubble()})
    assert result.abstained is False
    assert result.output["text"] is None
    assert result.output["certified"] is False
    assert result.output["regions"]
    box = result.output["regions"][0]
    assert box["label"] == "bubble"
    assert box["x"] <= 4 and box["y"] <= 2
    assert box["x"] + box["w"] >= 12 and box["y"] + box["h"] >= 8


def test_no_bright_region_abstains() -> None:
    result = locate_bubbles({"image": _dark_panel()})
    assert result.abstained is True
    assert result.output["regions"] == []
    assert result.output["text"] is None


def test_dark_provided_region_is_not_a_bubble() -> None:
    result = locate_bubbles({
        "image": _panel_with_bubble(),
        "regions": [{"x": 0, "y": 10, "w": 4, "h": 4}],
    })
    assert result.abstained is True
    assert result.output["reason"] == "no_bubble_label"


def test_dialogue_text_is_rejected() -> None:
    blocked = False
    try:
        locate_bubbles({"image": _panel_with_bubble(), "text": "hello"})
    except ValueError:
        blocked = True
    assert blocked


def test_missing_box_field_is_not_zeroed() -> None:
    blocked = False
    try:
        locate_bubbles({
            "image": _panel_with_bubble(),
            "regions": [{"x": 4, "y": 2, "w": 8}],
        })
    except ValueError as exc:
        blocked = "h" in str(exc)
    assert blocked


def test_memory_cannot_store_dialogue(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        '{"schema":"bellium.speech-bubble-memory/v1","items":[{'
        '"id":"x","label":"bubble","source":"x","text_stored":true,'
        '"features":{"fill":0.18,"aspect":0.75,"compactness":0.9,'
        '"brightness":0.94,"border_touch":0,"topness":0.3}}]}',
        encoding="utf-8",
    )
    blocked = False
    try:
        load_bubbles(path)
    except ValueError:
        blocked = True
    assert blocked

