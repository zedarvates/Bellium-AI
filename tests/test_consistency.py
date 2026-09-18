from bellium.knn.consistency import load_styles, retrieve_consistency


def _solid(color, jitter=1):
    image = []
    for r in range(12):
        row = []
        for c in range(12):
            pixel = []
            for i, ch in enumerate(color):
                delta = ((r * 3 + c * 5 + i * jitter) % 7) - 3
                pixel.append(max(0, min(255, ch + delta)))
            row.append(tuple(pixel))
        image.append(row)
    return image


def test_blue_panel_retrieves_abyss_style() -> None:
    result = retrieve_consistency({
        "family": "storycore-panel",
        "image": _solid((20, 44, 170)),
    })
    assert result.abstained is False
    assert result.output["style_id"] == "abyss-blue"
    assert result.output["recolor"] is False
    assert result.output["certified"] is False


def test_warm_panel_does_not_collapse_into_blue() -> None:
    result = retrieve_consistency({
        "family": "storycore-panel",
        "image": _solid((180, 90, 38)),
    })
    assert result.output["style_id"] == "warm-flat"


def test_families_do_not_share_styles() -> None:
    result = retrieve_consistency({
        "family": "sprite",
        "image": _solid((20, 44, 170)),
    })
    assert result.abstained is True
    gray = retrieve_consistency({
        "family": "sprite",
        "image": _solid((120, 122, 126)),
    })
    assert gray.output["style_id"] == "cool-gray"


def test_unknown_family_abstains() -> None:
    result = retrieve_consistency({
        "family": "photo-unknown",
        "image": _solid((20, 44, 170)),
    })
    assert result.abstained is True
    assert result.output["recolor"] is False


def test_mixed_style_neighbors_abstain() -> None:
    memory = load_styles()
    panel = [item for item in memory if item["family"] == "storycore-panel"]
    panel[0]["style_id"] = "warm-flat"
    result = retrieve_consistency({
        "family": "storycore-panel",
        "image": _solid((20, 44, 170)),
    }, memory=panel)
    assert result.abstained is True
    assert result.output["reason"] == "mixed_neighbor_styles"


def test_raw_pixels_cannot_be_kept(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        '{"schema":"bellium.consistency-memory/v1","items":[{'
        '"id":"x","family":"sprite","style_id":"cool-gray","source":"x",'
        '"raw_image_stored":true,"features":{"mean_r":0.4,"mean_g":0.4,'
        '"mean_b":0.4,"luma_std":0.1,"dark_ratio":0.0,"border_delta":0.0,'
        '"chroma":0.05}}]}',
        encoding="utf-8",
    )
    blocked = False
    try:
        load_styles(path)
    except ValueError:
        blocked = True
    assert blocked

