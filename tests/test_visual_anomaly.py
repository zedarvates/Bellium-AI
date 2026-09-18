from copy import deepcopy

from bellium.knn.visual_anomaly import assess_visual, extract_features, load_exemplars


def _solid(color, size=12, jitter=1):
    image = []
    for row in range(size):
        line = []
        for col in range(size):
            pixel = []
            for index, channel in enumerate(color):
                delta = ((row * 3 + col * 5 + index * jitter) % 7) - 3
                pixel.append(max(0, min(255, channel + delta)))
            line.append(tuple(pixel))
        image.append(line)
    return image


def test_known_tank_water_is_normal() -> None:
    result = assess_visual({
        "domain": "aquaponics-lab",
        "image": _solid((20, 84, 178)),
    })
    assert result.abstained is False
    assert result.output["label"] == "normal"
    assert result.output["in_known_space"] is True


def test_red_frame_is_novel_in_the_tank_domain() -> None:
    result = assess_visual({
        "domain": "aquaponics-lab",
        "image": _solid((190, 24, 18)),
    })
    assert result.abstained is False
    assert result.output["label"] == "novel"


def test_domains_do_not_share_visual_memory() -> None:
    tank = assess_visual({
        "domain": "asset-lab",
        "image": _solid((20, 84, 178)),
    })
    assert tank.output["label"] == "novel" or tank.abstained is True
    texture = assess_visual({
        "domain": "asset-lab",
        "image": _solid((120, 120, 124)),
    })
    assert texture.abstained is False
    assert texture.output["label"] == "normal"


def test_unknown_domain_abstains_instead_of_raising_an_alarm() -> None:
    result = assess_visual({
        "domain": "unseen-camera",
        "image": _solid((20, 84, 178)),
    })
    assert result.abstained is True
    assert result.output["reason"] == "unknown_visual_space"
    assert result.output["label"] is None


def test_missing_visual_feature_is_not_filled_with_zero() -> None:
    features = extract_features(_solid((20, 84, 178)))
    del features["mean_b"]
    blocked = False
    try:
        assess_visual({"domain": "aquaponics-lab", "features": features})
    except ValueError as exc:
        blocked = "mean_b" in str(exc)
    assert blocked


def test_mixed_labels_abstain() -> None:
    memory = deepcopy(load_exemplars())
    aqua = [item for item in memory if item["domain"] == "aquaponics-lab"]
    aqua[0]["label"] = "novel"
    result = assess_visual({
        "domain": "aquaponics-lab",
        "image": _solid((20, 84, 178)),
    }, memory=aqua)
    assert result.abstained is True
    assert result.output["reason"] == "mixed_visual_labels"


def test_raw_pixels_cannot_be_kept_in_memory(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        '{"schema":"bellium.visual-anomaly-memory/v1","items":[{'
        '"id":"x","domain":"aquaponics-lab","label":"normal","source":"x",'
        '"raw_image_stored":true,"features":{"mean_r":0.1,"mean_g":0.2,'
        '"mean_b":0.7,"luma_std":0.1,"dark_ratio":0.0,"border_delta":0.1,"chroma":0.6}}]}',
        encoding="utf-8",
    )
    blocked = False
    try:
        load_exemplars(path)
    except ValueError:
        blocked = True
    assert blocked


def test_log_anomaly_detector_stays_a_different_specialist() -> None:
    from bellium.registry.catalog import get_specialist
    visual = get_specialist("bellium/knn/visual-anomaly:v0")
    logs = get_specialist("bellium/micro-nn/anomaly-detector:legacy-botte")
    assert visual.task != logs.task
    assert visual.authority_mode.value == "consultative"
    assert logs.authority_mode.value == "observe"
