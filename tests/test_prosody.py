import json

from bellium.knn.prosody import extract_features, load_profiles, retrieve_prosody


def _iamb(jitter=0.0):
    durs = [0.35 + jitter, 0.55, 0.35, 0.60]
    return [
        {"stress": False, "duration": min(1.0, max(0.05, durs[0]))},
        {"stress": True, "duration": 0.55},
        {"stress": False, "duration": 0.35},
        {"stress": True, "duration": 0.60},
    ]


def _trochee():
    return [
        {"stress": True, "duration": 0.55},
        {"stress": False, "duration": 0.35},
        {"stress": True, "duration": 0.55},
        {"stress": False, "duration": 0.40},
    ]


def test_iambic_contour_matches_attested_profiles() -> None:
    result = retrieve_prosody({
        "language_id": "fixture-lab",
        "syllables": _iamb(0.02),
    })
    assert result.abstained is False
    assert result.output["label"] == "iambic"
    assert result.output["certified"] is False
    assert result.output["evidence"] == "attested"


def test_trochaic_contour_does_not_collapse_into_iambic() -> None:
    result = retrieve_prosody({
        "language_id": "fixture-lab",
        "syllables": _trochee(),
    })
    assert result.abstained is False
    assert result.output["label"] == "trochaic"


def test_single_reconstructed_even_profile_is_not_enough() -> None:
    result = retrieve_prosody({
        "language_id": "fixture-lab",
        "syllables": [
            {"stress": True, "duration": 0.45},
            {"stress": True, "duration": 0.45},
            {"stress": True, "duration": 0.45},
            {"stress": True, "duration": 0.50},
        ],
    })
    assert result.abstained is True
    assert result.output["certified"] is False


def test_unknown_language_abstains() -> None:
    result = retrieve_prosody({
        "language_id": "other-lab",
        "syllables": _iamb(),
    })
    assert result.abstained is True


def test_missing_duration_is_not_filled_with_zero() -> None:
    blocked = False
    try:
        extract_features([
            {"stress": True, "duration": None},
            {"stress": False, "duration": 0.4},
        ])
    except ValueError:
        blocked = True
    assert blocked


def test_raw_audio_is_rejected(tmp_path) -> None:
    feats = extract_features(_iamb())
    path = tmp_path / "bad.json"
    payload = {
        "schema": "bellium.prosody-memory/v1",
        "items": [{
            "id": "x",
            "language_id": "fixture-lab",
            "label": "iambic",
            "evidence": "attested",
            "source": "x",
            "raw_audio_stored": True,
            "features": feats,
        }],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    blocked = False
    try:
        load_profiles(path)
    except ValueError:
        blocked = True
    assert blocked
