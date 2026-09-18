from copy import deepcopy

from bellium.knn.phoneme import load_inventory, retrieve_phoneme
from bellium.knn.pronunciation import compare_pronunciation, load_pronunciations


def test_phoneme_knn_retrieves_voiced_bilabial() -> None:
    result = retrieve_phoneme({
        "language_id": "fixture-lab",
        "features": {"voicing": 1.0, "manner": 0.0, "place": 0.0, "rounding": 0.0},
    })
    assert result.abstained is False
    assert result.output["ipa"] == "b"
    assert result.output["evidence"] == "attested"
    assert result.output["certified"] is True
    assert result.output["neighbors"][0]["ipa"] == "b"
    assert len(result.output["neighbors"]) >= 3


def test_phoneme_knn_keeps_languages_isolated() -> None:
    result = retrieve_phoneme({
        "language_id": "other-lab",
        "features": {"voicing": 1.0, "manner": 0.0, "place": 0.0, "rounding": 0.0},
    })
    assert result.abstained is True
    assert result.output["evidence"] == "speculative"


def test_phoneme_knn_does_not_certify_reconstructed_neighbors() -> None:
    inventory = deepcopy(load_inventory())
    for item in inventory:
        if item["ipa"] == "m":
            item["evidence"] = "reconstructed"
    result = retrieve_phoneme({
        "language_id": "fixture-lab",
        "features": {"voicing": 1.0, "manner": 0.0, "place": 0.0, "rounding": 0.0},
    }, inventory=inventory)
    assert result.abstained is False
    assert result.output["certified"] is False
    assert result.output["evidence"] == "reconstructed"


def test_unknown_formants_are_not_filled_with_zero() -> None:
    result = retrieve_phoneme({
        "language_id": "fixture-lab",
        "features": {"voicing": 0.0, "manner": 0.0, "place": 0.0, "rounding": 0.0},
    })
    assert "f1" not in result.output
    assert result.output["ipa"] == "p"


def test_pronunciation_match_can_be_certified_only_if_attested() -> None:
    attested = compare_pronunciation({
        "language_id": "fixture-lab",
        "form": "pata",
        "ipa": ["p", "a", "t", "a"],
    })
    assert attested.abstained is False
    assert attested.output["bucket"] == "match"
    assert attested.output["certified"] is True
    assert attested.output["evidence"] == "attested"

    reconstructed = compare_pronunciation({
        "language_id": "fixture-lab",
        "form": "kama",
        "ipa": ["k", "a", "m", "a"],
    })
    assert reconstructed.output["bucket"] == "match"
    assert reconstructed.output["certified"] is False
    assert reconstructed.output["evidence"] == "reconstructed"


def test_pronunciation_reports_substitution() -> None:
    result = compare_pronunciation({
        "language_id": "fixture-lab",
        "form": "pata",
        "ipa": ["p", "a", "d", "a"],
    })
    assert result.output["certified"] is False
    assert result.output["bucket"] in {"close", "mismatch"}
    assert {"learner": "d", "reference": "t"} in result.output["mismatches"]


def test_pronunciation_abstains_without_a_word_form() -> None:
    result = compare_pronunciation({
        "language_id": "fixture-lab",
        "form": "missing-word",
        "ipa": ["p", "a"],
    })
    assert result.abstained is True
    assert result.output["certified"] is False


def test_raw_audio_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        '{"schema":"bellium.pronunciation-memory/v1","items":[{'
        '"id":"x","language_id":"fixture-lab","form":"x","ipa":["p"],'
        '"evidence":"attested","raw_audio_stored":true}]}',
        encoding="utf-8",
    )
    blocked = False
    try:
        load_pronunciations(path)
    except ValueError:
        blocked = True
    assert blocked
