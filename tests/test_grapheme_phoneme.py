from bellium.knn.grapheme_phoneme import load_mappings, map_form


def test_exact_attested_spelling_can_be_certified() -> None:
    result = map_form({
        "language_id": "fixture-lab",
        "direction": "g2p",
        "form": "pata",
    })
    assert result.abstained is False
    assert result.output["status"] == "exact"
    assert result.output["ipa"] == ["p", "a", "t", "a"]
    assert result.output["certified"] is True
    assert result.output["evidence"] == "attested"


def test_reconstructed_exact_mapping_is_not_certified() -> None:
    result = map_form({
        "language_id": "fixture-lab",
        "direction": "g2p",
        "form": "kama",
    })
    assert result.output["ipa"] == ["k", "a", "m", "a"]
    assert result.output["certified"] is False
    assert result.output["evidence"] == "reconstructed"


def test_neighbor_spelling_stays_inferred() -> None:
    result = map_form({
        "language_id": "fixture-lab",
        "direction": "g2p",
        "form": "kaah",
    })
    assert result.abstained is False
    assert result.output["status"] == "suggest"
    assert result.output["ipa"] == ["k", "a"]
    assert result.output["certified"] is False
    assert result.output["evidence"] == "inferred"


def test_mixed_neighbors_do_not_invent_a_pronunciation() -> None:
    result = map_form({
        "language_id": "fixture-lab",
        "direction": "g2p",
        "form": "pada",
    })
    assert result.abstained is True
    assert result.output["reason"] in {"mixed_neighbor_ipa", "too_few_similar_mappings"}


def test_phoneme_to_grapheme_reports_ambiguous_spellings() -> None:
    result = map_form({
        "language_id": "fixture-lab",
        "direction": "p2g",
        "ipa": ["k", "a"],
    })
    assert result.abstained is False
    assert result.output["status"] == "ambiguous"
    assert result.output["certified"] is False
    assert set(result.output["forms"]) == {"ka", "kaa", "kah", "kaaa"}


def test_unique_phoneme_sequence_maps_back() -> None:
    result = map_form({
        "language_id": "fixture-lab",
        "direction": "p2g",
        "ipa": ["p", "a", "t", "a"],
    })
    assert result.output["form"] == "pata"
    assert result.output["certified"] is True


def test_unknown_language_abstains() -> None:
    result = map_form({
        "language_id": "other-lab",
        "direction": "g2p",
        "form": "pata",
    })
    assert result.abstained is True


def test_raw_audio_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        '{"schema":"bellium.grapheme-phoneme/v1","items":[{'
        '"id":"x","language_id":"fixture-lab","form":"ka","ipa":["k","a"],'
        '"evidence":"attested","source":"x","raw_audio_stored":true}]}',
        encoding="utf-8",
    )
    blocked = False
    try:
        load_mappings(path)
    except ValueError:
        blocked = True
    assert blocked
