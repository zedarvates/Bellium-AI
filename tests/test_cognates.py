from bellium.knn.cognate import load_cognates, retrieve_cognates


def test_labelled_attested_set_links_other_lects() -> None:
    result = retrieve_cognates({
        "family_id": "lab-family",
        "language_id": "fixture-lab",
        "form": "pata",
    })
    assert result.abstained is False
    assert result.output["status"] == "linked"
    assert result.output["certified"] is False
    assert result.output["link_attested"] is True
    assert result.output["proto_form"] is None
    forms = {item["form"] for item in result.output["neighbors"]}
    assert forms == {"pada", "patah"}
    assert "pata" not in forms
    langs = {item["language_id"] for item in result.output["neighbors"]}
    assert "fixture-lab" not in langs


def test_reconstructed_member_prevents_attested_link() -> None:
    result = retrieve_cognates({
        "family_id": "lab-family",
        "language_id": "fixture-lab",
        "form": "kama",
    })
    assert result.output["status"] == "linked"
    assert result.output["link_attested"] is False
    assert result.output["certified"] is False
    assert result.output["evidence"] == "reconstructed"


def test_unlabelled_lookalikes_stay_inferred() -> None:
    result = retrieve_cognates({
        "family_id": "lab-family",
        "language_id": "fixture-lab",
        "form": "naka",
        "ipa": ["n", "a", "k", "a"],
    })
    assert result.abstained is False
    assert result.output["status"] == "suggest"
    assert result.output["gloss"] == "stone"
    assert result.output["certified"] is False
    assert result.output["link_attested"] is False
    assert result.output["evidence"] == "inferred"
    assert result.output["proto_form"] is None


def test_families_are_isolated() -> None:
    result = retrieve_cognates({
        "family_id": "other-family",
        "language_id": "fixture-lab",
        "form": "pata",
    })
    assert result.abstained is True


def test_same_language_lookalikes_are_not_cognates() -> None:
    result = retrieve_cognates({
        "family_id": "lab-family",
        "language_id": "sister-lab",
        "form": "zzzz",
    })
    assert result.abstained is True


def test_raw_audio_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        '{"schema":"bellium.cognate-memory/v1","items":[{'
        '"id":"x","family_id":"lab-family","language_id":"fixture-lab",'
        '"form":"pata","ipa":["p","a"],"gloss":"water","evidence":"attested",'
        '"source":"x","raw_audio_stored":true}]}',
        encoding="utf-8",
    )
    blocked = False
    try:
        load_cognates(path)
    except ValueError:
        blocked = True
    assert blocked
