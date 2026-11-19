from dataclasses import FrozenInstanceError

import pytest

from bellium.knn import PreparedMemory, rerank_memory


def _items():
    return [{"id": name, "domain": "fr", "title": "Le titre reste intact",
             "terms": "le détourage de cette image sur un fond transparent", "evidence": "attested"}
            for name in ("a", "b")]


def test_explicit_profile_filters_query_and_memory_identically():
    memory = _items()
    query = {"domain": "fr", "terms": "Je souhaite le détourage de mon image sur un fond transparent"}
    listed = rerank_memory(query, memory=memory, profile="fr-en")
    prepared = rerank_memory(query, memory=PreparedMemory(memory, profile="fr-en"))
    assert listed == prepared
    assert not listed.abstained
    assert listed.output["lexical_profile"] == "fr-en"
    assert listed.output["neighbors"][0]["similarity"] == 0.8
    assert listed.output["neighbors"][0]["title"] == "Le titre reste intact"
    assert memory[0]["terms"].startswith("le détourage")


def test_default_exact_profile_preserves_token_scoring():
    query = {"domain": "fr", "terms": "le détourage de cette image sur un fond transparent"}
    implicit = rerank_memory(query, memory=_items())
    assert implicit == rerank_memory(query, memory=_items(), profile="exact")
    assert implicit == rerank_memory(query, memory=PreparedMemory(_items()))
    assert "lexical_profile" not in implicit.output


def test_negations_uncertainty_identifiers_and_numbers_are_retained():
    text = "pas sans non ni not no never without might modèle42 2026"
    memory = [{"id": name, "domain": "fr", "title": name, "terms": text,
               "evidence": "attested"} for name in ("a", "b")]
    result = rerank_memory({"domain": "fr", "terms": text}, memory=memory, profile="fr-en")
    assert not result.abstained
    assert result.output["neighbors"][0]["similarity"] == 1.0
    words = next(PreparedMemory(memory, profile="fr-en")._matching("fr", {"pas"}))["terms"]
    assert words == frozenset(text.split())


def test_negation_is_not_removed_to_inflate_similarity():
    memory = [{"id": name, "domain": "en", "title": name, "terms": "image cutout background",
               "evidence": "attested"} for name in ("a", "b")]
    positive = rerank_memory({"domain": "en", "terms": "the image cutout background"},
                              memory=memory, profile="fr-en")
    negative = rerank_memory({"domain": "en", "terms": "not the image cutout background"},
                              memory=memory, profile="fr-en")
    assert positive.output["neighbors"][0]["similarity"] == 1.0
    assert negative.output["neighbors"][0]["similarity"] == 0.75


def test_profile_with_no_content_abstains_explicitly():
    query = {"domain": "fr", "terms": "le la et de"}
    result = rerank_memory(query, memory=PreparedMemory(_items(), profile="fr-en"))
    assert result.abstained
    assert result.output["reason"] == "no_content_terms"
    assert result.output["neighbors"] == []
    assert result.output["lexical_profile"] == "fr-en"


def test_grammatical_only_record_cannot_be_indexed_as_support():
    memory = _items()
    memory[0]["terms"] = "le la et de"
    with pytest.raises(ValueError, match="no content terms"):
        PreparedMemory(memory, profile="fr-en")
    with pytest.raises(ValueError, match="no content terms"):
        rerank_memory({"domain": "fr", "terms": "image"}, memory=memory, profile="fr-en")


@pytest.mark.parametrize("profile", [None, True, "unknown", "", {}])
def test_invalid_snapshot_profile_is_rejected(profile):
    with pytest.raises(ValueError, match="profile"):
        PreparedMemory(_items(), profile=profile)


def test_snapshot_profile_cannot_be_changed_or_overridden():
    memory = PreparedMemory(_items(), profile="fr-en")
    with pytest.raises(FrozenInstanceError):
        memory.profile = "exact"
    with pytest.raises(ValueError, match="does not match"):
        rerank_memory({"domain": "fr", "terms": "image"}, memory=memory, profile="exact")


def test_filtered_unknown_domain_remains_isolated():
    result = rerank_memory({"domain": "other", "terms": "le détourage image fond transparent"},
                           memory=PreparedMemory(_items(), profile="fr-en"))
    assert result.abstained
    assert result.output["neighbors"] == []
    assert result.output["lexical_profile"] == "fr-en"
