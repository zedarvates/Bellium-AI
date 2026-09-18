from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.registry.catalog import get_specialist, list_specialists


def test_specialist_result_preserves_authority_boundary() -> None:
    result = SpecialistResult(
        specialist_id="bellium/knn/example:v0",
        output={"proposal": "example"},
        confidence=0.8,
        abstained=False,
        authority_mode=AuthorityMode.CONSULTATIVE,
    )
    assert result.authority_mode is AuthorityMode.CONSULTATIVE


def test_registry_lists_legacy_and_native() -> None:
    specs = list_specialists()
    ids = {item.specialist_id for item in specs}
    assert "bellium/micro-nn/error-classifier:legacy-botte" in ids
    assert "bellium/knn/patch-inpaint:v0" in ids
    assert "bellium/micro-nn/inpaint-router:v0" in ids
    assert "bellium/knn/language-phoneme:v0" in ids
    assert "bellium/knn/pronunciation-similarity:v0" in ids
    assert "bellium/hybrid/tool-router:v0" in ids
    assert "bellium/knn/memory-reranker:v0" in ids
    assert "bellium/knn/visual-anomaly:v0" in ids
    assert "bellium/knn/grapheme-phoneme:v0" in ids
    assert "bellium/knn/cognate-retrieval:v0" in ids
    assert "bellium/knn/prosody-profile:v0" in ids
    assert "bellium/hybrid/panel-safe-crop:v0" in ids
    assert "bellium/hybrid/speech-bubble-region:v0" in ids
    assert "bellium/knn/consistency-retrieval:v0" in ids
    assert "bellium/hybrid/flat-color-helper:v0" in ids
    assert "bellium/hybrid/voice-activity:v0" in ids
    assert "bellium/hybrid/recording-quality:v0" in ids
    assert "bellium/knn/texture-repeat-xy:v0" in ids
    assert "bellium/knn/texture-tileability:v0" in ids
    assert "bellium/nano-nn/tile-seam:v0" in ids
    assert "bellium/hybrid/texture-tile-fixer:v0" in ids
    assert "bellium/knn/sprite-anchor:v0" in ids
    assert "bellium/hybrid/sprite-frame-prep:v0" in ids
    assert "bellium/micro-nn/npc-behavior-router:v0" in ids
    assert "bellium/hybrid/npc-behavior-router:v0" in ids
    assert "bellium/knn/physical-case-retrieval:v0" in ids
    assert "bellium/nano-nn/gravity-residual:v0" in ids
    assert "bellium/micro-nn/atmosphere-model:v0" in ids
    assert "bellium/hybrid/physical-estimate:v0" in ids
    assert "bellium/knn/easing-profile:v0" in ids
    assert "bellium/hybrid/animation-timing:v0" in ids
    assert "bellium/hybrid/photometric-normals:v0" in ids
    assert "bellium/hybrid/atlas-packing:v0" in ids
    assert all(item.decision_eligible is False for item in specs)
    native = get_specialist("bellium/knn/asset-quality:v0")
    assert native.authority_mode is AuthorityMode.SHADOW
