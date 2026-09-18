from __future__ import annotations

from dataclasses import dataclass

from bellium.contracts.schema import AuthorityMode


@dataclass(frozen=True)
class SpecialistSpec:
    specialist_id: str
    family: str
    task: str
    source: str
    authority_mode: AuthorityMode
    decision_eligible: bool
    notes: str


SPECIALISTS: tuple[SpecialistSpec, ...] = (
    SpecialistSpec("bellium/micro-nn/anomaly-detector:legacy-botte", "micro-nn", "anomaly-detector", "legacy-botte", AuthorityMode.OBSERVE, False, "Imported weights."),
    SpecialistSpec("bellium/micro-nn/binary-router:legacy-botte", "micro-nn", "binary-router", "legacy-botte", AuthorityMode.OBSERVE, False, "Imported weights."),
    SpecialistSpec("bellium/micro-nn/cloud-escalation:legacy-botte", "micro-nn", "cloud-escalation", "legacy-botte", AuthorityMode.OBSERVE, False, "Imported weights."),
    SpecialistSpec("bellium/micro-nn/compressibility:legacy-botte", "micro-nn", "compressibility", "legacy-botte", AuthorityMode.OBSERVE, False, "Imported weights."),
    SpecialistSpec("bellium/micro-nn/context-pruning:legacy-botte", "micro-nn", "context-pruning", "legacy-botte", AuthorityMode.OBSERVE, False, "Imported weights."),
    SpecialistSpec("bellium/micro-nn/effort-classifier:legacy-botte", "micro-nn", "effort-classifier", "legacy-botte", AuthorityMode.OBSERVE, False, "Imported weights."),
    SpecialistSpec("bellium/micro-nn/error-classifier:legacy-botte", "micro-nn", "error-classifier", "legacy-botte", AuthorityMode.OBSERVE, False, "Imported weights."),
    SpecialistSpec("bellium/micro-nn/response-length:legacy-botte", "micro-nn", "response-length", "legacy-botte", AuthorityMode.OBSERVE, False, "Imported weights."),
    SpecialistSpec("bellium/micro-nn/semantic-cache-hit:legacy-botte", "micro-nn", "semantic-cache-hit", "legacy-botte", AuthorityMode.OBSERVE, False, "Imported weights."),
    SpecialistSpec("bellium/micro-nn/skip-agent:legacy-botte", "micro-nn", "skip-agent", "legacy-botte", AuthorityMode.OBSERVE, False, "Imported weights."),
    SpecialistSpec("bellium/micro-nn/tool-call:legacy-botte", "micro-nn", "tool-call", "legacy-botte", AuthorityMode.OBSERVE, False, "Imported weights."),
    SpecialistSpec("bellium/knn/asset-quality:v0", "knn", "asset-quality", "bellium-native", AuthorityMode.SHADOW, False, "Deterministic gates first."),
    SpecialistSpec("bellium/knn/patch-inpaint:v0", "knn", "selected-zone-fill", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Small holes only."),
    SpecialistSpec("bellium/knn/color-cutout:v0", "knn", "image-cutout", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Hard-edge objects."),
    SpecialistSpec("bellium/micro-nn/inpaint-router:v0", "micro-nn", "inpaint-router", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Proposes patch_knn or escalate."),
    SpecialistSpec("bellium/hybrid/white-background-normalizer:v0", "hybrid", "white-background", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Cutout then white composite."),
    SpecialistSpec("bellium/knn/language-phoneme:v0", "knn", "phoneme-retrieval", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Provenance-aware phone retrieval."),
    SpecialistSpec("bellium/knn/pronunciation-similarity:v0", "knn", "pronunciation-similarity", "bellium-native", AuthorityMode.CONSULTATIVE, False, "DTW plus word k-NN; reconstructed cannot certify."),
    SpecialistSpec("bellium/knn/tool-router:v0", "knn", "tool-routing", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Closed-catalog exemplar routing."),
    SpecialistSpec("bellium/micro-nn/tool-router:v0", "micro-nn", "tool-routing", "bellium-native", AuthorityMode.CONSULTATIVE, False, "none / use_tool / escalate."),
    SpecialistSpec("bellium/hybrid/tool-router:v0", "hybrid", "tool-routing", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Veto then k-NN then micro-NN."),
    SpecialistSpec("bellium/knn/memory-reranker:v0", "knn", "memory-rerank", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Lexical rerank, titles only."),
    SpecialistSpec("bellium/knn/visual-anomaly:v0", "knn", "visual-anomaly", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Domain-isolated visual novelty. Not the log detector."),
    SpecialistSpec("bellium/knn/grapheme-phoneme:v0", "knn", "grapheme-phoneme", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Exact attested mappings first; neighbors stay uncertified."),
    SpecialistSpec("bellium/knn/cognate-retrieval:v0", "knn", "cognate-retrieval", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Labelled sets first; similarity never invents a proto-form."),
    SpecialistSpec("bellium/knn/prosody-profile:v0", "knn", "prosody-profile", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Syllable contours only; no audio, never certified."),
    SpecialistSpec("bellium/knn/panel-safe-crop:v0", "knn", "panel-safe-crop", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Family-local margin proposal."),
    SpecialistSpec("bellium/hybrid/panel-safe-crop:v0", "hybrid", "panel-safe-crop", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Never clip required content; k-NN only proposes margin."),
    SpecialistSpec("bellium/knn/speech-bubble-region:v0", "knn", "speech-bubble-region", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Labels a region bubble or not. No OCR."),
    SpecialistSpec("bellium/hybrid/speech-bubble-region:v0", "hybrid", "speech-bubble-region", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Finds bright compact regions; never reads dialogue."),
    SpecialistSpec("bellium/knn/consistency-retrieval:v0", "knn", "consistency-retrieval", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Retrieves a style id. Never recolours."),
    SpecialistSpec("bellium/knn/flat-color:v0", "knn", "flat-color", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Labels a region flat or textured."),
    SpecialistSpec("bellium/hybrid/flat-color-helper:v0", "hybrid", "flat-color", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Median-fills a hole only when the field is flat."),
    SpecialistSpec("bellium/knn/voice-activity:v0", "knn", "voice-activity", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Speech vs silence on compact frames."),
    SpecialistSpec("bellium/hybrid/voice-activity:v0", "hybrid", "voice-activity", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Windowed vote. Audio is discarded."),
    SpecialistSpec("bellium/knn/recording-quality:v0", "knn", "recording-quality", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Clean, noisy or clipped on compact frames."),
    SpecialistSpec("bellium/hybrid/recording-quality:v0", "hybrid", "recording-quality", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Clip-level quality. Audio is discarded."),
    SpecialistSpec("bellium/knn/byte-compression:v0", "knn", "byte-compression", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Causal bounded neighbors and lossless residuals; BLCP research codec."),
    SpecialistSpec("bellium/micro-nn/byte-compression:v0", "micro-nn", "byte-compression", "bellium-native", AuthorityMode.CONSULTATIVE, False, "One integer adaptive neuron; no pretrained weights; lossless residuals."),
    SpecialistSpec("bellium/hybrid/image-compression:v0", "hybrid", "image-compression", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Exact L/RGB/RGBA pixels; automatic measured codec selection."),
    SpecialistSpec("bellium/hybrid/stream-compression:v0", "hybrid", "stream-compression", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Independent lossless byte chunks; caller owns transport."),
    SpecialistSpec("bellium/hybrid/splats-compression:v0", "hybrid", "splats-compression", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Exact float32 records and explicit-base temporal XOR; not .fovea."),
    SpecialistSpec("bellium/hybrid/ply-archive:v0", "hybrid", "ply-archive", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Exact supported binary PLY bytes, including SH and unknown scalar fields; no value conversion."),
    SpecialistSpec("bellium/knn/texture-repeat-xy:v0", "knn", "texture-repeat", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Measured periods plus a family-local verdict; no pixels changed."),
    SpecialistSpec("bellium/knn/texture-tileability:v0", "knn", "texture-tileability", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Wrap continuity per axis; the deterministic baseline must agree."),
    SpecialistSpec("bellium/nano-nn/tile-seam:v0", "nano-nn", "texture-tileability", "bellium-native", AuthorityMode.CONSULTATIVE, False, "38-parameter nano model on six deterministic seam measures."),
    SpecialistSpec("bellium/hybrid/texture-tile-fixer:v0", "hybrid", "texture-tileability", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Bounded wrap feathering only when every tier agrees."),
    SpecialistSpec("bellium/knn/sprite-anchor:v0", "knn", "sprite-anchor", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Family-local anchor proposal inside the content box."),
    SpecialistSpec("bellium/hybrid/sprite-frame-prep:v0", "hybrid", "sprite-preparation", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Crop, pad and anchor; clipped silhouettes refuse preparation."),
    SpecialistSpec("bellium/micro-nn/npc-behavior-router:v0", "micro-nn", "npc-behavior", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Compact state classifier; imitates a documented rule."),
    SpecialistSpec("bellium/hybrid/npc-behavior-router:v0", "hybrid", "npc-behavior", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Deterministic veto before the classifier; nothing is executed."),
    SpecialistSpec("bellium/knn/clip-loop:v0", "knn", "clip-loop", "bellium-native", AuthorityMode.CONSULTATIVE, False, "First/last comparison; the threshold baseline must agree."),
    SpecialistSpec("bellium/nano-nn/frame-phase:v0", "nano-nn", "frame-phase", "bellium-native", AuthorityMode.CONSULTATIVE, False, "No model shipped: the published phase rule was not beaten."),
    SpecialistSpec("bellium/hybrid/sprite-sheet-prep:v0", "hybrid", "sprite-sheet", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Measured grid, frame phases and a loop check; no export."),
    SpecialistSpec("bellium/knn/consequence-precedent:v0", "knn", "consequence-prediction", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Verified precedents only; nothing is executed."),
    SpecialistSpec("bellium/micro-nn/consequence-predictor:v0", "micro-nn", "consequence-prediction", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Approximates the published risk rule at 0.900 agreement."),
    SpecialistSpec("bellium/hybrid/consequence-predictor:v0", "hybrid", "consequence-prediction", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Veto, precedents and micro-NN must all agree with the rule."),
    SpecialistSpec("bellium/knn/physical-case-retrieval:v0", "knn", "physical-estimate", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Bracketed or bilinear interpolation inside a declared domain; never a measurement."),
    SpecialistSpec("bellium/nano-nn/gravity-residual:v0", "nano-nn", "physical-estimate", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Four-parameter linear model on a physical basis; reference-first."),
    SpecialistSpec("bellium/micro-nn/atmosphere-model:v0", "micro-nn", "physical-estimate", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Log-pressure regressor, worst case 545 Pa; reference-first."),
    SpecialistSpec("bellium/hybrid/physical-estimate:v0", "hybrid", "physical-estimate", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Reference first, cheaper tiers reported beside it, provisional when alone."),
    SpecialistSpec("bellium/knn/texture-orientation:v0", "knn", "texture-orientation", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Grain direction and period; perspective variations abstain."),
    SpecialistSpec("bellium/knn/easing-profile:v0", "knn", "animation-timing", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Names the easing curve; the threshold rule must agree."),
    SpecialistSpec("bellium/hybrid/animation-timing:v0", "hybrid", "animation-timing", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Holds, peak, easing and duplicate frames; nothing is retimed or removed."),
    SpecialistSpec("bellium/hybrid/albedo-separation:v0", "hybrid", "material-separation", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Both candidates returned with evidence; a declared shading prior decides."),
    SpecialistSpec("bellium/hybrid/photometric-normals:v0", "hybrid", "pbr-normals", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Declared lights, per-patch trust, abstention when the Lambertian model fails."),
    SpecialistSpec("bellium/hybrid/direct-filters:v0", "hybrid", "image-editing", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Declarative recipe replayed on the source; source preserved."),
    SpecialistSpec("bellium/hybrid/magic-eraser:v0", "hybrid", "image-editing", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Reuses the inpaint router; uncertain selections escalate instead of filling."),
    SpecialistSpec("bellium/hybrid/atlas-packing:v0", "hybrid", "atlas-packing", "bellium-native", AuthorityMode.CONSULTATIVE, False, "Placement geometry only; frames that do not fit are reported unplaced."),
)


def list_specialists(*, family: str | None = None) -> tuple[SpecialistSpec, ...]:
    if family is None:
        return SPECIALISTS
    return tuple(item for item in SPECIALISTS if item.family == family)


def get_specialist(specialist_id: str) -> SpecialistSpec:
    for item in SPECIALISTS:
        if item.specialist_id == specialist_id:
            return item
    raise KeyError(specialist_id)
