# Existing specialist migration inventory

Source checked: `zedarvates/botte-secrete` on `main`, 2026-09-14.

Bellium migrates with provenance first. Existing Botte Secrète integrations remain unchanged until Bellium-native packages have tests and stable contracts.

## Existing micro-NN models

| Source model | Bellium ID |
|---|---|
| `anomaly_detector.json` | `bellium/micro-nn/anomaly-detector:legacy-botte` |
| `binary_router.json` | `bellium/micro-nn/binary-router:legacy-botte` |
| `cloud_escalation_predictor.json` | `bellium/micro-nn/cloud-escalation:legacy-botte` |
| `compressibility_predictor.json` | `bellium/micro-nn/compressibility:legacy-botte` |
| `context_pruning_predictor.json` | `bellium/micro-nn/context-pruning:legacy-botte` |
| `effort_classifier.json` | `bellium/micro-nn/effort-classifier:legacy-botte` |
| `error_classifier.json` | `bellium/micro-nn/error-classifier:legacy-botte` |
| `response_length_predictor.json` | `bellium/micro-nn/response-length:legacy-botte` |
| `semantic_cache_hit_predictor.json` | `bellium/micro-nn/semantic-cache-hit:legacy-botte` |
| `skip_agent_predictor.json` | `bellium/micro-nn/skip-agent:legacy-botte` |
| `tool_call_predictor.json` | `bellium/micro-nn/tool-call:legacy-botte` |

Source implementation includes Python feature extraction/inference/training and a Rust inference implementation. JSON weights and their matching feature contract must be migrated from the same pinned source revision.

## Existing k-NN

`skills/asset_quality/memory.py` implements family-isolated k-nearest-neighbor quality memory for image, texture, mesh, animation and Godot assets. Deterministic integrity/licensing gates run first; learned advice is shadow-only and abstains when neighbor support is insufficient.

Proposed ID: `bellium/knn/asset-quality-memory:legacy-botte`.

Migrate algorithm, schemas, fixtures and tests. **Do not migrate private/local neighbor ledgers.**

## Migration rules

1. Do not delete or redirect the original implementation during bootstrap.
2. Pin source revision for every imported model.
3. Preserve original weight bytes and SHA-256.
4. Preserve matching feature contracts.
5. Port tests before behavioral changes.
6. Keep imported models labelled `legacy-botte` until Bellium-native evaluation.
7. Migration grants no additional authority.
8. Never publish private neighbor/example memory.
9. Add provenance/model cards before replacement training.
10. Move consumers only through later bounded changes.
