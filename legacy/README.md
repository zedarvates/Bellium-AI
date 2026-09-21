# Legacy specialist migration documentation

This directory hosts the initial imported specialist models from `zedarvates/botte-secrete` (pinned revision `d8c10b89d3130913b4f62a0f94fa5316ede2a3a8`), integrated under Bellium AI with full provenance.

## Components

### 1. Micro-NN family (`legacy/micro_nn/`)
- **11 Weight-exact JSON models** under `legacy/micro_nn/models/`:
  - `anomaly_detector.json` (`bellium/micro-nn/anomaly-detector:legacy-botte`)
  - `binary_router.json` (`bellium/micro-nn/binary-router:legacy-botte`)
  - `cloud_escalation_predictor.json` (`bellium/micro-nn/cloud-escalation:legacy-botte`)
  - `compressibility_predictor.json` (`bellium/micro-nn/compressibility:legacy-botte`)
  - `context_pruning_predictor.json` (`bellium/micro-nn/context-pruning:legacy-botte`)
  - `effort_classifier.json` (`bellium/micro-nn/effort-classifier:legacy-botte`)
  - `error_classifier.json` (`bellium/micro-nn/error-classifier:legacy-botte`)
  - `response_length_predictor.json` (`bellium/micro-nn/response-length:legacy-botte`)
  - `semantic_cache_hit_predictor.json` (`bellium/micro-nn/semantic-cache-hit:legacy-botte`)
  - `skip_agent_predictor.json` (`bellium/micro-nn/skip-agent:legacy-botte`)
  - `tool_call_predictor.json` (`bellium/micro-nn/tool-call:legacy-botte`)
- **Provenance manifest** (`legacy/micro_nn/provenance.json`): stores pinned revision, size, and SHA-256 for each model.
- **Contrats de features & extracteurs** (`legacy/micro_nn/features.py`).
- **Inférence autonome** (`legacy/micro_nn/cli.py`): Python pur, déléguée au cœur Bellium validé. Le binaire Rust optionnel n'est pas livré ni validé dans cet extrait.
- **Calibration & Température** (`legacy/micro_nn/calibration.py`).
- **Tests validés** (`test_features.py`, `test_botte_nn.py`, `test_calibration.py`).

### 2. k-NN Asset Quality (`legacy/knn_asset_quality/`)
- **Algorithme d'évaluation k-NN** (`legacy/knn_asset_quality/memory.py`): filtres d'intégrité déterministes en amont, vote k-NN par famille d'assets (mesh, image, texture, etc.), mode shadow sans action automatique.
- **Fixtures & Tests** (`legacy/knn_asset_quality/test_asset_quality.py`).
- **Données privées** : **Exclues** conformément aux règles de migration.

## Running tests

```bash
python -m pytest -q
# Individual compatibility scripts, from the repository root:
python -m legacy.micro_nn.test_features
python -m legacy.micro_nn.test_botte_nn
python -m legacy.micro_nn.test_calibration
python -m legacy.knn_asset_quality.test_asset_quality
```

Temperature scaling is applied by the named legacy classifiers when a valid
calibration exists. Calibration sidecars are bound to model bytes. Automatic log
calibration is unavailable: `calibrate_from_logs` returns `None`; use `calibrate`
with explicitly verified probabilities and labels. No private log collector is imported.
