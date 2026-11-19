# Small-learning packaging gate - 9 October 2026

The k-NN and micro-NN changes now have a locally built wheel and source archive,
validated outside the checkout. These are uncommitted local candidates retaining
version 0.1.0, not a public release. No distributed model weights changed.

## Fixes made during validation

The source manifest omitted examples and documented root test/benchmark entry
points. `MANIFEST.in` now includes examples, their documentation/requirements,
run_all_tests.py, benchmarks.py, CONTRIBUTING.md, SECURITY.md and documentation
PNG/SVG assets. Archive contents are checked against the captured source bytes.

The core CLI also imported `bellium.filters` to obtain filter names before parsing
any command. That import loaded Pillow, so even `bellium --help` and consultative
routing failed in a core-only installation. Filter names now live in a shared
dependency-free catalog, still re-exported through the existing image operations.
Only image commands check Pillow; when it is absent they return code 2 with an
instruction to install `bellium-ai[image]`, without a traceback. Nine regression
tests reproduce the missing-Pillow environment even on a developer machine where
Pillow is installed.

`scripts/smoke_installed.py` now checks the new APIs and CLI startup. It exercises
exact and fr-en memory snapshots, Unicode normalization, snapshot independence,
neural domain abstention and deterministic veto, reproducible validation-based
classifier training, scalar regressor training, classification metric denominators,
and JSON model save/load. These tiny probes verify wiring and contracts rather
than training quality.

## Artifact proof

Builds ran from explicit source snapshots, preserving the original checkout's
build directories and prior changes. Earlier artifacts/logs remain diagnostic
evidence. The accepted artifacts are in `dist-verified/`.

- Wheel: bellium_ai-0.1.0-py3-none-any.whl
  - SHA-256: c8cb6318755efdc9bba5dd23f073cac2a3b43ed0ca1048f1b3bc13aa080983e9
- Source archive: bellium_ai-0.1.0.tar.gz
  - SHA-256: 737788c745feb99366c5d4db7a65cbe0774d44f0321ab0c06499ba29773bd9b9

The wheel contains 198 package Python modules, 54 canonical model/resource JSON
files, and 12 legacy JSON files (11 models plus provenance), all byte-identical
to the source snapshot. In each installed environment, those 264 files match
the wheel bytes. Installer metadata also identifies the accepted wheel hash.
The archive contains the examples, fixtures and documented test runners.

## Separate verification claims

| Proof | Environment and result |
| --- | --- |
| Installed core wheel | Fresh venv, only Bellium AI and pip; no mandatory external runtime package. All new API probes, CLI help/filter help, routing and missing-image-extra handling pass. |
| Installed image wheel | Separate fresh venv with Pillow 11.3.0; the same API/CLI probes plus the existing image smoke check pass. |
| Resource/catalog delivery | Both installed environments expose 54 JSON resources, 11 legacy models and 90 registry entries. |
| Source archive tests | Actual extracted archive: 1,190 pytest tests, 15 internal suites, 14 example tests and Ruff pass. |
| Standalone example | The archive's memory example runs with the isolated core interpreter and imports the installed wheel; exact abstention, fr-en hits and unknown-domain abstention pass. |

Installed checks use `-I`, run outside the checkout and verify their package path
lies inside the expected venv. Source tests run from the extracted source archive
with developer test tools; they are distinct from installed-wheel smoke checks.
Windows/Python 3.13.5 is the only platform/runtime verified here. The wheel tag
does not establish runtime testing on every supported Python/platform.

## Evidence and reproduction

The consolidated report is
`benchmarks/manifests/small-learning-packaging-v1.json`. It links the accepted
artifact hashes, source snapshot, installed-environment proofs and archive logs.
The full local evidence directory also contains run-manifest.json, source hashes,
build logs and the helper scripts used for inspection.

From an isolated environment where the wheel is installed:

```powershell
python -I <path-to-checkout>/scripts/smoke_installed.py
```

Append `--image` only in an environment with the image extra installed. The updated
smoke script is already part of the repository's existing wheel CI workflow.

Next: assess native micro-NN probability calibration with separate calibration
and fresh evaluation splits before changing confidence thresholds or shipped
weights. A successful package test does not establish real-world model quality.
