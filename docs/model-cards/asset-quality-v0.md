# Model card: bellium/knn/asset-quality:v0

- Task: provide shadow advice from externally verified asset reports.
- Hard integrity/license checks run first; learned advice never activates assets.
- The library consumes caller-supplied feature/check values. It does not itself
  inspect geometry, determine ownership, or measure every image defect.
- Memory: bounded JSONL, family and hash deduplication, validated records. A full
  ledger rejects further writes until the caller rotates it. One writer assumed.
- Evidence: deterministic veto, deduplication and corrupt-ledger regression tests.
- Independent asset-quality accuracy and RAM/VRAM: not measured.
