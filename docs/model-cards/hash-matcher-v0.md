# Model card: bellium/knn/hash-matcher:v0

- Task and bounded inputs: retrieves the nearest known image from a caller-supplied
  gallery of perceptual hashes. The query is a 64-bit hash and the gallery entries
  carry their own hashes; the match is decided by Hamming distance.
- Public API and version: `bellium.knn.hash_matcher.match_hash_knn(query_hash,
  gallery, *, k=1, max_distance=10) -> SpecialistResult`. v0. The hashes themselves
  come from `bellium.magick.hashes` (`compute_ahash`, `compute_dhash`,
  `compute_phash`, `hash_to_hex`).
- Authority mode: `consultative`.
- Model / exemplar source revision and license: no model file in this repository. The
  gallery is supplied entirely by the caller, so no third-party data is bundled.
- Training data and seeds: none; hashes are deterministic functions of the input
  image.
- Independent evaluation data: none published for the retrieval itself.
  `tests/test_fastimage.py` (`test_perceptual_hashes_and_knn_matching`) covers the
  hash functions and a synthetic match.
- Deterministic baseline: exact Hamming comparison over the gallery. The k-NN adds a
  ranked shortlist and a declared `max_distance` bound rather than a different
  metric.
- Quality, false agreements and abstention: an empty gallery abstains with reason
  `empty_gallery` and echoes the query hash in hexadecimal. A near-duplicate and a
  different image are separated only by the declared distance bound, so the caller
  owns that threshold; no false-positive rate is published.
- Hardware and software versions: Windows workstation, CPython 3.14.
- Latency, cold/warm policy, repetitions: not measured.
- RAM / VRAM: not measured.
- Escalation and invalid input behavior: abstention on an empty gallery; otherwise the
  ranked candidates, their distances and the bound are returned with the authority
  mode.
- Known limitations: perceptual hashes are sensitive to crop, rotation and colour
  changes, and nothing here measures that sensitivity. The distance bound is the only
  guard against a false agreement.
- Evidence paths and reproduction command:
  `python -m pytest tests/test_fastimage.py -q`.
