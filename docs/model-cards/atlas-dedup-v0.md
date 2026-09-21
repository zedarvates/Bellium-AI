Model card: bellium/hybrid/atlas-dedup:v0

- Task: remove exact duplicate frames before packing: one stored copy per distinct image plus an
  alias map for the declared frames.
- Authority: consultative. written_files and certified stay false: nothing is written and no
  engine runs.
- Exactness: two frames merge only when width, height, channel count and the digest of their pixel
  bytes are identical. A single differing channel value keeps both frames.
- Bounded tolerance: a caller may declare max_delta, a per-channel bound in [0, 255]. Two frames
  then merge when no channel of any pixel differs by more than that bound; alpha always counts,
  the worst measured difference is reported per alias, per group and for the whole atlas, and a
  matching frame whose earlier stored frame was within the bound is refused as a skipped match.
  The bound is not transitive: frames never chain through an alias to reach a canonical.
- Mirrors: a frame may be stored as none, flip-x, flip-y or rotate-180 of another. Those four keep
  the drawn size; a quarter turn is refused because it would need an engine that rotates a region.
  Every alias declares its transform, and the output counts the transforms used so a caller whose
  engine cannot apply one can reject the atlas before it ships.
- Channel rule: all frames of one atlas share a channel count; a mixed RGB and RGBA set is refused
  rather than converted.
- Verification: check_aliases recomputes every alias against the sources (size, channel count,
  digest, canonical identity, declaration order) and re-adds original_bytes, stored_bytes and each
  group's savings. A report that fails is never used: the specialist abstains with
  alias_failed_verification.
- Pipeline: analyze, verify the alias map, pack only the distinct frames, verify the plan,
  rasterize, verify the pages, optionally encode and decode them back. The output carries
  placements, so a caller can draw any declared frame through its alias and region.
- Measured: benchmarks/manifests/atlas-dedup-v1.json, 9 cases: 108 declared frames to 59 stored,
  86592 bytes saved, 0 bound violations and 0 violations. A mirrored walk of 12 frames stores 4
  images and redraws every frame exactly through its declared transform; the jittered set stores 4
  at a declared bound of 3 with a worst measured delta of 2, and stores 12 at a bound of 1.
- Limits: no perceptual metric (the tolerance is a declared per-channel bound), no quarter turns,
  no flag to switch mirror matching off (a caller must read the transforms), no cross-atlas
  deduplication, no engine-side frame list rewriting, and no real project sheet was measured.
- Evidence: tests/test_atlas_dedup.py, scripts/benchmark_atlas_dedup.py.
