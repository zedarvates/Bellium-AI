# P4 deduplication gate: one stored copy, aliases, and a pixel-exact redraw

Animations repeat poses; atlases do not have to store them twice. This gate removes exact
duplicates before packing and proves that every declared frame still draws the pixels it
declared. Exactness is the whole contract: there is no tolerance and no threshold.

    python scripts/benchmark_atlas_dedup.py

Report: benchmarks/manifests/atlas-dedup-v1.json.

## What exists

| Piece | Behaviour |
| --- | --- |
| content_key | the identity of an image: width, height, channel count and the digest of its bytes |
| analyze_frames | canonical is the first declared occurrence; groups, aliases and byte accounting |
| unique_frames | the subset that must be stored, in declaration order |
| check_aliases | recomputes aliases, groups, canonical identity, order and every byte total |
| dedup_atlas | analyze, verify, pack the distinct frames only, rasterize, verify, optionally encode |
| placements | returned with the pages, so any declared frame can be drawn through its alias |

A report that fails its own check is never used, and a stored page that fails raster or PNG
verification is never returned: the specialist abstains with the matching reason.

## Invariants, recomputed

| Invariant | Result |
| --- | --- |
| Every declared frame has exactly one alias | holds |
| An alias and its canonical hold identical pixels | holds |
| Unique frames keep the declaration order | holds |
| Original bytes equal stored bytes plus saved bytes | holds |
| Group savings match their member counts | holds |
| A single differing channel value prevents merging | holds: the near-miss control saves 0 bytes |
| Declared frames redrawn from the stored atlas | 0 mismatches over 9 cases, transforms included |
| Violations across the benchmark | 0 |
| Drawn pixels differing from the declared bound | 0 cases beyond the declared max_delta |

## Measured

| Case | Declared | Stored | Saved bytes | Ratio | Pages (all to stored) | PNG bytes | Redrawn mismatches |
| --- | --- | --- | --- | --- | --- | --- | --- |
| walk_duplicates | 12 | 4 | 8192 | 0.667 | 1 to 1 | 3604 | 0 |
| idle_repeats | 8 | 3 | 2880 | 0.625 | 1 to 1 | 1601 | 0 |
| no_duplicates | 12 | 12 | 0 | 0.000 | 1 to 1 | 10914 | 0 |
| near_misses | 12 | 12 | 0 | 0.000 | 1 to 1 | 10922 | 0 |
| mixed_sizes | 12 | 4 | 9984 | 0.667 | 1 to 1 | 3540 | 0 |
| page_pressure | 16 | 4 | 49152 | 0.750 | 2 incomplete to 1 complete | 14041 | 0 |

Totals: 108 declared frames to 59 stored, 86592 bytes saved, 0 bound violations, 0 violations.
Analysis costs 13.1 ms and the full pipeline including PNG encoding 139 ms per case; the analysis
grew with mirror matching, which searches transforms rather than only equal digests.

## Mirrors: exact, declared, and counted

A sprite that walks left and right usually stores both directions. The gate matches a frame to an
earlier stored image under one of four transforms that keep the drawn size: none, flip-x, flip-y
and rotate-180. A quarter turn is refused, because it would change the drawn size and need an
engine that rotates a region rather than a runtime flip.

| Case | Declared | Stored | Transforms used | Saved bytes | Drawn mismatches |
| --- | --- | --- | --- | --- | --- |
| mirrored_walk | 12 | 4 | none, flip-x, flip-y | 8192 | 0 |

Twelve frames of four poses with two mirror directions store as four images, and every declared
frame redraws pixel for pixel once the alias transform is applied. Every alias declares its
transform, the duplicate groups list the transforms of their members, and the output counts them,
so a caller whose engine cannot flip a region can reject the atlas before it ships. There is no
flag to switch mirror matching off: the transforms are part of the answer, not a hidden option.

## Bounded tolerance, declared and measured

Exactness is the default, not the only setting. A caller may declare max_delta, a per-channel
bound, and two frames then merge when no channel of any pixel differs by more than that bound.
The bound is declared before the run, alpha always counts, and the worst measured difference is
reported for every alias, every group and the atlas as a whole. The same generated jittered set
drives both rows below, so the only difference is the declared bound.

| Case | Bound | Declared | Stored | Saved bytes | Worst measured delta | Differing pixels | Bound violations |
| --- | --- | --- | --- | --- | --- | --- | --- |
| jitter_within_bound | 3 | 12 | 4 | 8192 | 2 | 47 of 12288 | 0 |
| jitter_beyond_bound | 1 | 12 | 12 | 0 | 0 | 0 | 0 |

With the bound at 3 the set stores a third of the frames and the worst pixel that reaches a page
differs from the artist's frame by 2 out of 255 on one channel. With the bound at 1 nothing
merges, which is the same answer as exactness: a caller that declares a tight bound gets a tight
result rather than a silent approximation. Non-transitivity is enforced too: a frame never joins
an alias to reach a canonical, so a chain of small differences cannot accumulate into one big one.
Across the whole benchmark the only pixels that differ between what was declared and what a page
would draw are those 47, all inside the bound the caller declared.

## The honest reading

The bytes claim is real and measured: three of six cases remove two thirds of the frame bytes,
and one case turns a two-page set that did not fit into a single complete page. The near-miss
control is the other half of the claim: twelve frames that differ by one channel value save
nothing, and their encoded pages land within 8 bytes of the no-duplicates control, which shows
how close they were and that exactness did not blur them.

What this gate does not do is decide when two frames are "close enough". That decision changes
pixels an artist drew, so it stays out: a tolerance-based pass would need its own declared
threshold, its own error measurement and its own review, and nothing here should be read as a
step towards silently dropping near duplicates. Deduplication also does not reduce the frame
list an engine sees; it reduces what the atlas stores, and the alias map is how the caller keeps
its animation intact.

Not established: a perceptual metric or any claim about what a viewer notices, quarter turns or
mirror matching across frame sizes, deduplication across several atlases, engine-side frame list rewriting, memory or
import-time savings inside an engine, and any measurement on a real project sheet. The bounded
tier changes pixels on purpose, so it needs the caller's own review before a shipped asset uses
it; the default stays exact.
