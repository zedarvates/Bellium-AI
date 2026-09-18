# P4 atlas packing gate: bounded geometry, verified invariants, measured occupancy

The gate asks for a placement plan before any pixel is written. Every case here is
authored geometry, so the measurement is reproducible without assets, and the plan is
checked against its own claims instead of being trusted.

    python scripts/benchmark_atlas.py

Report: benchmarks/manifests/atlas-packing-v1.json.

## What exists

| Piece | Behaviour |
| --- | --- |
| frames | declared name, width and height; duplicates, unknown keys and non-integer sizes raise |
| skyline | bottom-left placement, sorted by descending height, then width, then name |
| shelf, next-fit | the published baselines, kept beside the shipped packer |
| padding | a gutter kept around every frame and against the page border |
| max_pages | bounded at 64, default 1; frames beyond it come back unplaced |
| check_plan | recomputes bounds, size, single placement, separation, unplaced bookkeeping and used_area |
| pack_atlas | consultative specialist; abstains with frames_do_not_fit instead of shrinking a frame |

Frames that are wider or taller than the page are reported with reason `oversized`; frames
that no longer fit on the declared pages are reported with `no_room`. Neither is scaled,
rotated or cropped.

## Invariants, recomputed on every case

| Invariant | Result |
| --- | --- |
| Every frame is placed exactly once or reported unplaced | holds |
| Placements stay inside the page with their padding | holds |
| Two cells (content inflated by padding) never overlap | holds |
| The skyline plan ignores the declared frame order | holds |
| The declared frames are not mutated | holds |
| Invariant violations across 10 cases x 3 methods | 0 |

## Measured occupancy

| Case | Frames | Page | Skyline | Sorted shelf | Naive shelf |
| --- | --- | --- | --- | --- | --- |
| uniform_32 | 64 x 32x32 | 256 | 1.0000 complete | 1.0000 complete | 1.0000 complete |
| uniform_16 | 256 x 16x16 | 256 | 1.0000 complete | 1.0000 complete | 1.0000 complete |
| near_full_24 | 100 x 24x24 | 256 | 0.8789 complete | 0.8789 complete | 0.8789 complete |
| many_small | 400 x 8x8 | 128 | 1.0000, 144 unplaced | 1.0000, 144 unplaced | 1.0000, 144 unplaced |
| mixed_sprite | 60 mixed | 256 | 0.9001 complete | 0.8767, 4 unplaced | 0.5173, 22 unplaced |
| mixed_sprite_padded | 60 mixed, padding 2 | 256 | 0.7646, 18 unplaced | 0.6826, 28 unplaced | 0.4487, 32 unplaced |
| tall_and_wide | 40 x 16x96 / 96x16 | 256 | 0.9375 complete | 0.7031, 10 unplaced | 0.3750, 24 unplaced |
| measured_96 | 14 mixed | 96 | 0.7847 complete | 0.7847 complete | 0.5312, 4 unplaced |
| three_pages | 200 x 32x32 | 128, 3 pages | 1.0000, 152 unplaced | 1.0000, 152 unplaced | 1.0000, 152 unplaced |
| one_oversized | 512x512 and 16x16 | 256 | 1 oversized | 1 oversized | 1 oversized |

Latency, p50 of five runs: skyline 0.22 ms at 64 frames and 1.05 ms at 400 frames against
0.12 ms and 0.62 ms for the sorted shelf. The shipped packer costs roughly twice the
baseline and stays below a millisecond on these sizes.

## The honest reading

The skyline packer wins where sizes are mixed and the page does not divide evenly: it places
all 60 mixed frames at 0.9001 occupancy where the sorted shelf drops 4 and the naive shelf
drops 22, and all 40 extreme-aspect frames at 0.9375 against 0.7031 and 0.3750. It ties the
baselines on the uniform and near-full cases, which is expected when a shelf layout already
packs rows exactly. No case in this set shows a baseline beating the shipped packer, so the
counted result is 0 wins against it for both baselines; that is a statement about these ten
cases, not about packing in general.

Three limits are visible in the table and worth stating plainly. First, padding costs pages:
the padded mixed case leaves 18 frames unplaced on one page. Second, a bounded page count
leaves frames unplaced by design: 200 frames of 32x32 do not fit into three 128 x 128 pages.
Third, occupancy is measured against the pages actually used, so a case that fills its pages
completely reports 1.0000 while still leaving work unplaced.

## Neural tier

None was retained. Packing here is exact integer geometry with a hard no-overlap constraint:
a learned tier would add parameters without adding a decision, and its output would still
have to pass check_plan. A future tier would have to be measured against the published
skyline on these same cases before it could replace it.

Not established: rasterization, rotation, trimming, polygon or mesh atlases, duplicate-frame
removal, engine import conventions, GPU paths, interactive preview and any measurement on
real project sheets. This closes the atlas-packing half of the P4 gap named in the eleventh
wave; engine-import validation remains open.

