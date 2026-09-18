Model card: bellium/hybrid/atlas-packing:v0

- Task: place declared frames (name, width, height) on bounded atlas pages and return the
  geometry: page, x, y per frame.
- Authority: consultative. pixels_written stays false: a plan is geometry, not an image.
- Public API: bellium.atlas.packing.pack, check_plan and power_of_two_ceiling; the specialist
  is pack_atlas({"frames", "width", "height", "method", "padding", "max_pages", "power_of_two"}).
- Method: bottom-left skyline placement, frames sorted by descending height, then width, then
  name. Two published baselines stay in the same module: insertion-order shelf (next-fit rows)
  and height-sorted shelf.
- Bounds: no rotation, no trimming and no scaling; max_pages is capped at 64 and defaults to 1.
  padding is a gutter kept around every frame and against the page border.
- Abstention: a frame that cannot fit comes back in unplaced with reason oversized or no_room
  and the response abstains. Nothing is shrunk to make a page succeed.
- Invalid input: duplicate names, unknown keys, non-positive or non-integer sizes, negative
  padding, more than 64 pages and unknown methods raise ValueError.
- Verification: check_plan recomputes bounds, size preservation, single placement, pairwise
  separation, the unplaced bookkeeping and the reported used_area on every response.
- Determinism: the skyline plan does not depend on the declared frame order; an empty page is
  never emitted.
- Measured: benchmarks/manifests/atlas-packing-v1.json, 10 authored cases x 3 methods with
  0 invariant violations. Against the skyline packer, the sorted shelf scores 0 wins / 7 ties /
  3 losses and the naive shelf 0 wins / 6 ties / 4 losses. On the 96 x 96 measured case the
  skyline places all 14 frames where insertion order leaves 4 unplaced.
- Latency: 0.22 to 1.05 ms for 64 to 400 frames on this workstation, roughly twice the shelf
  cost and still sub-millisecond.
- Limits: no rasterization, no rotation or trimming, no engine import, no duplicate-frame
  removal and no evaluation on real project sheets.
- Evidence: tests/test_atlas_packing.py, scripts/benchmark_atlas.py.

