# P9 first gate: direct filters, selections and an editable stack

The gate asks for deterministic filters, selection masks and undo/redo before
anything interactive is attempted. Everything here is pure integer arithmetic on
the repository RGB contract, so a filter can be checked pixel by pixel.

    python scripts/benchmark_editing.py

Report: benchmarks/manifests/editing-v1.json.

## What exists

| Tool | Behaviour |
| --- | --- |
| grayscale | documented luma weights or a channel mean, adjustable strength |
| sepia | published tone matrix, adjustable strength, kept separate from grayscale |
| invert | channel inversion with strength |
| brightness_contrast | classic shift and factor, both declared in [-1, 1] |
| saturation | distance from the luma level, factor in [0, 4] |
| tint | per-channel multipliers in [0, 4] |
| threshold | black and white at a declared luma level, a distinct filter |
| rect, ellipse, polygon, brush | deterministic selection masks in [0, 1] |
| feather | box-blurred mask edges, so an edit fades instead of ending on a line |
| apply_mask | blends an edit by the mask and leaves uncovered pixels byte for byte |
| EditStack | ordered steps, undo, redo, before/after, recipe export and replay |

Strength blends between the source pixel and the fully filtered one, so partial
applications do not need a second implementation.

## Invariants, all verified by the benchmark

| Invariant | Result |
| --- | --- |
| A filter never writes to the source | holds on generated cases |
| Uncovered pixels stay byte for byte | holds with a feathered mask |
| Undo then redo reproduces the same pixels | holds |
| A recipe replays to the same pixels from the source | holds |
| The specialist reports the source as preserved | holds |

## Measured latency (pure Python, this workstation)

| Operation | 64 x 64 preview | 192 x 192 full |
| --- | --- | --- |
| grayscale | 8.3 ms | 74.6 ms |
| sepia | 18.1 ms | 152.2 ms |
| invert | 3.8 ms | 34.3 ms |
| brightness_contrast | 9.0 ms | 83.1 ms |
| saturation | 9.3 ms | 80.5 ms |
| tint | 6.3 ms | 56.3 ms |
| threshold | 4.3 ms | 37.4 ms |
| masked sepia | 18.2 ms | 162.7 ms |
| stack replay, cached | 0.011 ms | 0.108 ms |
| magic eraser preview | 12.6 ms | 109.4 ms |

The honest reading: a 64 x 64 preview costs between 4 and 18 ms per operation and
a 192 x 192 full-resolution edit between 34 and 163 ms. **This is not
interactive**, and no real-time claim is made. The cached stack replay is the one
fast path, which is why preview recomputation is worth avoiding.

## Neural tier

None was retained for the direct filters, and the benchmark is the reason: every
operation is exact integer arithmetic with a documented formula, so a learned
tier would add parameters without adding a decision. The magic eraser does not
add one either: it reuses the existing inpaint router and patch k-NN, so an
uncertain selection escalates instead of being filled.

Not established: real-time interaction, GPU paths, colour management, level
histograms, curve editing, blend modes, layer compositing and any evaluation on
real game assets. The magic eraser remains bounded to small selections of
locally predictable regions.
