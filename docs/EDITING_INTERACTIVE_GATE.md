# P9 second gate: an interactive editing path, measured

The previous gate measured 4 to 18 ms per 64 x 64 operation and 34 to 163 ms at
192 x 192 in pure Python, and stated plainly that this was not interactive. This
gate asks what it takes to become interactive, and answers with numbers instead of
an optimisation story.

    python scripts/benchmark_editing_interactive.py

Report: benchmarks/manifests/editing-interactive-v1.json. The budget is declared
before the measurement: a preview pipeline at or below 33 ms at the 95th
percentile counts as interactive, which is two frames at 60 Hz.

## What was added

| Piece | Behaviour |
| --- | --- |
| Vectorised backend | every filter has an array path when NumPy is importable; NumPy stays optional |
| Lookup tables | per-channel filters use 256-entry tables when NumPy is absent |
| Backend reporting | backend_for says which path a call will take |
| Downsampled preview | area average to a bounded side, reported with its scale and size |
| Preview session | the source is converted once, then repeated previews stay in array space |
| Preview versus export | preview_only, resampled, scale and export_size are always reported separately |

The three paths must agree: a parametrised test compares the vectorised, lookup and
plain paths for every filter and every parameter set. That check found three real
rounding defects, all now fixed: the blend was rounded before clipping instead of
after, the contrast scale had lost its 255 denominator, and a saturated colour
blended differently because the two paths clipped at different moments.

## Measured result

| Size | Operation | Plain p50 | Vectorised p50 | Speedup |
| --- | --- | --- | --- | --- |
| 64 x 64 | sepia | 10.6 ms | 3.7 ms | 2.9 x |
| 64 x 64 | grayscale | 6.0 ms | 3.7 ms | 1.6 x |
| 192 x 192 | sepia | 97.7 ms | 38.1 ms | 2.6 x |
| 512 x 512 | sepia | 814.8 ms | 277.9 ms | 2.9 x |
| 512 x 512 | grayscale | 540.2 ms | 300.4 ms | 1.8 x |

Preview pipeline, 512 x 512 source, three steps, 128 x 128 preview:

| Path | p95 | Interactive |
| --- | --- | --- |
| plain, one shot | 506.0 ms | no |
| vectorised, one shot | 485.2 ms | no |
| **vectorised session, repeated preview** | **14.7 ms** | **yes** |

Peak memory, measured with tracemalloc: 23.1 MB for a full 512 x 512 stack apply
against 14.7 MB for a 128 x 128 preview. The session opens in 455 ms, which is the
one-time conversion of the source list into the array representation.

Reading of the measurement:

- The vectorised backend is 1.5 to 2.9 times faster per operation, but that is not
  what makes editing interactive.
- What makes it interactive is keeping the image in the array representation for
  the whole session. A cold one-shot preview of a 512 x 512 source costs about
  0.5 s because converting the list representation dominates, and no amount of
  filter tuning changes that.
- Repeated previews inside a session land at 14.7 ms, inside the declared budget.
  A preview is still never an export, and the export still runs at full resolution.

Not established: real-time at full resolution, GPU execution, multi-threaded
editing, memory beyond 512 x 512, colour management, and any performance figure on
a machine other than this one. NumPy remains optional: without it the tool falls
back to lookup tables and the plain path, and the session reports backend plain.
