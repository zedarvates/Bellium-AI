# Animation timing gate: easing, holds and duplicate frames

The gate completes the animation part of P4: after the grid, the phases and the loop
check, a clip now gets a timing report. Measurements run on held-out clips built from the
published curves with unseen seeds and hold patterns.

    python scripts/benchmark_animation_timing.py

Report: benchmarks/manifests/animation-timing-v1.json.

## Measured result

| Noise on the transitions | Clips | Easing named | Published rule | Abstentions | Duplicates found | Hold totals matched | p50 timing |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.00 | 20 | 12 | 15 | 7 | 45 of 45 | 16 of 20 | 0.33 ms |
| 0.02 | 20 | 12 | 14 | 6 | 45 of 45 | 14 of 20 | 0.34 ms |
| 0.05 | 20 | 8 | 11 | 9 | 45 of 45 | 11 of 20 | 0.35 ms |

Reading of the measurement:

- Duplicate detection is exact on this set: 45 of 45 across all noise levels.
- The published threshold rule names more curves than the retrieval, because a clip with
  only four or five transitions cannot support a seven-point curve fit: the retrieval
  abstains and the rule still answers. Abstention is the intended behaviour, and the
  report says which tier answered.
- Hold totals match on the clean clips where the relative threshold separates a real
  hold from slow motion. Both the relative threshold and the hold-trimmed progress curve
  are fixes that came out of this measurement, not from reading the code.
- The neural candidate was measured and not shipped: a 57-parameter network reached 0.477
  agreement against 0.658 for the published rule under noise, so no weights were written.

## What changed during the gate

| Change | Reason |
| --- | --- |
| Progress curve starts at the origin | The first measured sample was the first transition, so no clip could match a reference curve |
| Hold threshold made relative to the clip | A fixed absolute threshold called every frame of a small clip a hold |
| Easing computed on the moving part only | A trailing hold flattened the curve and hid the easing |
| Neural easing candidate measured before shipping | The rule was better under noise, so no model was written |

Not established: artistically correct timing, engine curve import, bezier or spring
easings, retiming or trimming, and any judgement on real game clips. Repeat periods and
timings are frame counts on the tested clips.
