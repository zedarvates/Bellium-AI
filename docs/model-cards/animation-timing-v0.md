Model card: bellium/hybrid/animation-timing:v0

- Task: report the timing of a clip: holds, peak, easing and duplicate frames.
- Authority: consultative. The report says frames_changed 0: nothing is retimed,
  trimmed, removed or exported.
- Inputs: family plus a transition series or a list of per-frame deltas, and optional
  frame_mismatches between consecutive frames.
- Measures: hold runs with a threshold relative to the strongest transition of the same
  clip (a small sprite changes far fewer pixels than a large one), the peak transition,
  the spacing roughness and the normalised progress curve with leading and trailing holds
  removed, because the easing describes the moving part and not the wait around it.
- Order: deterministic measures first, then curve retrieval through
  bellium/knn/easing-profile:v0, which must agree with the published threshold rule.
- Warnings: no_visible_motion, duplicate_frames, uneven_spacing, peak_at_last_frame,
  no_hold_at_end (ui and effect), easing_not_confirmed.
- Refusals: fewer than three transitions abstains with clip_too_short; a mismatch list
  that does not match the transitions raises; unknown families abstain.
- Duplicates: frames whose mismatch is at or below 0.02 are listed with their value and
  never removed.
- Measured on held-out clips: duplicates 45 of 45 found at every noise level, hold totals
  matching 16 of 20 clean clips, p50 timing 0.33 ms. See the gate record.
- Limits: no frame export, no engine curve, no bezier or spring model, no retiming, and
  no judgement about whether a timing is artistically right.
- Evidence: tests/test_animation_timing.py, scripts/benchmark_animation_timing.py.
