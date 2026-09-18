Model card: bellium/hybrid/sprite-sheet-prep:v0

- Task: measure a sprite sheet, split it into frames, name each frame phase and check the loop.
- Authority: consultative. Frames are returned as boxes in a report; the source image is
  never modified and nothing is exported to disk or to an engine.
- Grid: the background comes from the declared colour or the most common border colour. The
  grid is accepted only when the content runs and the repeat period agree, or when the caller
  declares a cell size that the measured content respects. Otherwise the specialist abstains
  with grid_not_confirmed and the measured evidence.
- Agreement: frame phases come from bellium/nano-nn/frame-phase:v0 (baseline_only today) and
  the loop verdict from bellium/knn/clip-loop:v0. Warnings cover mixed_background,
  single_row_assumed, empty_frames, no_visible_motion, no_peak_frame, peak_at_last_frame and
  loop_not_confirmed.
- Pixels: returned per frame only when include_pixels is true.
- Limits: geometry and timing report only. No atlas packing, no pivot convention, no engine
  import, no duplicate-frame removal and no artist review. Sheets whose sprites leave their
  cell are refused, because the grid cannot be confirmed from the image alone.
- Evidence: tests/test_sprite_sheet.py.
