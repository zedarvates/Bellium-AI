Model card: bellium/hybrid/engine-import-validation:v0

- Task: check an atlas plan against a declared import target and return the findings plus the
  manifest an importer would need.
- Authority: consultative. imported and written_files stay false: no engine runs, no file is
  written and certified stays false.
- Profiles: models/imports/target-profiles-v0.json holds four declared contracts (godot-4,
  unity-sprite-atlas, phaser-3-json, generic-regions). Each carries its own source string and a
  note, and the values are conservative declared defaults rather than measurements of an engine.
- Overrides: profile_overrides merges caller values onto a loaded profile and passes through the
  same strict contract; changing the profile id is refused.
- Errors: plan_invalid, unplaced_frame, page_too_large, not_power_of_two, not_square,
  too_many_pages, padding_below_required. Warning: padding_below_recommended.
- Verdicts: ready with warnings, or abstain with import_contract_violated. A manifest that fails
  its own checks abstains with manifest_failed_verification.
- Manifest shapes: godot-atlas-textures, unity-spritesheet (rect converted to a bottom-left
  origin and re-checked), phaser-json (with related_multi_packs on several pages) and
  generic-regions. These are JSON mirrors returned as data; no .tres, .meta or project file is
  produced.
- Measured: benchmarks/manifests/engine-import-v1.json, 21 runs over 5 cases x 4 profiles plus one
  caller-tightened override: 17 ready, 4 abstain, 0 manifest violations, manifests of 1.2 to
  12.4 KB and 0.32 to 1.8 ms per run.
- Limits: no engine was executed, no importer accepted or rejected anything and no pixel was
  written. Compression formats, sprite mesh settings, nine-patch borders, artist pivots, engine
  version differences and platform texture limits are not covered.
- Evidence: tests/test_import_validation.py, scripts/benchmark_import_validation.py.

