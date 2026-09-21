# P4 engine-import gate: a declared contract, findings and manifest mirrors

The gate asks what can be checked before an atlas reaches an engine. The honest answer is:
the geometry against a contract the caller declares, plus the manifest an importer would
need, returned as data. No engine runs here, and every profile says so about itself.

    python scripts/benchmark_import_validation.py

Report: benchmarks/manifests/engine-import-v1.json.

## What exists

| Piece | Behaviour |
| --- | --- |
| target profiles | four declared contracts in models/imports/target-profiles-v0.json, each with a source string, a note and conservative limits |
| profile_overrides | merges caller values onto a loaded profile through the same strict contract; changing the id is refused |
| validate_plan | errors: plan_invalid, unplaced_frame, page_too_large, not_power_of_two, not_square, too_many_pages, padding_below_required; warning: padding_below_recommended |
| build_manifest | JSON mirrors in four shapes, including the Unity bottom-left rect conversion |
| check_manifest | recomputes every region against the plan, the origin convention, the page file names and the engine_verified flag |
| validate_import | one call: pack, validate, mirror. ready with warnings, or abstain with import_contract_violated |

Neither the specialist nor the manifest writer touches a file: the caller receives the mirror
as a dictionary, and `written_files` stays false.

## Declared profiles

| id | manifest shape | max texture | power of two | square | required / recommended padding | pages per asset |
| --- | --- | --- | --- | --- | --- | --- |
| godot-4 | godot-atlas-textures | 8192 | no | no | 0 / 1 | 1 |
| unity-sprite-atlas | unity-spritesheet | 8192 | no | no | 0 / 2 | 1 |
| phaser-3-json | phaser-json | 4096 | no | no | 0 / 1 | 1 |
| generic-regions | generic-regions | 8192 | no | no | 0 / 0 | 64 |

These are conservative defaults declared by this repository, not verified engine limits. The
override path exists because a real project must replace them with its own platform values.

## Measured

| Case | Frames | Page | godot-4 | unity-sprite-atlas | phaser-3-json | generic-regions |
| --- | --- | --- | --- | --- | --- | --- |
| uniform_16 | 64 x 16x16 | 128 | ready, 1 warning | ready, 1 warning | ready, 1 warning | ready |
| mixed_sprite | 12 mixed | 128 | ready, 1 warning | ready, 1 warning | ready, 1 warning | ready |
| mixed_sprite_padded | 12 mixed, padding 2 | 128 | ready | ready | ready | ready |
| measured_96 | 14 mixed | 96 | ready, 1 warning | ready, 1 warning | ready, 1 warning | ready |
| three_pages | 24 x 32x32 | 128, 3 pages | abstain: too_many_pages | abstain: too_many_pages | abstain: too_many_pages | ready |
| measured_96, tightened override | 14 mixed | 96, max 48, power of two, required padding 3 | abstain: page_too_large, not_power_of_two, padding_below_required | | | |

21 runs in total: 17 ready, 4 abstain, 0 manifest violations, manifests between 1.2 and 12.4 KB,
0.32 to 1.8 ms per run. The warnings are the recommended padding: a plan packed with padding 0
satisfies the declared requirement but misses the bleed recommendation, and the report says
padding 2 clears it.

## The honest reading

What this gate proves is internal: the plan satisfies the contract it was checked against, the
mirror reproduces the plan, and the origin conversion is recomputed rather than trusted. The
counted abstentions are real behaviour: a three-page set is refused by every single-texture
profile and accepted by the multi-page one, and a caller that tightens its own contract to a
48-pixel, power-of-two, padded page gets three errors instead of a plan.

What it does not prove is that any engine accepts the mirror. No importer was run, no project
was opened, and no compressed or platform-specific texture format was produced. The profiles
are declarations with provenance, and the caller owns their accuracy.

Not established: engine execution, platform texture limits (iOS power-of-two rules, ASTC or ETC2
block alignment), sprite meshes, nine-patch borders, artist pivots, trim and rotation paths,
duplicate-frame removal, and any measurement on real project sheets. Rasterization and the
engine round trip remain open in P4.

