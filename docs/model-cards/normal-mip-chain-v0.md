Model card: bellium/hybrid/normal-mip-chain:v0

- Task: reduce a normal field into level-of-detail levels and report what each level did to the
  field.
- Authority: consultative and inert. written_files, certified and pixels_changed stay false, false
  and 0: the levels are returned as data and nothing is written back.
- Inputs: normals, optional mask, levels, factor, method and max_pixels. Undocumented query keys are
  refused rather than ignored.
- Methods: renormalized (the default), naive and slopes, published side by side so they can be
  compared instead of trusted. A block counts only when every source pixel under it is valid, so a
  closed silhouette keeps a clean rim instead of a filled one.
- Outputs: the per-level statistics (shape, mean Z, coverage, the length the filter produced), the
  drift of the chain, the levels themselves when the budget allows, and the warnings.
- The budget: pixels_returned is summed before anything is materialized, and a chain above
  max_pixels (512 x 512 by default) returns statistics only with status statistics_only and reason
  chain_too_large. A large source is never silently copied into hundreds of megabytes of lists.
- Measured: reducing a sphere by two costs 0.1772 degrees of direction where the naive form costs
  3.7021 as stored, with the identical 0.1772 direction; slope-space filtering costs 0.4696 and is
  therefore worse here, and it is exactly equal to the renormalized form on a cone. The integral is
  unchanged by all three, because a scale on a normal cancels in the slope.
- The drift: mean Z rises by 0.09489 over two levels of a sphere, 0.00606 for a cone and 0.00000 for
  a plane. The declared limit is 0.03, and crossing it raises flattening_drift and lowers the
  confidence to 0.5. That drift is the flattening a mip chain shows.
- Warnings: flattening_drift, partial_coverage, chain_stopped_early, chain_too_large.
- Limits: controlled 32 x 32 fields, a box average with an integer factor, no kernel, no anisotropic
  footprint, no UV-aware wrap, no hardware mip generation and no compressed format. A length defect
  in the source is not repaired: the specialist reports it and leaves the choice to the caller.
- Evidence: tests/test_normal_resample.py, scripts/benchmark_normal_resample.py,
  docs/NORMAL_RESAMPLE_GATE.md.

