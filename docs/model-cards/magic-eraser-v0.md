Model card: bellium/hybrid/magic-eraser:v0

- Task: propose a bounded fill for a selected region of an image.
- Authority: consultative and preview-only. The candidate is returned as a new image,
  certified false, and the source is untouched.
- Order: the selection is thresholded at 0.5 into the inpaint mask, the existing inpaint
  router decides whether the bounded patch method applies, and patch k-NN produces the
  candidate. Nothing runs when the router does not confirm the method.
- Refusals: an empty selection abstains with empty_selection; a router verdict of
  escalate abstains with escalate; a candidate that cannot be confirmed locally abstains
  with candidate_not_confirmed.
- Measured: 12.6 ms for a 64 x 64 preview and 109 ms at 192 x 192 on this workstation.
- Limits: small selections of locally predictable regions only. Photographic object
  removal, large holes, textured backgrounds and semantic fill are not supported, and the
  candidate must be compared before applying.
- Evidence: tests/test_editing.py, scripts/benchmark_editing.py.
