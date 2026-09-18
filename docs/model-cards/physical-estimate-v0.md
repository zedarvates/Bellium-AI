Model card: bellium/hybrid/physical-estimate:v0

- Task: one entry point for gravity, atmosphere pressure, atmosphere density and
  hydrostatic pressure previews.
- Authority: consultative. The reference answers whenever the caller declares it
  available, and every result carries its unit, method, assumptions, valid range,
  provenance and uncertainty.
- Order: reference first; the cheaper tiers are computed beside it and reported with
  their live deviation, so a caller can see what the approximation would have said.
  Nothing is promoted by this result.
- Provisional path: with reference_available false the best permitted tier answers, the
  result is marked provisional and it carries the declared band and the published worst
  case. It abstains with no_available_method when nothing may answer.
- Flags: reference_available, allow_regressor and allow_case_table must all be booleans;
  unknown families abstain and missing inputs raise.
- Limits: preview estimates only. No humidity, contact, coupled-force, terrain, weather
  or drag coefficient is estimated, and no physical coefficient is inferred from
  appearance.
- Evidence: tests/test_physical_specialist.py.
