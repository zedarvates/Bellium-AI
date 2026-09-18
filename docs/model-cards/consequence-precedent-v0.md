Model card: bellium/knn/consequence-precedent:v0

- Task: retrieve verified precedents for what a proposed action would do in one family.
- Authority: consultative. executes stays false and nothing is launched.
- Families: filesystem, asset-pipeline, engine, deployment, network.
- Features: reversible, scope_ratio, touches_protected, has_backup, dry_run_available,
  blast_radius, idempotent, recent_failures.
- Memory: models/knn/tools/consequence-precedents-v0.json, 60 authored precedents
  (5 families x 3 labels x 4 variants) rebuilt by scripts/build_game_memories.py. Labels come
  from the published risk rule and the builder refuses to write a fixture whose label the rule
  does not reproduce.
- Decision: at least three similar verified precedents must agree; unverified items never
  decide. Mixed labels or too few verified neighbours abstain.
- Limits: authored synthetic precedents, not recorded incidents. No incident ledger, no
  multi-user coordination and no execution evidence was evaluated.
- Evidence: tests/test_consequence.py.
- Evidence validation: disk and injected memories share strict boolean verification,
  unique IDs and the same normalized feature contract. Masked source strings remain valid.
