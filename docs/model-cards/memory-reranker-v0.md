Model card: bellium/knn/memory-reranker:v0

- Task: rerank memory titles in one domain.
- Authority: consultative.
- Method: Jaccard on term tokens, weighted by evidence class.
- Tokens: Unicode letters/numbers, casefold and NFC normalization. Accents are
  preserved; composed and decomposed forms match. This is lexical retrieval,
  without transliteration, stemming or Chinese word segmentation.
- Selection: partial top-k, ordered by descending evidence-weighted score,
  descending similarity, then ascending id. Equal scores no longer depend on
  exemplar file order. Files and caller-supplied memories share validation;
  duplicate ids cannot count as independent neighbors.
- Outputs: ids, titles, scores. Bodies, prompts and content are forbidden.
- Repeated queries: optional `PreparedMemory` validates once, snapshots immutable
  records and indexes matching tokens within each domain. Queries preserve exact
  scores, tie order and abstentions. Rebuild explicitly when the source changes;
  the default/list API retains its existing loading and validation behavior.
- Optional prose profile: `profile="fr-en"` filters a fixed list of French/English
  grammatical words from both query and corpus tokens. Default `exact` remains
  unchanged. Prepared queries inherit their immutable snapshot profile; a
  conflicting explicit query profile is rejected. Negations remain lexical tokens,
  but the specialist does not interpret their meaning.
- Limits: domain isolation; one weak neighbor is not enough.
- Evidence: tests/test_tool_and_memory.py, tests/test_memory_rerank_unicode.py,
  and [the small-learning gate](../SMALL_LEARNING_GATE.md).
- Scale evidence: tests/test_prepared_memory.py and
  [the prepared-memory gate](../MEMORY_PREPARED_GATE.md).
- Phrase evidence: [the lexical relevance gate](../MEMORY_RELEVANCE_GATE.md),
  authored development/test cases and tests/test_memory_lexical_profile.py.
