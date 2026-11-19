# Lexical phrase retrieval gate - 9 October 2026

The prepared-memory gate improved query cost while preserving exact results.
This increment addresses phrase queries, for which grammatical words enlarge the
Jaccard union and can prevent otherwise relevant titles from meeting the threshold.

## Explicit profile

```python
from bellium.knn import PreparedMemory, rerank_memory

memory = PreparedMemory(profile="fr-en")
result = rerank_memory(
    {"domain": "bellium", "terms": "How can I find the image cutout for this background?"},
    memory=memory,
)
```

`profile="fr-en"` removes a fixed list of common grammatical words from both
query and memory tokens. The list is declared in `bellium/knn/memory_rerank.py`;
it contains no domain-specific retrieval keywords and has no learned parameters.
The default `exact` profile retains the existing behavior and output fields.
There is no stemming, transliteration, translation, synonym expansion or embedding.

List queries can explicitly pass `profile="fr-en"`. A prepared snapshot fixes
its profile at construction, and queries inherit it. Passing a different explicit
profile is rejected rather than mixing corpus and query preprocessing. Rebuild
the snapshot to adopt a different profile. Titles, ids, domain isolation, evidence
weights, minimum neighbor support and similarity threshold are unchanged.

Filtered results identify `lexical_profile: fr-en`. A query containing only removed
words abstains with reason=no_content_terms. A record left with no content terms
is rejected during preparation or list validation; it cannot count as support.

Negations such as pas, sans, non, not, no, never and without are retained, as are
uncertainty words, numeric tokens and identifiers. This prevents deletion of those
tokens; it does not establish understanding of negation. For example, `not image
cutout background` can still retrieve related titles, at lower lexical similarity.
The profile is opt-in because grammatical-looking words can be meaningful labels
or units in a technical corpus. Filtering can increase false suggestions on corpora
not covered by this gate. Similarity/confidence remains uncalibrated.

## Authored evaluation

The original corpus contains 24 title/term records: six topics, two records per
topic, in each of two domains (French and English). It includes attested and
reconstructed evidence. Query cases were stored separately before implementation:
18 development cases and 18 test cases, each with 12 positives and six expected
abstentions. Cases include natural phrase wording, unknown domains, unrelated
requests, vague terms, unsupported paraphrases and cross-domain terms.

The same agent authored the corpus and relevance labels. These are controlled
lexical fixtures, not independent human annotation, user logs or field-quality
evidence. The final cases share the corpus and topics with development; they test
new wording, not generalization to unseen topics. No private memory is included.

Thresholds were declared before evaluation: at least 80% positive top-id hits and
zero suggestions on negative cases. Development observed exact-profile failures;
the fixed grammatical-word profile was implemented before assessing final outcomes.
The profile was not retuned after those final outcomes.

| Split / profile | Correct positive top ids | Negative suggestions | Gate |
| --- | --- | --- | --- |
| Development / exact | 1/12 | 0/6 | Fail |
| Development / fr-en | 12/12 | 0/6 | Pass |
| Test / exact | 2/12 | 0/6 | Fail |
| Test / fr-en | 12/12 | 0/6 | Pass |

An abstention cannot count as a successful accepted hit even if partial neighbors
are returned. On final cases, accepted-neighbor precision is 1.0 for both profiles,
while positive coverage and mean accepted recall at five improve from 2/12 to
12/12. The benefit here is reduced positive abstention, not a demonstrated increase
in precision. Six negative cases cannot establish a general false-positive rate.

The evaluation script records corpus, case, reranker and script SHA-256 hashes.
The final corpus/case hashes are:

- Corpus: e740f4c6092989bc67fb13eb6d712ef521992a5702f3fb748440491568cf8232
- Cases: 9a2ad5e05c6e8e7f97367f8e28b92f6bc7690b64924c0728d24c56c13da186fc

## Reproduction and verification

```powershell
python scripts/evaluate_memory_retrieval.py --split test --profiles exact fr-en --output benchmarks/results/memory-relevance-test-2026-10-09.json
python -m examples.memory_reranking
```

The example uses the bundled memory independently of the evaluation corpus. The
exact profile abstains on its phrase, fr-en proposes mem_cutout/mem_white, and an
unknown domain abstains. It performs no operation besides local retrieval.

Files: `benchmarks/fixtures/memory-retrieval-corpus-v1.json`,
`benchmarks/fixtures/memory-retrieval-cases-v1.json`,
`scripts/evaluate_memory_retrieval.py`, `examples/memory_reranking.py`.
Reports are in `benchmarks/results/memory-relevance-*-2026-10-09.json`.

Verification on Windows/Python 3.13.5: 1,181 pytest tests pass, 15 internal suites
and 14 example tests pass, Ruff and diff --check pass, and the new example runs
successfully. Fifteen new tests verify profile consistency, source preservation,
negations/identifiers, empty-content handling, immutable policy, domain isolation
and evaluation denominators. Existing exact-profile/index equivalence tests pass.

No model weights, commit or publication changed. Next: verify packaged APIs and
resources from a freshly built wheel in an isolated environment, then consider a
new independently labeled corpus. These observed final cases should now be treated
as regression fixtures if the profile changes further.
