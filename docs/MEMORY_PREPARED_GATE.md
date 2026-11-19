# Prepared lexical memory gate - 9 October 2026

Repeated memory queries spent most of their time validating and copying every
record. Even an unknown domain required a full corpus scan. This increment adds
an optional stdlib-only `PreparedMemory` snapshot for repeated searches.

## API and behavior

```python
from bellium.knn import PreparedMemory, rerank_memory
from bellium.knn.memory_rerank import load_memory

# Validate and index once, then reuse the same immutable snapshot.
memory = PreparedMemory(load_memory("my-memory.json"))
result = rerank_memory(
    {"domain": "bellium", "terms": "mémoire réseau"},
    memory=memory,
    k=5,
)
```

`PreparedMemory()` prepares the bundled fixture memory. Passing a list constructs
a snapshot from that list, including string terms or already tokenized sets.
`len(memory)` reports the record count. Empty snapshots are supported and abstain.

Preparation uses the existing validation contract. Duplicate ids, forbidden
payload fields and invalid evidence are rejected before indexing. Records and
their token sets are copied into read-only mappings and frozensets. Domain records
and inverted token postings are immutable tuples behind read-only mappings.
Repeated canonical tokens share string storage inside a snapshot.

Queries tokenize their input and take the union of matching postings in the
requested domain. They apply the existing Jaccard threshold and evidence weighting,
then the existing partial top-k selection. Excluding zero-overlap records is exact
because they cannot reach the positive similarity threshold. Equal scores retain
the same similarity/id tie-break. Unknown domains and unrelated terms abstain
without scanning all records.

The source list or file can change without changing an existing snapshot. Build a
new `PreparedMemory` explicitly to adopt those changes. No automatic file watching,
global cache, persistence format or private-memory export is introduced. Passing a
list or omitting `memory` still follows the previous validation/loading path.

## Measured repeated-query cost

`scripts/benchmark_memory_rerank.py` creates 1,000, 10,000 and 50,000 authored
records across eight domains. Topics, Unicode terms and four evidence classes are
deterministic. Each query has one warm-up and seven timed runs; p95 is nearest-rank
and, with seven runs, equals the maximum observation. Timing and `tracemalloc`
allocation measurements run separately. These results are local to Windows and
Python 3.13.5 and do not measure retrieval quality on real corpora.

Supported query results from the same comparison run:

| Records | List path p50 | Prepared p50 | List peak traced allocation | Prepared peak traced allocation |
| --- | --- | --- | --- | --- |
| 1,000 | 4.234 ms | 0.044 ms | 691,876 bytes | 6,879 bytes |
| 10,000 | 43.916 ms | 0.2114 ms | 7,093,285 bytes | 13,023 bytes |
| 50,000 | 232.4918 ms | 1.0227 ms | 34,952,189 bytes | 166,287 bytes |

At 50,000 records, unrelated terms take 0.0062 ms and an unknown domain takes
0.0053 ms with the prepared path, compared with 276.1556 ms and 248.9194 ms for
the validating list path. A broad query covers every topic token and therefore
visits all 6,250 records in its domain: p50 4.5252 ms versus 228.0571 ms, with
identical abstention because none reaches the similarity threshold. This confirms
that index selectivity affects cost; it is not a constant-time search guarantee.

Preparation has a separate cost:

| Records | One-time preparation | Incremental retained traced allocation | Peak traced allocation during preparation |
| --- | --- | --- | --- |
| 1,000 | 7.45 ms | 636,826 bytes | 1,244,944 bytes |
| 10,000 | 74.13 ms | 5,706,506 bytes | 11,897,502 bytes |
| 50,000 | 374.96 ms | 28,066,506 bytes | 59,013,462 bytes |

The index remains resident between queries. Traced bytes exclude the already
created source corpus and shared existing strings; they are neither process RSS
nor total resident memory. Construction must be amortized across repeated calls,
and rebuilding per query would lose the benefit. No GPU, VRAM or target-device
memory claim is made.

## Correctness and verification

The benchmark checks exact `SpecialistResult` equivalence, including ids, titles,
scores, confidence, notes and abstention, in all twelve corpus/query combinations.
The baseline report was captured before implementing the snapshot. The comparison
report also includes the current list path, so performance comparisons above come
from one run rather than mixing separate baseline and candidate timings.

Thirteen new tests cover source/file mutation isolation, immutable records and
postings, validation and duplicate support, no repeated loading/validation,
default fixtures, and 64 randomized query/domain/k comparisons against the list
path. The prior independent full-sort top-k test remains in force.

Verification: 1,166 pytest tests pass, 15 internal suites and 14 examples pass,
Ruff and diff --check pass on Windows / Python 3.13.5. No distributed model weights,
commit or publication changed.

```powershell
python scripts/benchmark_memory_rerank.py --output benchmarks/results/memory-scale-prepared-2026-10-09.json
```

Reports: `benchmarks/results/memory-scale-before-2026-10-09.json` and
`benchmarks/results/memory-scale-prepared-2026-10-09.json`.

Next: evaluate retrieval quality and query cost with an independently authored
corpus and varied token/domain distributions before broadening the index to other
k-NN families. The current benchmark proves an exact optimization of lexical
retrieval, not semantic relevance or usefulness of its fixture titles.
