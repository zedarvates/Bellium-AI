Model card: bellium/knn/grapheme-phoneme:v0

- Task: map a spelling to phones, or phones back to a spelling, in one language.
- Authority: consultative.
- Method: exact labelled pairs first, then Levenshtein k-NN. Unanimous neighbors may suggest.
- Certification: only an exact attested pair. Neighbor suggestions are inferred or weaker.
- Limits: lab orthography, no audio, no invented language, mixed neighbors abstain.
- Evidence: tests/test_grapheme_phoneme.py.

