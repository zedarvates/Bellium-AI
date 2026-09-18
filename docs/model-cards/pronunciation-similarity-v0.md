Model card: bellium/knn/pronunciation-similarity:v0

- Task: compare a learner phone sequence to a labelled reference.
- Authority: consultative.
- Method: DTW on shared distinctive features, then word-form k-NN.
- Certification: only an attested exact match can be certified.
- Limits: no raw audio, no private speakers, reconstructed references stay uncertified.
- Evidence: tests/test_language_knn.py.

