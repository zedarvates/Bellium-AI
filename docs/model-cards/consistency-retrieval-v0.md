Model card: bellium/knn/consistency-retrieval:v0

- Task: retrieve the labelled visual style of a small image in one family.
- Authority: consultative.
- Method: compact colour/structure features, family-isolated k-NN.
- Outputs: a style id. recolor is always false. Pixels are not changed.
- Limits: not style transfer; unknown families and mixed neighbors abstain.
- Evidence: tests/test_consistency.py.

