Model card: bellium/knn/visual-anomaly:v0

- Task: detect whether a small image still looks like known states of one domain.
- Authority: consultative. No incident is opened.
- Method: seven compact visual features, domain-isolated k-NN, one-class novelty.
- Inputs: image pixels for scoring, or named features. Memory stores features only.
- Not: the Botte Secrete log anomaly_detector, a camera system, or aquaponics proof.
- Limits: tiny lab images; unknown domains abstain; mixed labels abstain.
- Evidence: tests/test_visual_anomaly.py.
