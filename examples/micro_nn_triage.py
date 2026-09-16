"""Consult an imported micro-NN using a synthetic error, without remediation."""

import json
import math

from legacy.micro_nn.features import classify, error_classifier_values


def suggest_error_label(error_text: str, exit_code: int = 1, threshold: float = 0.8) -> dict:
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and between 0 and 1.")
    values = error_classifier_values(error_text, exit_code=exit_code)
    label, score, probabilities = classify("error_classifier", values)
    return {
        "model": "legacy/error_classifier",
        "mode": "consultative",
        "suggested_label": label,
        "accepted_label": label if score >= threshold else None,
        "score": score,
        "threshold": threshold,
        "abstained": score < threshold,
        "probabilities": probabilities,
        "action_executed": False,
        "limits": "Model scores are not calibrated confidence; validate on your own labeled data.",
    }


if __name__ == "__main__":
    print(json.dumps(suggest_error_label("FileNotFoundError: demo-input.txt does not exist"),
                     indent=2))
