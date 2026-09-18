"""Temperature scaling — calibrate micro-NN confidence so the abstention threshold
is meaningful (Hermes audit #3).

The micro-NNs are MSE-trained and output softmax probabilities that are usually
over- or under-confident, so a raw 0.66 cutoff is arbitrary. A single scalar
temperature T per model, fit on labeled data, rescales the probabilities:

    calibrated = softmax(log(p) / T)          # the constant in log p cancels

T > 1 softens an over-confident model; T < 1 sharpens an under-confident one. It is
fit by minimizing negative log-likelihood on held-out labeled data — which the
active-learning loop supplies over time (`calibrate_from_logs`). Pure stdlib.

A calibration is stored next to the model as models/<name>.calib.json; absent file
→ T = 1.0 (identity), so nothing changes until a model is actually calibrated.
"""

from __future__ import annotations

import json
import math
import hashlib
import os
import tempfile
from pathlib import Path
from typing import Optional

_MODELS_DIR = Path(__file__).resolve().parent / "models"
_EPS = 1e-12
_cache: dict[str, float] = {}


def apply_temperature(probs: list[float], temperature: float) -> list[float]:
    """Rescale a probability vector by temperature T via softmax(log(p)/T)."""
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("temperature must be positive and finite")
    if any(
        isinstance(p, bool)
        or not isinstance(p, (int, float))
        or not math.isfinite(p)
        or not 0 <= p <= 1
        for p in probs
    ):
        raise ValueError("probabilities must be finite numbers in [0, 1]")
    if temperature == 1.0 or not probs:
        return list(probs)
    logits = [math.log(max(p, _EPS)) / temperature for p in probs]
    hi = max(logits)
    exps = [math.exp(value - hi) for value in logits]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


def nll(probs_list: list[list[float]], labels: list[int], temperature: float) -> float:
    """Mean negative log-likelihood of the true class after temperature scaling."""
    if not labels:
        return 0.0
    total = 0.0
    for probs, y in zip(probs_list, labels):
        cal = apply_temperature(probs, temperature)
        total += -math.log(max(cal[y], _EPS))
    return total / len(labels)


def fit_temperature(probs_list: list[list[float]], labels: list[int]) -> float:
    """Find the T minimizing NLL via coarse→fine grid search (no SciPy needed)."""
    if not probs_list:
        return 1.0
    best_t, best = 1.0, nll(probs_list, labels, 1.0)
    for t in (0.5, 0.7, 0.85, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
        v = nll(probs_list, labels, t)
        if v < best:
            best, best_t = v, t
    for d in (-0.15, -0.1, -0.05, 0.05, 0.1, 0.15):  # refine around the winner
        t = round(best_t + d, 3)
        if t <= 0.05:
            continue
        v = nll(probs_list, labels, t)
        if v < best:
            best, best_t = v, t
    return round(best_t, 3)


def expected_calibration_error(
    probs_list: list[list[float]], labels: list[int], *, bins: int = 10
) -> float:
    """ECE — mean gap between confidence and accuracy across confidence bins.

    0 = perfectly calibrated (a 0.8-confident batch is right 80% of the time)."""
    if not labels:
        return 0.0
    buckets: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for probs, y in zip(probs_list, labels):
        conf = max(probs)
        pred = max(range(len(probs)), key=probs.__getitem__)
        idx = min(bins - 1, int(conf * bins))
        buckets[idx].append((conf, 1 if pred == y else 0))
    ece = 0.0
    for bucket in buckets:
        if not bucket:
            continue
        avg_conf = sum(c for c, _ in bucket) / len(bucket)
        acc = sum(h for _, h in bucket) / len(bucket)
        ece += (len(bucket) / len(labels)) * abs(avg_conf - acc)
    return round(ece, 4)


# ── persistence ────────────────────────────────────────────────────────────────


def _calib_path(model_name: str) -> Path:
    if not model_name or any(
        ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for ch in model_name
    ):
        raise ValueError("invalid model name")
    return _MODELS_DIR / f"{model_name}.calib.json"


def save_temperature(
    model_name: str,
    temperature: float,
    *,
    ece_before: float = 0.0,
    ece_after: float = 0.0,
    samples: int = 0,
) -> None:
    apply_temperature([0.5, 0.5], temperature)
    destination = _calib_path(model_name)
    model = _MODELS_DIR / f"{model_name}.json"
    payload = json.dumps(
        {
            "temperature": round(float(temperature), 3),
            "ece_before": ece_before,
            "ece_after": ece_after,
            "samples": samples,
            "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest()
            if model.exists()
            else None,
        },
        indent=2,
        allow_nan=False,
    )
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    _cache[model_name] = float(temperature)


def load_temperature(model_name: str) -> float:
    """Calibrated T for a model, or 1.0 (identity) if it was never calibrated."""
    try:
        payload = json.loads(_calib_path(model_name).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return 1.0
    t = payload["temperature"]
    apply_temperature([0.5, 0.5], t)
    model = _MODELS_DIR / f"{model_name}.json"
    if (
        model.exists()
        and payload.get("model_sha256") != hashlib.sha256(model.read_bytes()).hexdigest()
    ):
        raise ValueError("calibration does not match model bytes")
    _cache[model_name] = t
    return t


def calibrate(model_name: str, probs_list: list[list[float]], labels: list[int]) -> dict:
    """Fit + persist a temperature from labeled (probs, label) data. Returns a report."""
    before = expected_calibration_error(probs_list, labels)
    t = fit_temperature(probs_list, labels)
    after = expected_calibration_error([apply_temperature(p, t) for p in probs_list], labels)
    save_temperature(model_name, t, ece_before=before, ece_after=after, samples=len(labels))
    return {
        "model": model_name,
        "temperature": t,
        "ece_before": before,
        "ece_after": after,
        "samples": len(labels),
    }


def calibrate_from_logs(model_name: str) -> Optional[dict]:
    """No log adapter is shipped in Bellium. Use calibrate() with verified data.

    Kept as a compatibility entry point: unavailable log data never triggers a
    calibration, reads another project's private logs, or changes model weights.
    """
    _calib_path(model_name)  # validate the requested identifier
    return None
