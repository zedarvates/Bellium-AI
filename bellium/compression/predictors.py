"""Causal byte predictors. Every update uses only a reconstructed byte.

These integer algorithms are part of the BLCP v1 wire contract. Changing them
requires a new format version. State resets every 4096 bytes in both directions.
"""

from collections import deque
from heapq import nsmallest


class KNNPredictor:
    """Three nearest two-byte contexts, in a bounded 64-example online memory."""

    def __init__(self):
        self.context = (0, 0)
        self.memory = deque(maxlen=64)

    def predict(self):
        if not self.memory:
            return self.context[-1]
        a, b = self.context
        neighbors = nsmallest(3, (
            (abs(a - x) + abs(b - y), age, value)
            for age, ((x, y), value) in enumerate(reversed(self.memory))
        ))
        # Integer distances, deterministic recency tie-break, no external memory.
        weights = [1 + 256 // (1 + distance) for distance, _, _ in neighbors]
        return sum(w * item[2] for w, item in zip(weights, neighbors)) // sum(weights)

    def observe(self, value):
        self.memory.append((self.context, value))
        self.context = (self.context[-1], value)


class MicroNNPredictor:
    """One adaptive linear neuron: three inputs plus bias, four Q12 weights.

Integer online gradient updates learn from past bytes during encoding AND
decoding. There are no pretrained weights or floating-point operations.
This is a minimal neural baseline, not a pretrained image/splat autoencoder.
"""

    def __init__(self):
        self.previous = self.older = 128
        self.weights = [4096, 0, 0, 0]

    def features(self):
        return (self.previous - 128, self.older - 128,
                (self.previous + self.older) // 2 - 128, 128)

    def predict(self):
        value = 128 + sum(w * x for w, x in zip(self.weights, self.features())) // 4096
        return max(0, min(255, value))

    def observe(self, value):
        error = value - self.predict()
        self.weights = [max(-32768, min(32768, w + error * x // 128))
                        for w, x in zip(self.weights, self.features())]
        self.older, self.previous = self.previous, value


def residuals(data: bytes, method: str, *, decode=False) -> bytes:
    output = bytearray()
    previous = 0
    predictor = None
    for index, value in enumerate(data):
        if index % 4096 == 0:
            previous = 0
            predictor = (KNNPredictor() if method == "knn" else
                         MicroNNPredictor() if method == "micro-nn" else None)
        prediction = previous if predictor is None else predictor.predict()
        reconstructed = (value + prediction) & 255 if decode else value
        output.append(reconstructed if decode else (value - prediction) & 255)
        if predictor is not None:
            predictor.observe(reconstructed)
        previous = reconstructed
    return bytes(output)
