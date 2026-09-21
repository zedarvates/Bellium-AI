"""Clip-level voice activity. Windows samples, never stores audio."""

from __future__ import annotations

from typing import Any

from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.knn.voice_activity import classify_voice

SPECIALIST_ID = "bellium/hybrid/voice-activity:v0"
FRAME = 64


def detect_voice(query: dict[str, Any]) -> SpecialistResult:
    samples = query.get("samples")
    if samples is None:
        result = classify_voice(query)
        return SpecialistResult(
            SPECIALIST_ID,
            {**result.output, "speech_frames": None, "total_frames": 1 if not result.abstained else 0},
            result.confidence, result.abstained, AuthorityMode.CONSULTATIVE,
            notes=result.notes,
        )
    if not isinstance(samples, list):
        raise ValueError("samples must be a list")
    frames = []
    start = 0
    while start + FRAME <= len(samples):
        frames.append(samples[start:start + FRAME])
        start += FRAME
    if not frames:
        frames.append(samples)
    labels = []
    for frame in frames:
        item = classify_voice({"samples": frame})
        if item.abstained:
            labels.append(None)
        else:
            labels.append(item.output.get("label"))
    speech = labels.count("speech")
    silence = labels.count("silence")
    total = len(labels)
    if speech == 0 and silence == 0:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "no_classifiable_frame", "label": None,
             "speech_frames": 0, "total_frames": total, "audio": None, "certified": False},
            0.0, True, AuthorityMode.CONSULTATIVE,
        )
    if speech == silence:
        return SpecialistResult(
            SPECIALIST_ID,
            {"status": "abstain", "reason": "tied_speech_silence", "label": None,
             "speech_frames": speech, "total_frames": total, "audio": None, "certified": False},
            0.0, True, AuthorityMode.CONSULTATIVE,
        )
    label = "speech" if speech > silence else "silence"
    return SpecialistResult(
        SPECIALIST_ID,
        {"status": "suggest", "label": label, "speech_frames": speech,
         "total_frames": total, "audio": None, "certified": False},
        min(0.9, max(speech, silence) / total),
        False, AuthorityMode.CONSULTATIVE,
        notes=("Windowed k-NN vote. Audio is discarded after features.",),
    )

