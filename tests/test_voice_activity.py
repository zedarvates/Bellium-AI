import math

from bellium.knn.voice_activity import classify_voice, extract_features, load_voice_frames
from bellium.specialists.voice_activity import detect_voice


def _sine(amp=0.25, period=32, n=64, phase=1.0):
    return [amp * math.sin(2 * math.pi * (i + phase) / period) for i in range(n)]


def _hush(n=64, amp=0.002, seed=1):
    return [amp * (((i * 17 + seed * 9) % 11) - 5) / 5.0 for i in range(n)]


def test_periodic_energy_is_speech() -> None:
    result = classify_voice({"samples": _sine()})
    assert result.abstained is False
    assert result.output["label"] == "speech"
    assert result.output["audio"] is None
    assert result.output["certified"] is False


def test_near_silence_is_silence() -> None:
    result = classify_voice({"samples": _hush()})
    assert result.output["label"] == "silence"


def test_clip_vote_counts_speech_frames() -> None:
    result = detect_voice({"samples": _sine() + _sine()})
    assert result.output["label"] == "speech"
    assert result.output["speech_frames"] == 2
    assert result.output["audio"] is None


def test_unknown_sample_is_not_zeroed() -> None:
    blocked = False
    try:
        extract_features([0.1] * 31 + [None])
    except ValueError:
        blocked = True
    assert blocked


def test_waveform_path_is_rejected() -> None:
    blocked = False
    try:
        classify_voice({"samples": _sine(), "audio_path": "clip.wav"})
    except ValueError:
        blocked = True
    assert blocked


def test_raw_audio_cannot_be_kept(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        '{"schema":"bellium.voice-activity-memory/v1","items":[{'
        '"id":"x","label":"speech","source":"x","raw_audio_stored":true,'
        '"features":{"energy":0.5,"zcr":0.1,"periodicity":0.8,'
        '"crest":0.2,"high_ratio":0.1}}]}',
        encoding="utf-8",
    )
    blocked = False
    try:
        load_voice_frames(path)
    except ValueError:
        blocked = True
    assert blocked

