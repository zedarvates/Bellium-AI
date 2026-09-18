import math

from bellium.contracts.schema import AuthorityMode
from bellium.knn.recording_quality import classify_quality, extract_features, load_quality_frames
from bellium.specialists.recording_quality import assess_recording


def _sine(amp=0.25, period=32, n=64, phase=0.0):
    return [amp * math.sin(2 * math.pi * (i + phase) / period) for i in range(n)]


def _clipped(amp=0.99, period=32, n=64, phase=0.0):
    return [max(-1.0, min(1.0, v)) for v in _sine(amp=amp, period=period, n=n, phase=phase)]


def _noisy(amp=0.2, noise=0.35, period=32, n=64, seed=1):
    out = []
    for i in range(n):
        tone = amp * math.sin(2 * math.pi * i / period)
        hiss = noise * (((i * 19 + seed * 7) % 13) - 6) / 6.0
        out.append(max(-1.0, min(1.0, tone + hiss)))
    return out


def test_clean_sine_is_clean() -> None:
    result = classify_quality({"samples": _sine()})
    assert result.abstained is False
    assert result.output["label"] == "clean"
    assert result.output["audio"] is None
    assert result.output["certified"] is False
    assert result.authority_mode is AuthorityMode.CONSULTATIVE


def test_clipped_sine_is_clipped() -> None:
    result = classify_quality({"samples": _clipped()})
    assert result.output["label"] == "clipped"
    assert result.output["audio"] is None


def test_sine_with_hiss_is_noisy() -> None:
    result = classify_quality({"samples": _noisy()})
    assert result.output["label"] == "noisy"


def test_hybrid_keeps_consultative_label() -> None:
    result = assess_recording({"samples": _sine()})
    assert result.specialist_id == "bellium/hybrid/recording-quality:v0"
    assert result.output["label"] == "clean"
    assert result.output["certified"] is False
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
        classify_quality({"samples": _sine(), "audio_path": "clip.wav"})
    except ValueError:
        blocked = True
    assert blocked


def test_in_memory_waveform_is_rejected() -> None:
    blocked = False
    try:
        classify_quality({"samples": _sine(), "waveform": _sine()})
    except ValueError:
        blocked = True
    assert blocked


def test_raw_audio_cannot_be_kept(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        '{"schema":"bellium.recording-quality-memory/v1","items":[{'
        '"id":"x","label":"clean","source":"x","raw_audio_stored":true,'
        '"features":{"energy":0.5,"clipping":0.0,"noise_floor":0.2,'
        '"periodicity":0.8,"zcr":0.1}}]}',
        encoding="utf-8",
    )
    blocked = False
    try:
        load_quality_frames(path)
    except ValueError:
        blocked = True
    assert blocked
