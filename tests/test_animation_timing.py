import pytest

from bellium.knn._motion import (
    REFERENCE_CURVES,
    hold_runs,
    motion_series,
    progress_samples,
    spacing_variation,
)
from bellium.knn.easing_profile import classify_easing, load_profiles
from bellium.specialists.animation_timing import report_timing


def _series(curve, scale=0.1, holds_end=0):
    transitions = [
        max(0.0, curve[index] - curve[index - 1]) * scale for index in range(1, len(curve))
    ]
    return transitions + [0.0] * holds_end


@pytest.mark.parametrize("name", sorted(REFERENCE_CURVES))
def test_every_published_curve_is_named(name) -> None:
    result = classify_easing({"family": "ui", "series": _series(REFERENCE_CURVES[name])})
    assert result.abstained is False
    assert result.output["easing"] == name
    assert result.output["baseline"] == name


def test_progress_starts_at_the_origin() -> None:
    samples = progress_samples(_series(REFERENCE_CURVES["linear"]))
    assert samples[0] == pytest.approx(0.0)
    assert samples[-1] == pytest.approx(1.0, rel=1e-3)
    assert samples[3] == pytest.approx(0.5, abs=1e-3)


def test_unknown_family_abstains() -> None:
    result = classify_easing({"family": "portrait", "series": _series(REFERENCE_CURVES["linear"])})
    assert result.abstained is True
    assert result.output["reason"] == "unknown_family"


def test_ambiguous_fit_abstains() -> None:
    flat = [0.05] * 7
    result = classify_easing({"family": "ui", "samples": flat})
    assert result.abstained is True
    assert result.output["reason"] in {"ambiguous_easing", "no_curve_close_enough"}


def test_wrong_sample_count_is_refused() -> None:
    blocked = False
    try:
        classify_easing({"family": "ui", "samples": [0.0, 1.0]})
    except ValueError:
        blocked = True
    assert blocked


def test_holds_and_spacing_are_measured() -> None:
    series = [0.2, 0.0, 0.0, 0.3, 0.0]
    runs = hold_runs(series)
    assert runs == [{"start": 1, "length": 2}, {"start": 4, "length": 1}]
    assert spacing_variation([0.2, 0.2, 0.2]) == pytest.approx(0.0)
    assert spacing_variation(series) > 0.5


def test_series_validation_refuses_out_of_range_values() -> None:
    blocked = False
    try:
        motion_series([0.1, 1.4, 0.2])
    except ValueError:
        blocked = True
    assert blocked


def test_timing_report_carries_holds_peak_and_warnings() -> None:
    report = report_timing({"family": "ui", "series": _series(REFERENCE_CURVES["linear"])})
    assert report.output["status"] == "ready"
    assert report.output["easing"] == "linear"
    assert report.output["certified"] is False
    assert report.output["frames_changed"] == 0
    # A linear curve has no peak worth reporting: every transition is equal.
    assert spacing_variation(_series(REFERENCE_CURVES["linear"])) < 0.01
    accelerating = report_timing({"family": "ui", "series": _series(REFERENCE_CURVES["ease-in"])})
    assert accelerating.output["easing"] == "ease-in"
    assert accelerating.output["peak"]["index"] == len(_series(REFERENCE_CURVES["ease-in"])) - 1
    assert "peak_at_last_frame" in accelerating.output["warnings"]


def test_timing_refuses_a_clip_that_is_too_short() -> None:
    report = report_timing({"family": "ui", "series": [0.1, 0.2]})
    assert report.abstained is True
    assert report.output["reason"] == "clip_too_short"


def test_duplicate_frames_are_reported_never_removed() -> None:
    report = report_timing({
        "family": "ui",
        "series": [0.2, 0.3, 0.25, 0.4],
        "frame_mismatches": [0.01, 0.3, 0.0, 0.25],
    })
    assert len(report.output["duplicates"]) == 2
    assert "duplicate_frames" in report.output["warnings"]


def test_mismatch_list_must_match_the_transitions() -> None:
    blocked = False
    try:
        report_timing({"family": "ui", "series": [0.2, 0.3, 0.25],
                       "frame_mismatches": [0.1, 0.2]})
    except ValueError:
        blocked = True
    assert blocked


def test_memory_holds_only_published_curves() -> None:
    profiles = load_profiles()
    assert len(profiles) == 15
    assert all(item["source"].startswith("reference-curve:") for item in profiles)
    assert {item["label"] for item in profiles} == set(REFERENCE_CURVES)
