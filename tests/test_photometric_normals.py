import pytest

from bellium.material.controlled import procedural_albedo, to_image
from bellium.material.photometric import (
    light_directions,
    multilight_capture,
    normal_error,
    photometric_normals,
    reliable_patches,
    reliability_features,
    synthetic_geometry,
    validate_lights,
)
from bellium.specialists.photometric import recover_normals

SIZE = 32
LIGHTS = light_directions(6)


def _albedo(kind="checker", seed=4):
    return [
        [sum(pixel) / 3.0 for pixel in row]
        for row in procedural_albedo(kind, size=SIZE, seed=seed)
    ]


def _images(captures):
    """Grey captures as the repository RGB integer contract."""
    return [to_image([[(value, value, value) for value in row] for row in image])
            for image in captures]


def test_lambertian_planes_and_waves_are_recovered_exactly() -> None:
    for geometry in ("tilted-plane", "waves"):
        normals, mask = synthetic_geometry(geometry, size=SIZE)
        captures = multilight_capture(_albedo(), normals, LIGHTS, ambient=0.05)
        fit = photometric_normals(captures, LIGHTS)
        error = normal_error(fit["normals"], normals, mask)
        assert error["mean_deg"] == pytest.approx(0.0, abs=1e-6)
        assert error["max_deg"] < 1e-6


def test_the_gate_isolates_the_shadowed_curve_of_a_sphere() -> None:
    normals, mask = synthetic_geometry("sphere", size=SIZE)
    captures = multilight_capture(_albedo(), normals, LIGHTS, ambient=0.05)
    fit = photometric_normals(captures, LIGHTS)
    overall = normal_error(fit["normals"], normals, mask)
    patches = reliable_patches(reliability_features(fit["residual"], captures))
    trusted = [index for patch in patches if patch["trusted"] for index in patch["indices"]]
    gated = normal_error(fit["normals"], normals, mask, indices=trusted)
    assert overall["mean_deg"] > 5.0
    assert gated["mean_deg"] < 0.5
    assert 0 < gated["pixels"] < overall["pixels"]


def test_self_shadowed_pixels_carry_a_residual() -> None:
    normals, mask = synthetic_geometry("sphere", size=SIZE)
    captures = multilight_capture(_albedo(), normals, LIGHTS, ambient=0.0)
    fit = photometric_normals(captures, LIGHTS)
    worst = max(
        fit["residual"][r][c]
        for r in range(SIZE)
        for c in range(SIZE)
        if mask[r][c] > 0.0
    )
    assert worst > 0.05


def test_specular_captures_lose_their_trusted_patches() -> None:
    normals, _mask = synthetic_geometry("waves", size=SIZE)
    captures = multilight_capture(_albedo(), normals, LIGHTS, ambient=0.05, specular=0.25)
    fit = photometric_normals(captures, LIGHTS)
    patches = reliable_patches(reliability_features(fit["residual"], captures))
    assert all(not patch["trusted"] for patch in patches)


def test_lights_must_be_declared_and_normalised() -> None:
    with pytest.raises(ValueError, match="at least"):
        validate_lights([[0.0, 0.0, 1.0]] * 3)
    with pytest.raises(ValueError, match="normalize"):
        validate_lights([[0.0, 0.0, 0.0]] * 4)
    normalised = validate_lights([[0.0, 0.0, 2.0]] * 4)
    assert normalised[0] == pytest.approx((0.0, 0.0, 1.0))


def test_capture_validation() -> None:
    normals, _ = synthetic_geometry("waves", size=SIZE)
    captures = multilight_capture(_albedo(), normals, LIGHTS)
    with pytest.raises(ValueError, match="one image per light"):
        photometric_normals(captures[:-1], LIGHTS)
    broken = [row[:] for row in captures[0]]
    broken[0][0] = 1.5
    with pytest.raises(ValueError, match="between 0 and 1"):
        photometric_normals([broken] + captures[1:], LIGHTS)


def test_reliability_features_add_intensity_statistics() -> None:
    normals, _ = synthetic_geometry("waves", size=SIZE)
    captures = multilight_capture(_albedo(), normals, LIGHTS)
    fit = photometric_normals(captures, LIGHTS)
    without = reliability_features(fit["residual"])[0]["features"]
    with_captures = reliability_features(fit["residual"], captures)[0]["features"]
    assert "mean_intensity" not in without
    assert "mean_intensity" in with_captures and "intensity_range" in with_captures
    assert all(0.0 <= value <= 1.0 for value in with_captures.values())


def test_specialist_reports_normals_and_trust() -> None:
    normals, _ = synthetic_geometry("waves", size=SIZE)
    captures = multilight_capture(_albedo(), normals, LIGHTS, ambient=0.05)
    result = recover_normals({
        "captures": _images(captures),
        "lights": [list(direction) for direction in LIGHTS],
    })
    assert result.abstained is False
    assert result.output["method"] == "photometric-stereo-least-squares"
    assert result.output["certified"] is False
    assert result.output["pixels_changed"] == 0
    assert result.output["trusted_coverage"] > 0.5
    assert result.output["mean_residual"] < 0.02


def test_specialist_abstains_on_strong_specular() -> None:
    normals, _ = synthetic_geometry("sphere", size=SIZE)
    captures = multilight_capture(_albedo(), normals, LIGHTS, ambient=0.05, specular=0.25)
    result = recover_normals({
        "captures": _images(captures),
        "lights": [list(direction) for direction in LIGHTS],
    })
    assert result.abstained is True
    assert "no_reliable_patch" in result.output["warnings"]


def test_specialist_requires_declared_lights() -> None:
    normals, _ = synthetic_geometry("waves", size=SIZE)
    captures = multilight_capture(_albedo(), normals, LIGHTS)
    blocked = False
    try:
        recover_normals({
            "captures": _images(captures),
        })
    except ValueError:
        blocked = True
    assert blocked


def test_shadow_band_geometry_is_refused() -> None:
    blocked = False
    try:
        synthetic_geometry("dragon", size=SIZE)
    except ValueError:
        blocked = True
    assert blocked
