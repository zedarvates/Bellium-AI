"""Height from a normal field: relative only, and the offset is reported.

Every number asserted here was measured on the controlled geometries by
scripts/benchmark_normal_integration.py; the report beside it holds the same
table. Nothing in this file claims a real capture was ever integrated.
"""

import math

import pytest

from bellium.contracts.schema import AuthorityMode
from bellium.material.controlled import procedural_albedo, to_image
from bellium.material.integration import (
    DEFAULT_METHOD,
    METHODS,
    analytic_height,
    gradients,
    height_error,
    integrate_height,
)
from bellium.material.photometric import (
    light_directions,
    multilight_capture,
    synthetic_geometry,
)
from bellium.specialists.photometric import recover_height, recover_normals

SIZE = 32
LIGHTS = light_directions(6)
HEIGHT_SPECIALIST_ID = "bellium/hybrid/normal-to-height:v0"


def _case(geometry, size=SIZE):
    normals, mask = synthetic_geometry(geometry, size=size)
    truth, truth_mask = analytic_height(geometry, size=size)
    scale = 1.0 if geometry == "waves" else 1.0 / (size * 0.45)
    return normals, mask, truth, truth_mask, scale


def _relative(geometry, method, **options):
    normals, mask, truth, truth_mask, scale = _case(geometry)
    result = integrate_height(
        normals, mask, method=method, pixel_scale=scale, **options
    )
    return result, height_error(result["height"], truth, truth_mask)


def test_methods_and_default_are_declared() -> None:
    assert DEFAULT_METHOD == "cumulative-average"
    assert set(METHODS) == {
        "least-squares",
        "cumulative",
        "cumulative-average",
        "cumulative-vertical",
    }


def test_least_squares_is_exact_on_a_constant_slope() -> None:
    result, error = _relative("tilted-plane", "least-squares", iterations=10)
    assert error["relative_rmse"] == 0.0
    assert error["max_abs"] == 0.0
    assert result["convergence"]["mean_slope_restored"] == [0.35, 0.2]


def test_vertical_alignment_is_exact_on_a_constant_slope_without_solving() -> None:
    result, error = _relative("tilted-plane", "cumulative-vertical")
    assert error["relative_rmse"] == 0.0
    assert result["convergence"] is None


def test_row_alignment_loses_the_vertical_slope_of_a_constant_ramp() -> None:
    # Documented negative result: identical rows carry no vertical signal for an
    # alignment that compares row integrals, so the slope is dropped.
    _, error = _relative("tilted-plane", "cumulative")
    assert error["relative_rmse"] > 0.05


def test_the_row_column_average_wins_on_a_curved_silhouette() -> None:
    for geometry in ("sphere", "cone"):
        _, average = _relative(geometry, "cumulative-average")
        _, rows = _relative(geometry, "cumulative")
        _, vertical = _relative(geometry, "cumulative-vertical")
        assert average["relative_rmse"] < rows["relative_rmse"]
        assert average["relative_rmse"] < vertical["relative_rmse"]


def test_vertical_alignment_wins_on_an_oscillating_surface() -> None:
    _, vertical = _relative("waves", "cumulative-vertical")
    _, rows = _relative("waves", "cumulative")
    assert vertical["relative_rmse"] < rows["relative_rmse"]


def test_least_squares_reports_convergence_instead_of_assuming_it() -> None:
    normals, mask, _, _, _ = _case("sphere")
    stopped = integrate_height(normals, mask, method="least-squares", iterations=5)
    convergence = stopped["convergence"]
    assert convergence["iterations"] == 5
    assert convergence["converged"] is False
    assert convergence["last_change"] > convergence["tolerance"]
    relaxed = integrate_height(
        normals, mask, method="least-squares", iterations=5, tolerance=1e6
    )
    assert relaxed["convergence"]["converged"] is True


def test_the_unrecoverable_offset_is_removed_from_the_error() -> None:
    normals, mask, truth, truth_mask, scale = _case("sphere")
    result = integrate_height(
        normals, mask, method="cumulative-average", pixel_scale=scale
    )
    shifted = [[value + 100.0 for value in row] for row in truth]
    plain = height_error(result["height"], truth, truth_mask)
    lifted = height_error(result["height"], shifted, truth_mask)
    assert lifted["offset"] == pytest.approx(plain["offset"] - 100.0, abs=1e-4)
    assert lifted["rmse"] == pytest.approx(plain["rmse"], abs=1e-6)
    assert lifted["max_abs"] == pytest.approx(plain["max_abs"], abs=1e-6)
    assert lifted["relative_rmse"] == pytest.approx(plain["relative_rmse"], abs=1e-6)


def test_height_error_refuses_to_divide_by_a_flat_reference() -> None:
    error = height_error(
        [[3.0] * 4 for _ in range(4)], [[7.0] * 4 for _ in range(4)], [[1.0] * 4 for _ in range(4)]
    )
    assert error["rmse"] == 0.0
    assert error["relative_rmse"] is None


def test_height_error_without_a_valid_pixel_reports_nothing() -> None:
    error = height_error([[1.0]], [[2.0]], [[0.0]])
    assert error["pixels"] == 0
    assert error["rmse"] is None
    assert error["max_abs"] is None
    assert error["offset"] is None


def test_the_pixel_scale_is_declared_and_scales_the_integral() -> None:
    normals, mask, _, _, _ = _case("cone")
    single = integrate_height(normals, mask, method="cumulative-average")
    double = integrate_height(
        normals, mask, method="cumulative-average", pixel_scale=2.0
    )
    assert single["pixel_scale"] == 1.0
    assert double["pixel_scale"] == 2.0
    assert double["offset_removed"] == pytest.approx(
        single["offset_removed"] * 2.0, abs=1e-5
    )
    for row_single, row_double in zip(single["height"], double["height"]):
        for value_single, value_double in zip(row_single, row_double):
            assert value_double == pytest.approx(2.0 * value_single, abs=1e-5)


def test_edge_on_and_missing_normals_carry_no_slope() -> None:
    normals = [
        [(0.6, 0.0, 0.0), None, (0.0, 0.0, 0.0)],
        [(0.0, 0.5, 0.0), (1.0, 0.0, 1e-9), (0.0, 1.0, 0.0)],
    ]
    slope_x, slope_y, valid = gradients(normals, None)
    assert valid == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert slope_x == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert slope_y == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    for method in METHODS:
        result = integrate_height(normals, None, method=method)
        assert result["pixels"] == 0
        assert result["height"] == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


def test_masked_pixels_stay_at_zero_and_unmeasured() -> None:
    normals, mask, _, _, _ = _case("sphere")
    result = integrate_height(normals, mask, method="cumulative-average")
    for r, c in ((0, 0), (0, SIZE - 1), (SIZE - 1, 0), (SIZE - 1, SIZE - 1)):
        assert result["valid"][r][c] == 0.0
        assert result["height"][r][c] == 0.0


def test_integration_is_deterministic() -> None:
    normals, mask, _, _, scale = _case("waves")
    for method in METHODS:
        first = integrate_height(
            normals, mask, method=method, iterations=20, pixel_scale=scale
        )
        second = integrate_height(
            normals, mask, method=method, iterations=20, pixel_scale=scale
        )
        assert first == second


@pytest.mark.parametrize("method", ["row-by-row", "", "CUMULATIVE", None, 3])
def test_unknown_methods_are_refused(method) -> None:
    with pytest.raises(ValueError, match="method must be one of"):
        integrate_height([[(0.0, 0.0, 1.0)]], method=method)


@pytest.mark.parametrize(
    "scale", [0.0, -1.0, float("nan"), float("inf"), True, "1.0", None]
)
def test_a_non_positive_or_non_numeric_pixel_scale_is_refused(scale) -> None:
    with pytest.raises(ValueError, match="pixel_scale must be"):
        integrate_height([[(0.0, 0.0, 1.0)]], pixel_scale=scale)


@pytest.mark.parametrize("tolerance", [0.0, -1e-3, float("nan"), True, "loose"])
def test_a_non_positive_or_non_numeric_tolerance_is_refused(tolerance) -> None:
    with pytest.raises(ValueError, match="tolerance must be"):
        integrate_height([[(0.0, 0.0, 1.0)]], tolerance=tolerance)


@pytest.mark.parametrize("iterations", [0, -1, True, 2.5, "many"])
def test_a_bad_sweep_count_is_refused(iterations) -> None:
    with pytest.raises(ValueError, match="iterations must be"):
        integrate_height(
            [[(0.0, 0.0, 1.0)]], method="least-squares", iterations=iterations
        )


def test_shapes_and_components_are_validated() -> None:
    with pytest.raises(ValueError, match="mask must share the normal shape"):
        integrate_height([[(0.0, 0.0, 1.0)]], mask=[[1.0, 1.0]])
    with pytest.raises(ValueError, match="equal width"):
        integrate_height([[(0.0, 0.0, 1.0)], [(0.0, 0.0, 1.0), (0.0, 0.0, 1.0)]])
    with pytest.raises(ValueError, match="three components"):
        integrate_height([[(0.0, 1.0)]])
    with pytest.raises(ValueError, match="numeric"):
        integrate_height([[("a", 0.0, 1.0)]])
    with pytest.raises(ValueError, match="numeric"):
        integrate_height([[(True, 0.0, 1.0)]])
    with pytest.raises(ValueError, match="finite"):
        integrate_height([[(0.0, 0.0, float("inf"))]])
    with pytest.raises(ValueError, match="normals are empty"):
        integrate_height([])
    with pytest.raises(ValueError, match="normals are empty"):
        integrate_height([[]])


def test_analytic_height_refuses_an_unknown_geometry_or_a_tiny_grid() -> None:
    with pytest.raises(ValueError, match="unknown geometry"):
        analytic_height("paraboloid", size=SIZE)
    with pytest.raises(ValueError, match="at least eight"):
        analytic_height("sphere", size=7)


def test_height_specialist_returns_a_relative_surface() -> None:
    normals, mask, _, _, scale = _case("sphere")
    result = recover_height(
        {"normals": normals, "mask": mask, "pixel_scale": scale}
    )
    assert result.specialist_id == HEIGHT_SPECIALIST_ID
    assert result.abstained is False
    assert result.authority_mode is AuthorityMode.CONSULTATIVE
    output = result.output
    assert output["status"] == "ready"
    assert output["method"] == DEFAULT_METHOD
    assert output["units"] == "declared-world-units"
    assert output["pixels"] == 648
    assert output["valid_ratio"] == pytest.approx(0.6328, abs=1e-4)
    assert output["relief"] > 0.0
    assert output["warnings"] == []
    assert output["certified"] is False
    assert output["pixels_changed"] == 0
    assert "constant" in output["note"]


def test_height_specialist_defaults_to_the_measured_method_in_pixels() -> None:
    normals, mask, _, _, _ = _case("cone")
    result = recover_height({"normals": normals, "mask": mask})
    assert result.output["method"] == DEFAULT_METHOD
    assert result.output["units"] == "pixels"
    assert result.output["pixel_scale"] == 1.0


def test_height_specialist_warns_when_the_solver_has_not_finished() -> None:
    normals, mask, _, _, _ = _case("sphere")
    result = recover_height(
        {"normals": normals, "mask": mask, "method": "least-squares", "iterations": 5}
    )
    assert result.abstained is False
    assert "least_squares_not_converged" in result.output["warnings"]
    assert result.output["convergence"]["converged"] is False
    assert result.confidence == 0.4


def test_height_specialist_reports_a_flat_field_without_inventing_relief() -> None:
    normals = [[(0.0, 0.0, 1.0)] * 4 for _ in range(4)]
    result = recover_height({"normals": normals})
    assert result.abstained is False
    assert result.output["relief"] == 0.0
    assert result.output["warnings"] == ["flat_field"]
    assert result.output["height"] == [[0.0] * 4 for _ in range(4)]
    assert result.confidence == 0.5


def test_height_specialist_abstains_without_a_recoverable_slope() -> None:
    normals = [[(1.0, 0.0, 0.0)] * 4 for _ in range(4)]
    result = recover_height({"normals": normals})
    assert result.abstained is True
    assert result.confidence == 0.0
    assert result.output["reason"] == "no_recoverable_slope"
    assert result.output["pixels"] == 0
    assert result.output["warnings"] == ["no_recoverable_slope"]
    assert result.output["certified"] is False


def test_height_specialist_refuses_undocumented_keys_and_missing_normals() -> None:
    with pytest.raises(ValueError, match="unknown query keys"):
        recover_height({"normals": [[(0.0, 0.0, 1.0)]], "certify": True})
    with pytest.raises(ValueError, match="normals must be provided"):
        recover_height({})
    with pytest.raises(ValueError, match="query must be an object"):
        recover_height(None)


_CHAIN: dict = {}


def _chained_normals():
    """The published chain: controlled captures, then the normals specialist."""
    if not _CHAIN:
        normals, mask = synthetic_geometry("sphere", size=SIZE)
        albedo = [
            [sum(pixel) / 3.0 for pixel in row]
            for row in procedural_albedo("checker", size=SIZE, seed=4)
        ]
        captures = multilight_capture(albedo, normals, LIGHTS, ambient=0.05)
        images = [
            to_image([[(value, value, value) for value in row] for row in capture])
            for capture in captures
        ]
        fitted = recover_normals({"captures": images, "lights": LIGHTS})
        assert fitted.abstained is False
        _CHAIN["normals"] = fitted.output["normals"]
        _CHAIN["mask"] = mask
    return _CHAIN["normals"], _CHAIN["mask"]


def test_the_capture_chain_needs_the_declared_grazing_floor() -> None:
    """Measured: near-plane normals dominate an unguarded integral.

    1.37 of relative error over the whole silhouette when every offered pixel is
    integrated, against 0.19 once the 26 pixels below a vertical component of 0.1
    are refused. The deterministic normals of the same geometry reach 0.059, so
    the difference is the capture, not the integrator.
    """
    normals, mask = _chained_normals()
    truth, truth_mask = analytic_height("sphere", size=SIZE)
    scale = 1.0 / (SIZE * 0.45)
    unguarded = recover_height(
        {
            "normals": normals,
            "mask": mask,
            "pixel_scale": scale,
            "min_cosine": 0.0,
        }
    )
    assert unguarded.output["grazing_floor"] == 0.0
    assert unguarded.output["grazing_dropped"] == 0
    plain = height_error(unguarded.output["height"], truth, truth_mask)
    assert plain["relative_rmse"] > 1.0

    guarded = recover_height(
        {"normals": normals, "mask": mask, "pixel_scale": scale}
    )
    assert guarded.abstained is False
    output = guarded.output
    assert output["grazing_floor"] == 0.1
    assert output["offered"] == 648
    assert output["pixels"] == 622
    assert output["grazing_dropped"] == 26
    assert "grazing_pixels_dropped" in output["warnings"]
    kept = height_error(output["height"], truth, truth_mask)
    assert kept["relative_rmse"] == pytest.approx(0.194386, abs=1e-4)
    assert kept["relative_rmse"] < plain["relative_rmse"] / 4
    assert math.isfinite(kept["rmse"])


def test_the_grazing_floor_changes_nothing_on_a_deterministic_field() -> None:
    """The same geometry from analytic normals: no pixel sits below the floor."""
    normals, mask, truth, truth_mask, scale = _case("sphere")
    plain = integrate_height(
        normals, mask, method="cumulative-average", pixel_scale=scale
    )
    guarded = integrate_height(
        normals, mask, method="cumulative-average", pixel_scale=scale, min_cosine=0.1
    )
    assert guarded["pixels"] == plain["pixels"] == 648
    assert guarded["min_cosine"] == 0.1
    assert height_error(guarded["height"], truth, truth_mask)["relative_rmse"] == (
        height_error(plain["height"], truth, truth_mask)["relative_rmse"]
    )


def test_the_grazing_floor_is_validated_and_narrows_the_field() -> None:
    normals = [
        [(0.6, 0.0, 0.8), (0.6, 0.0, 0.2)],
        [(0.6, 0.0, 1.0), (0.0, 0.0, 0.05)],
    ]
    _, _, valid = gradients(normals, None)
    assert valid == [[1.0, 1.0], [1.0, 1.0]]
    _, _, floored = gradients(normals, None, min_cosine=0.5)
    assert floored == [[1.0, 0.0], [1.0, 0.0]]
    for bad in (-0.1, 1.0, 1.5, float("nan"), True, "high"):
        with pytest.raises(ValueError, match="min_cosine must be"):
            integrate_height(normals, min_cosine=bad)
