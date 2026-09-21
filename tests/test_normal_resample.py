"""Reductions of a normal field: what the filter changes and what it does not."""

import math

import pytest

from bellium.contracts.schema import AuthorityMode
from bellium.material.controlled import sampled_geometry, sampled_height
from bellium.material.integration import height_error, integrate_height
from bellium.material.normals_io import inspect_normal_map
from bellium.material.photometric import normal_error
from bellium.material.resample import (
    DEFAULT_METHOD,
    FLATTENING_LIMIT,
    METHODS,
    downsample_normals,
    mip_chain,
)
from bellium.specialists.normal_mip_chain import build_normal_mips

BASE = 32


def _stored_lengths(field, mask):
    return [
        math.sqrt(sum(value * value for value in field[r][c]))
        for r in range(len(field))
        for c in range(len(field[0]))
        if field[r][c] is not None and mask[r][c] > 0.0
    ]


def _normalized(field):
    out = []
    for row in field:
        values = []
        for normal in row:
            if normal is None:
                values.append(None)
                continue
            length = math.sqrt(sum(value * value for value in normal))
            values.append(tuple(value / length for value in normal))
        out.append(values)
    return out


def _as_stored_bytes(field):
    return [
        [
            (128, 128, 255)
            if normal is None
            else tuple(
                min(255, max(0, int(round((value + 1.0) * 0.5 * 255.0)))) for value in normal
            )
            for normal in row
        ]
        for row in field
    ]


def test_the_reduction_changes_the_length_and_not_the_direction() -> None:
    normals, mask = sampled_geometry("sphere", BASE)
    naive = downsample_normals(normals, 2, mask=mask, method="naive")
    clean = downsample_normals(normals, 2, mask=mask, method="renormalized")
    truth, truth_mask = sampled_geometry("sphere", BASE // 2)
    both = [
        [1.0 if (naive["mask"][r][c] > 0 and truth_mask[r][c] > 0) else 0.0
         for c in range(len(naive["mask"][0]))]
        for r in range(len(naive["mask"]))
    ]
    naive_raw = normal_error(naive["normals"], truth, both)["mean_deg"]
    clean_raw = normal_error(clean["normals"], truth, both)["mean_deg"]
    naive_direction = normal_error(_normalized(naive["normals"]), truth, both)["mean_deg"]
    clean_direction = normal_error(_normalized(clean["normals"]), truth, both)["mean_deg"]
    assert naive_direction == clean_direction
    assert naive_raw > 20 * clean_raw
    assert clean_raw == clean_direction
    lengths = _stored_lengths(naive["normals"], naive["mask"])
    assert 0.99 < sum(lengths) / len(lengths) < 1.0
    assert all(
        length == pytest.approx(1.0, abs=1e-9)
        for length in _stored_lengths(clean["normals"], clean["mask"])
    )


def test_the_integral_cannot_see_a_scale_on_a_normal() -> None:
    """A length defect is invisible to the integrator, which is why it survives."""
    normals, mask = sampled_geometry("sphere", BASE)
    truth, truth_mask = sampled_height("sphere", BASE // 2)
    errors = {}
    for method in ("naive", "renormalized"):
        reduced = downsample_normals(normals, 2, mask=mask, method=method)
        result = integrate_height(
            reduced["normals"], reduced["mask"], method="cumulative-average",
            pixel_scale=1.0 / ((BASE // 2) * 0.45),
        )
        errors[method] = height_error(result["height"], truth, truth_mask)["rmse"]
    assert errors["naive"] == errors["renormalized"]


def test_the_inspector_cannot_see_the_length_defect_either() -> None:
    normals, mask = sampled_geometry("sphere", BASE)
    naive = downsample_normals(normals, 4, mask=mask, method="naive")
    stored = _as_stored_bytes(naive["normals"])
    report = inspect_normal_map(stored)
    assert report["verdict"] == "tangent-space"
    assert report["mean_non_unit"] < 0.2
    assert report["mean_non_unit"] > 0.005


def test_slope_space_filtering_loses_the_direction_on_a_curved_field() -> None:
    """The textbook advice measures worse here, and the cone is the exception."""
    for geometry, sphere_case in (("sphere", True), ("cone", False)):
        normals, mask = sampled_geometry(geometry, BASE)
        truth, truth_mask = sampled_geometry(geometry, BASE // 2)
        scores = {}
        for method in METHODS:
            reduced = downsample_normals(normals, 2, mask=mask, method=method)
            both = [
                [1.0 if (reduced["mask"][r][c] > 0 and truth_mask[r][c] > 0) else 0.0
                 for c in range(len(reduced["mask"][0]))]
                for r in range(len(reduced["mask"]))
            ]
            scores[method] = normal_error(
                _normalized(reduced["normals"]), truth, both
            )["mean_deg"]
        if sphere_case:
            assert scores["slopes"] > scores["renormalized"] * 2.0
        else:
            assert scores["slopes"] == scores["renormalized"]


def test_the_chain_reports_each_level_and_its_drift() -> None:
    normals, mask = sampled_geometry("waves", BASE)
    chain = mip_chain(normals, mask=mask, levels=2)
    assert chain["method"] == DEFAULT_METHOD
    assert [level["shape"] for level in chain["levels"]] == [[32, 32], [16, 16], [8, 8]]
    assert [level["level"] for level in chain["pixels"]] == [1, 2]
    assert [len(level["normals"]) for level in chain["pixels"]] == [16, 8]
    assert chain["levels"][1]["mean_z"] > chain["levels"][0]["mean_z"]
    assert chain["warnings"] == []
    assert chain["pixels_returned"] == 16 * 16 + 8 * 8
    assert abs(chain["drift"]) < FLATTENING_LIMIT


def test_a_curved_field_flattens_across_a_chain_and_a_plane_does_not() -> None:
    sphere, sphere_mask = sampled_geometry("sphere", BASE)
    flat = mip_chain(sphere, mask=sphere_mask, levels=2)
    assert "flattening_drift" in flat["warnings"]
    assert flat["drift"] > FLATTENING_LIMIT
    # the circular silhouette cannot fill a whole block at its rim
    assert "partial_coverage" in flat["warnings"]
    plane, plane_mask = sampled_geometry("tilted-plane", BASE)
    still = mip_chain(plane, mask=plane_mask, levels=2)
    assert still["drift"] == 0.0
    assert still["warnings"] == []


def test_a_small_source_stops_the_chain_instead_of_inventing_levels() -> None:
    normals, mask = sampled_geometry("waves", 16)
    chain = mip_chain(normals, mask=mask, levels=6)
    assert "chain_stopped_early" in chain["warnings"]
    assert len(chain["levels"]) < 7
    shapes = [level["shape"][0] for level in chain["levels"]]
    assert shapes == sorted(shapes, reverse=True)
    assert min(shapes) >= 1


def test_a_factor_of_one_returns_the_field_unchanged() -> None:
    normals, mask = sampled_geometry("cone", BASE)
    same = downsample_normals(normals, 1, method="naive")
    assert same["factor"] == 1
    assert same["coverage"] == 1.0
    assert same["normals"][5][5] == normals[5][5]
    assert same["normals"] is not normals


def test_the_specialist_returns_the_chain_within_its_budget() -> None:
    normals, mask = sampled_geometry("cone", BASE)
    result = build_normal_mips({"normals": normals, "mask": mask, "levels": 2})
    assert result.authority_mode is AuthorityMode.CONSULTATIVE
    output = result.output
    assert output["status"] == "ready"
    assert output["written_files"] is False
    assert output["certified"] is False
    assert output["pixels_changed"] == 0
    assert len(output["pixels"]) == 2
    assert "flattening_drift" not in output["warnings"]
    assert result.confidence > 0.5


def test_the_specialist_refuses_to_materialize_an_oversized_chain() -> None:
    normals, mask = sampled_geometry("sphere", BASE)
    result = build_normal_mips({"normals": normals, "mask": mask, "max_pixels": 10})
    assert result.abstained is False
    assert result.output["status"] == "statistics_only"
    assert result.output["reason"] == "chain_too_large"
    assert result.output["pixels"] is None
    assert result.output["pixels_returned"] > result.output["max_pixels"]
    assert "chain_too_large" in result.output["warnings"]
    assert result.output["levels"][0]["shape"] == [32, 32]


def test_the_flattening_warning_lowers_the_confidence() -> None:
    normals, mask = sampled_geometry("sphere", BASE)
    result = build_normal_mips({"normals": normals, "mask": mask, "levels": 2})
    assert "flattening_drift" in result.output["warnings"]
    assert result.confidence == 0.5


def test_the_reduction_validates_its_inputs() -> None:
    normals, mask = sampled_geometry("sphere", 16)
    for factor in (0, -1, True, 1.5, "two"):
        with pytest.raises(ValueError, match="factor must be"):
            downsample_normals(normals, factor, mask=mask)
    for method in ("lanczos", "", None, 3):
        with pytest.raises(ValueError, match="method must be one of"):
            downsample_normals(normals, 2, mask=mask, method=method)
    for levels in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="levels must be"):
            mip_chain(normals, mask=mask, levels=levels)
    with pytest.raises(ValueError, match="mask must share the normal shape"):
        downsample_normals(normals, 2, mask=[[1.0]])
    with pytest.raises(ValueError, match="normals are empty"):
        downsample_normals([], 2)
    with pytest.raises(ValueError, match="normal components must be numeric"):
        downsample_normals([[(True, 0.0, 1.0)]], 1)
    with pytest.raises(ValueError, match="normal components must be finite"):
        downsample_normals([[(0.0, 0.0, float("inf"))]], 1)


def test_the_specialist_refuses_undocumented_keys_and_missing_normals() -> None:
    normals, _ = sampled_geometry("sphere", 16)
    with pytest.raises(ValueError, match="unknown query keys"):
        build_normal_mips({"normals": normals, "mipmaps": True})
    with pytest.raises(ValueError, match="normals must be provided"):
        build_normal_mips({})
    with pytest.raises(ValueError, match="query must be an object"):
        build_normal_mips(None)
    with pytest.raises(ValueError, match="max_pixels must be"):
        build_normal_mips({"normals": normals, "max_pixels": 0})

