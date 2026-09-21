"""Tests for horizon-based ambient occlusion and specialist contract."""

import pytest

from bellium.contracts.schema import AuthorityMode
from bellium.material.ambient_occlusion import horizon_ambient_occlusion
from bellium.material.controlled import sampled_geometry, sampled_height
from bellium.specialists.ambient_occlusion import estimate_ambient_occlusion

SIZE = 32


def test_v_groove_occlusion_profile() -> None:
    # A symmetric V-groove should be most occluded at center and least at edges
    center = (SIZE - 1) / 2.0
    height = [[abs(c - center) for c in range(SIZE)] for r in range(SIZE)]
    res = horizon_ambient_occlusion(height, radius=6, directions=8)
    
    ao = res["ao"]
    assert res["pixels"] == SIZE * SIZE
    # Center should have lower AO (more occluded) than edges
    mid = int(center)
    assert ao[mid][mid] < ao[mid][0]
    assert ao[mid][mid] < 0.55
    assert ao[mid][0] > 0.70


def test_nearly_flat_plane_has_near_full_accessibility() -> None:
    flat = [[0.0] * SIZE for _ in range(SIZE)]
    res = horizon_ambient_occlusion(flat, radius=4)
    assert res["mean_ao"] == 1.0
    for row in res["ao"]:
        for val in row:
            assert val == 1.0


def test_elevation_scale_increases_occlusion() -> None:
    center = (SIZE - 1) / 2.0
    height = [[abs(c - center) for c in range(SIZE)] for r in range(SIZE)]
    res_low = horizon_ambient_occlusion(height, radius=4, elevation_scale=0.5)
    res_high = horizon_ambient_occlusion(height, radius=4, elevation_scale=2.0)
    assert res_high["mean_ao"] < res_low["mean_ao"]


def test_specialist_runs_from_normals_or_height() -> None:
    normals, mask = sampled_geometry("sphere", size=SIZE)
    h, v = sampled_height("sphere", size=SIZE)

    res_from_norm = estimate_ambient_occlusion({"normals": normals, "mask": mask})
    assert res_from_norm.authority_mode is AuthorityMode.CONSULTATIVE
    assert res_from_norm.output["status"] == "ready"
    assert res_from_norm.output["source_kind"] == "integrated_from_normals"
    assert 0.0 < res_from_norm.output["mean_ao"] < 1.0

    res_from_h = estimate_ambient_occlusion({"height": h, "mask": v})
    assert res_from_h.output["status"] == "ready"
    assert res_from_h.output["source_kind"] == "height"
    assert 0.0 < res_from_h.output["mean_ao"] < 1.0


def test_specialist_abstains_on_empty_field() -> None:
    empty_h = [[0.0] * 4 for _ in range(4)]
    empty_m = [[0.0] * 4 for _ in range(4)]
    res = estimate_ambient_occlusion({"height": empty_h, "mask": empty_m})
    assert res.abstained is True
    assert res.output["status"] == "abstain"
    assert res.output["reason"] == "no_valid_pixels"


def test_invalid_parameters_raise_value_error() -> None:
    valid_h = [[0.0] * 4 for _ in range(4)]
    with pytest.raises(ValueError, match="must be a non-empty grid"):
        horizon_ambient_occlusion([])
    with pytest.raises(ValueError, match="height rows must have equal width"):
        horizon_ambient_occlusion([[1.0], [1.0, 2.0]])
    with pytest.raises(ValueError, match="radius must be an integer"):
        horizon_ambient_occlusion(valid_h, radius=0)
    with pytest.raises(ValueError, match="directions must be an integer"):
        horizon_ambient_occlusion(valid_h, directions=2)
    with pytest.raises(ValueError, match="pixel_scale must be positive"):
        horizon_ambient_occlusion(valid_h, pixel_scale=-1.0)
    with pytest.raises(ValueError, match="either 'height' or 'normals'"):
        estimate_ambient_occlusion({})
    with pytest.raises(ValueError, match="unknown query keys"):
        estimate_ambient_occlusion({"height": valid_h, "bogus": 123})
