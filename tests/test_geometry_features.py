"""Feature-scale, flow and anisotropic-density hints for ApproxSurface."""

from bellium.geometry.features import (
    asset_scale,
    density_hint,
    face_flow_hint,
    feature_scales,
)
from bellium.geometry.mesh import box


def test_asset_scale_uses_bounding_diagonal() -> None:
    mesh = box(2.0, 4.0, 4.0)
    assert asset_scale(mesh) == 6.0


def test_box_edges_are_macro_at_fixture_scale() -> None:
    mesh = box(2.0, 2.0, 2.0)
    scales = feature_scales(mesh)
    assert len(scales) == 12
    assert {item.category for item in scales} == {"macro"}


def test_rectangular_face_has_principal_long_direction() -> None:
    mesh = box(6.0, 2.0, 2.0)
    hint = face_flow_hint(mesh, 0)
    assert hint.anisotropy == 3.0
    assert abs(hint.principal_direction[0]) == 1.0


def test_anisotropic_density_uses_more_cross_flow_density() -> None:
    mesh = box(6.0, 2.0, 2.0)
    hint = density_hint(mesh, 0, base_density=1.0)
    assert hint.density_v > hint.density_u
    assert "anisotropic-flow" in hint.reason


def test_importance_increases_density_budget() -> None:
    mesh = box(6.0, 2.0, 2.0)
    baseline = density_hint(mesh, 0)
    important = density_hint(
        mesh,
        0,
        deformation_importance=1.0,
        silhouette_importance=1.0,
    )
    assert important.density_u > baseline.density_u
    assert important.density_v > baseline.density_v
    assert "deformation" in important.reason
    assert "silhouette" in important.reason


def test_invalid_importance_fails_closed() -> None:
    mesh = box(2.0, 2.0, 2.0)
    try:
        density_hint(mesh, 0, deformation_importance=1.2)
    except ValueError as exc:
        assert "between 0 and 1" in str(exc)
    else:
        raise AssertionError("expected invalid importance to fail")
