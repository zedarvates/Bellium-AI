"""Benchmark cheap neighbour projection against exact closest-triangle reference."""

from bellium.geometry.closest import closest_point_on_mesh, closest_point_on_triangle
from bellium.geometry.mesh import Mesh
from bellium.geometry.projection_compare import compare_projection


def _plane_quad() -> Mesh:
    return Mesh(
        vertices=(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ),
        faces=((0, 1, 2, 3),),
    )


def test_closest_point_inside_triangle_projects_to_plane() -> None:
    point = closest_point_on_triangle(
        (0.25, 0.25, 2.0),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )
    assert point == (0.25, 0.25, 0.0)


def test_closest_point_outside_triangle_hits_edge_or_vertex() -> None:
    point = closest_point_on_triangle(
        (2.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    )
    assert point == (1.0, 0.0, 0.0)


def test_closest_point_on_quad_uses_triangle_fan() -> None:
    result = closest_point_on_mesh(_plane_quad(), (0.5, 0.5, 3.0))
    assert result.abstained is False
    assert result.point == (0.5, 0.5, 0.0)
    assert result.distance == 3.0


def test_neighbour_projection_matches_exact_at_symmetric_center() -> None:
    result = compare_projection(
        _plane_quad(),
        (0.5, 0.5, 1.0),
        k=4,
        tolerance=1e-12,
    )
    assert result.abstained is False
    assert result.point_error == 0.0
    assert result.recommendation == "neighbour-ok"


def test_neighbour_projection_can_diverge_near_corner() -> None:
    result = compare_projection(
        _plane_quad(),
        (0.05, 0.05, 1.0),
        k=4,
        tolerance=0.01,
    )
    assert result.abstained is False
    assert result.point_error is not None
    assert result.point_error > 0.01
    assert result.recommendation == "use-exact"


def test_empty_mesh_abstains() -> None:
    result = compare_projection(Mesh(vertices=(), faces=()), (0.0, 0.0, 0.0))
    assert result.abstained
    assert result.recommendation == "exact-or-abstain"
