"""Historical geometry-kernel ideas reused in the new retopology pipeline."""

from bellium.geometry.mesh import Mesh
from bellium.geometry.projection import project_to_neighbour_centroid
from bellium.geometry.qef import QEFConstraint, solve_qef
from bellium.geometry.qem import (
    best_edge_collapse_candidate,
    plane_quadric,
    quadric_error,
)


def test_neighbour_centroid_projection_uses_nearby_points() -> None:
    mesh = Mesh(
        vertices=(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
        ),
        faces=(),
    )
    result = project_to_neighbour_centroid(mesh, (0.5, 0.5, 1.0), k=4)
    assert result.abstained is False
    assert result.point[2] == 0.0
    assert 0.0 <= result.point[0] <= 1.0
    assert 0.0 <= result.point[1] <= 1.0


def test_projection_abstains_outside_radius() -> None:
    mesh = Mesh(vertices=((0.0, 0.0, 0.0),), faces=())
    result = project_to_neighbour_centroid(mesh, (10.0, 0.0, 0.0), max_radius=1.0)
    assert result.abstained
    assert result.reason == "outside_projection_radius"


def test_qef_solves_three_orthogonal_planes() -> None:
    result = solve_qef((
        QEFConstraint((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        QEFConstraint((0.0, 2.0, 0.0), (0.0, 1.0, 0.0)),
        QEFConstraint((0.0, 0.0, 3.0), (0.0, 0.0, 1.0)),
    ))
    assert result.abstained is False
    assert result.point == (1.0, 2.0, 3.0)
    assert result.error == 0.0


def test_qef_abstains_on_parallel_constraints() -> None:
    result = solve_qef((
        QEFConstraint((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        QEFConstraint((2.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        QEFConstraint((3.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    ))
    assert result.abstained
    assert result.reason == "singular_constraints"


def test_qem_plane_has_zero_cost_on_plane() -> None:
    quadric = plane_quadric((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert quadric_error(quadric, (0.0, 2.0, 3.0)) == 0.0
    assert quadric_error(quadric, (2.0, 0.0, 0.0)) > 0.0


def test_qem_protected_edge_refuses_collapse() -> None:
    q = plane_quadric((0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    result = best_edge_collapse_candidate(
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        q,
        q,
        protected=True,
    )
    assert result.protected
    assert result.cost == float("inf")
    assert result.reason == "protected_feature"
