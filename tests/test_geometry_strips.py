"""Consultative quad-strip construction from compatible guide paths."""

from bellium.geometry.mesh import Mesh
from bellium.geometry.retopo import RetopoPath
from bellium.geometry.strips import path_vertices, propose_quad_strip


def _grid_mesh() -> Mesh:
    return Mesh(
        vertices=(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (2.0, 1.0, 0.0),
        ),
        faces=(),
    )


def _path(edges, *, ambiguous=False, closed=False) -> RetopoPath:
    return RetopoPath(
        edges=tuple(edges),
        score=1.0,
        reason="test",
        closed=closed,
        ambiguous=ambiguous,
    )


def test_path_vertices_reconstructs_chain() -> None:
    path = _path(((0, 1), (1, 2)))
    assert path_vertices(path) == (0, 1, 2)


def test_parallel_paths_propose_two_quads() -> None:
    mesh = _grid_mesh()
    left = _path(((0, 1), (1, 2)))
    right = _path(((3, 4), (4, 5)))
    result = propose_quad_strip(mesh, left, right)
    assert result.rejected is False
    assert result.quads == ((0, 1, 4, 3), (1, 2, 5, 4))
    assert 0.0 < result.score <= 1.0


def test_ambiguous_path_abstains() -> None:
    mesh = _grid_mesh()
    left = _path(((0, 1), (1, 2)), ambiguous=True)
    right = _path(((3, 4), (4, 5)))
    result = propose_quad_strip(mesh, left, right)
    assert result.rejected
    assert result.rejection_reason == "ambiguous_path"


def test_vertex_count_mismatch_abstains() -> None:
    mesh = _grid_mesh()
    left = _path(((0, 1), (1, 2)))
    right = _path(((3, 4),))
    result = propose_quad_strip(mesh, left, right)
    assert result.rejected
    assert result.rejection_reason == "vertex_count_mismatch"


def test_paths_sharing_vertices_abstain() -> None:
    mesh = _grid_mesh()
    left = _path(((0, 1), (1, 2)))
    right = _path(((1, 4), (4, 5)))
    result = propose_quad_strip(mesh, left, right, min_alignment=0.0)
    assert result.rejected
    assert result.rejection_reason == "paths_share_vertices"


def test_excessive_width_abstains() -> None:
    mesh = Mesh(
        vertices=(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (0.0, 10.0, 0.0),
            (1.0, 10.0, 0.0),
            (2.0, 10.0, 0.0),
        ),
        faces=(),
    )
    left = _path(((0, 1), (1, 2)))
    right = _path(((3, 4), (4, 5)))
    result = propose_quad_strip(mesh, left, right, max_width_ratio=4.0)
    assert result.rejected
    assert result.rejection_reason == "strip_too_wide"


def test_direction_mismatch_abstains() -> None:
    mesh = Mesh(
        vertices=(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 2.0, 0.0),
            (0.0, 3.0, 0.0),
        ),
        faces=(),
    )
    left = _path(((0, 1), (1, 2)))
    right = _path(((3, 4), (4, 5)))
    result = propose_quad_strip(mesh, left, right, min_alignment=0.75)
    assert result.rejected
    assert result.rejection_reason == "direction_mismatch"
