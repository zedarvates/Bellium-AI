"""Minimal deterministic execution for ShapeGrammar 3D."""

from bellium.geometry.executor import execute_plan
from bellium.geometry.fixtures import fixture
from bellium.geometry.mesh import box, cylinder, mirror, sphere, transform


def test_box_has_expected_topology() -> None:
    mesh = box(2.0, 4.0, 6.0)
    assert len(mesh.vertices) == 8
    assert len(mesh.faces) == 6


def test_cylinder_and_sphere_are_deterministic() -> None:
    first = cylinder(1.0, 2.0, segments=8)
    second = cylinder(1.0, 2.0, segments=8)
    assert first == second
    assert len(first.vertices) == 18

    ball = sphere(1.0, rings=4, segments=8)
    assert len(ball.vertices) == 26
    assert len(ball.faces) == 32


def test_transform_moves_vertices_without_changing_faces() -> None:
    mesh = box(2.0, 2.0, 2.0)
    moved = transform(mesh, (1.0, 2.0, 3.0), (0.0, 0.0, 0.0), (2.0, 1.0, 1.0))
    assert moved.faces == mesh.faces
    assert moved.vertices[0] == (-1.0, 1.0, 2.0)


def test_mirror_reflects_across_offset_plane() -> None:
    mesh = box(2.0, 2.0, 2.0)
    mirrored = mirror(mesh, "x", offset=2.0)
    xs = sorted(vertex[0] for vertex in mirrored.vertices)
    assert xs[0] == 3.0
    assert xs[-1] == 5.0


def test_sci_fi_hammer_executes_safe_subset() -> None:
    result = execute_plan(fixture("sci-fi-hammer"))
    assert "mirror-core-detail" in result.executed_operations
    assert "bevel-head" in result.skipped_operations
    assert "core-pair" in result.meshes
    assert "head-beveled" in result.meshes
    assert len(result.combined.vertices) > 0
    assert len(result.combined.faces) > 0
    assert ("bevel-pending", "bevel-head") in result.meshes["head-beveled"].metadata


def test_hammer_execution_is_deterministic() -> None:
    first = execute_plan(fixture("sci-fi-hammer"))
    second = execute_plan(fixture("sci-fi-hammer"))
    assert first == second
