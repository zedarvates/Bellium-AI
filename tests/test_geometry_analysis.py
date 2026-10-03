"""Deterministic surface-analysis primitives for the retopology bridge."""

from bellium.geometry.analyze import (
    analyze_surface,
    crease_records,
    edge_adjacency,
    face_normal,
    silhouette_edge_count,
)
from bellium.geometry.executor import execute_plan
from bellium.geometry.fixtures import fixture
from bellium.geometry.mesh import box


def test_box_edge_adjacency_is_closed_manifold() -> None:
    mesh = box(2.0, 2.0, 2.0)
    adjacency = edge_adjacency(mesh)
    assert len(adjacency) == 12
    assert all(len(faces) == 2 for faces in adjacency.values())


def test_box_face_normals_are_unit_length_axes() -> None:
    mesh = box(2.0, 2.0, 2.0)
    normals = {face_normal(mesh, i) for i in range(len(mesh.faces))}
    assert normals == {
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
    }


def test_box_edges_are_detected_as_creases() -> None:
    mesh = box(2.0, 2.0, 2.0)
    creases = crease_records(mesh, threshold_degrees=35.0)
    assert len(creases) == 12
    assert all(record.dihedral_degrees == 90.0 for record in creases)


def test_box_has_silhouette_edges_from_axis_view() -> None:
    mesh = box(2.0, 2.0, 2.0)
    assert silhouette_edge_count(mesh, (1.0, 0.0, 0.0)) == 4
    assert silhouette_edge_count(mesh, (0.0, 0.0, 1.0)) == 4


def test_hammer_surface_analysis_is_deterministic() -> None:
    mesh = execute_plan(fixture("sci-fi-hammer")).combined
    first = analyze_surface(mesh)
    second = analyze_surface(mesh)
    assert first == second
    assert first.vertex_count == len(mesh.vertices)
    assert first.face_count == len(mesh.faces)
    assert first.edge_count > 0
    assert first.boundary_edge_count == 0
    assert first.non_manifold_edge_count == 0
    assert len(first.silhouette_edge_counts) == 6


def test_invalid_crease_threshold_is_rejected() -> None:
    mesh = box(2.0, 2.0, 2.0)
    try:
        crease_records(mesh, threshold_degrees=181.0)
    except ValueError as exc:
        assert "between 0 and 180" in str(exc)
    else:
        raise AssertionError("expected invalid threshold to fail")
