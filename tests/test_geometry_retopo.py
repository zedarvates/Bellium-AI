"""Consultative retopology path proposals."""

from bellium.geometry.executor import execute_plan
from bellium.geometry.fixtures import fixture
from bellium.geometry.mesh import box
from bellium.geometry.retopo import (
    crease_paths,
    flow_paths,
    propose_retopo_paths,
    scored_edges,
)


def test_scored_edges_are_deterministic() -> None:
    mesh = box(6.0, 2.0, 2.0)
    first = scored_edges(mesh)
    second = scored_edges(mesh)
    assert first == second
    assert first[0][1] >= first[-1][1]


def test_box_has_crease_path_proposals() -> None:
    mesh = box(2.0, 2.0, 2.0)
    paths = crease_paths(mesh)
    assert paths
    assert all(path.reason == "crease-chain" for path in paths)
    assert all(0.0 <= path.score <= 1.0 for path in paths)


def test_rectangular_box_has_surface_flow_paths() -> None:
    mesh = box(6.0, 2.0, 2.0)
    paths = flow_paths(mesh, min_alignment=0.95)
    assert paths
    assert all(path.reason == "surface-flow" for path in paths)


def test_invalid_alignment_is_rejected() -> None:
    mesh = box(2.0, 2.0, 2.0)
    try:
        flow_paths(mesh, min_alignment=1.1)
    except ValueError as exc:
        assert "between 0 and 1" in str(exc)
    else:
        raise AssertionError("expected invalid alignment to fail")


def test_hammer_retopo_proposals_are_repeatable() -> None:
    mesh = execute_plan(fixture("sci-fi-hammer")).combined
    first = propose_retopo_paths(mesh)
    second = propose_retopo_paths(mesh)
    assert first == second
    assert first
    assert any(path.reason == "crease-chain" for path in first)


def test_proposals_do_not_modify_source_mesh() -> None:
    mesh = box(2.0, 2.0, 2.0)
    before = mesh
    propose_retopo_paths(mesh)
    assert mesh == before
