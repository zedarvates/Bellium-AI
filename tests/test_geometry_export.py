"""Deterministic export evidence for the minimal geometry kernel."""

from bellium.geometry.executor import execute_plan
from bellium.geometry.export import bounds, extent, mesh_to_obj
from bellium.geometry.fixtures import fixture
from bellium.geometry.mesh import box


def test_bounds_and_extent() -> None:
    mesh = box(2.0, 4.0, 6.0)
    assert bounds(mesh) == ((-1.0, -2.0, -3.0), (1.0, 2.0, 3.0))
    assert extent(mesh) == (2.0, 4.0, 6.0)


def test_obj_export_is_stable_and_one_based() -> None:
    mesh = box(2.0, 2.0, 2.0)
    first = mesh_to_obj(mesh, "cube")
    second = mesh_to_obj(mesh, "cube")
    assert first == second
    assert first.startswith("o cube\n")
    assert "\nf 1 2 3 4\n" in first


def test_hammer_can_be_exported_as_obj_text() -> None:
    result = execute_plan(fixture("sci-fi-hammer"))
    obj = mesh_to_obj(result.combined, "fixture-sci-fi-hammer-v0")
    assert obj.count("\nv ") == len(result.combined.vertices)
    assert obj.count("\nf ") == len(result.combined.faces)
    assert "fixture-sci-fi-hammer-v0" in obj
