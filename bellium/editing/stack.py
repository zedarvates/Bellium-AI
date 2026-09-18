"""Ordered, replayable edit stack with undo, redo and before/after comparison.

The stack keeps its own copy of the source, never writes to it, and rebuilds the
result from the active steps, so an edit is reproducible from its recipe alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bellium.editing.filters import apply_filter
from bellium.editing.selection import apply_mask
from bellium.knn._image import Image, copy_image, shape


@dataclass(frozen=True)
class EditStep:
    """One declared operation: a filter, its parameters and an optional mask."""

    name: str
    parameters: dict[str, Any] = field(default_factory=dict)
    mask: list[list[float]] | None = None
    label: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("a step needs a filter name")
        if self.mask is not None and not isinstance(self.mask, list):
            raise ValueError("a step mask must be a list of rows")
        object.__setattr__(self, "parameters", dict(self.parameters))

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parameters": dict(self.parameters),
            "masked": self.mask is not None,
            "label": self.label,
        }


class EditStack:
    """A source image plus an ordered list of steps that can be undone and redone."""

    def __init__(self, source: Image, *, steps: list[EditStep] | None = None) -> None:
        shape(source)
        self._source = copy_image(source)
        self._steps: list[EditStep] = list(steps or [])
        self._redo: list[EditStep] = []
        self._cache: Image | None = None

    @property
    def source(self) -> Image:
        return copy_image(self._source)

    def source_view(self) -> Image:
        """The internal source without copying, for read-only work such as a preview.

        Never mutate the returned list: use source when the caller needs its own copy.
        """
        return self._source

    def add(self, step: EditStep) -> "EditStack":
        if not isinstance(step, EditStep):
            raise ValueError("add expects an EditStep")
        height, width = shape(self._source)
        if step.mask is not None and (
            len(step.mask) != height or any(len(row) != width for row in step.mask)
        ):
            raise ValueError("the step mask must share the source shape")
        self._steps.append(step)
        self._redo.clear()
        self._cache = None
        return self

    def undo(self) -> bool:
        if not self._steps:
            return False
        self._redo.append(self._steps.pop())
        self._cache = None
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._steps.append(self._redo.pop())
        self._cache = None
        return True

    def history(self) -> list[dict[str, Any]]:
        return [step.describe() for step in self._steps]

    def active_steps(self) -> tuple[EditStep, ...]:
        """The steps currently applied, in order, without exposing the list itself."""
        return tuple(self._steps)

    def pending_redo(self) -> int:
        return len(self._redo)

    def apply(self) -> Image:
        if self._cache is None:
            working = copy_image(self._source)
            for step in self._steps:
                filtered = apply_filter(working, step.name, **step.parameters)
                working = (
                    filtered if step.mask is None else apply_mask(working, filtered, step.mask)
                )
            self._cache = working
        return copy_image(self._cache)

    def before_after(self) -> tuple[Image, Image]:
        return copy_image(self._source), self.apply()

    def export(self) -> Image:
        """A copy of the current result; the source and the stack are untouched."""
        return self.apply()

    def to_recipe(self) -> dict[str, Any]:
        return {
            "steps": [
                {
                    "name": step.name,
                    "parameters": dict(step.parameters),
                    "mask": step.mask,
                    "label": step.label,
                }
                for step in self._steps
            ]
        }

    @classmethod
    def from_recipe(cls, source: Image, recipe: object) -> "EditStack":
        if not isinstance(recipe, dict) or not isinstance(recipe.get("steps"), list):
            raise ValueError("a recipe needs a list of steps")
        stack = cls(source)
        for raw in recipe["steps"]:
            if not isinstance(raw, dict):
                raise ValueError("each recipe step must be an object")
            stack.add(EditStep(
                name=str(raw.get("name") or ""),
                parameters=raw.get("parameters") or {},
                mask=raw.get("mask"),
                label=str(raw.get("label") or ""),
            ))
        return stack
