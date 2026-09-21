"""Declared import targets and the findings a plan produces against them.

A profile here is a contract the caller declares, not a measurement of an engine:
every shipped profile carries its own source string and conservative defaults.
Validation compares geometry against that contract. No engine is executed and no
file is written.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bellium.atlas.packing import AtlasPlan, Frame, check_plan
from bellium.resources import model_path

SCHEMA = "bellium.import-target-memory/v1"
MANIFEST_SHAPES = ("godot-atlas-textures", "unity-spritesheet", "phaser-json", "generic-regions")
COLOR_SPACES = ("sRGB", "linear")
SEVERITIES = ("error", "warning")
PROFILE_KEYS = (
    "id",
    "source",
    "note",
    "manifest_shape",
    "color_space",
    "max_texture",
    "power_of_two",
    "square",
    "required_padding",
    "recommended_padding",
    "max_pages_per_asset",
)
DEFAULT_PROFILES = model_path("imports", "target-profiles-v0.json")


@dataclass(frozen=True)
class TargetProfile:
    id: str
    source: str
    note: str
    manifest_shape: str
    color_space: str
    max_texture: int
    power_of_two: bool
    square: bool
    required_padding: int
    recommended_padding: int
    max_pages_per_asset: int


@dataclass(frozen=True)
class Finding:
    severity: str
    code: str
    message: str
    frame: str | None = None


def _text(label: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _positive_int(label: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _non_negative_int(label: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _boolean(label: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def target_profile(raw: object) -> TargetProfile:
    """Validate one declared profile. Unknown keys and missing fields raise."""
    if not isinstance(raw, dict):
        raise ValueError("a target profile must be an object")
    extra = sorted(set(raw) - set(PROFILE_KEYS))
    if extra:
        raise ValueError(f"profile has unknown keys: {', '.join(extra)}")
    missing = [key for key in PROFILE_KEYS if key not in raw]
    if missing:
        raise ValueError(f"profile is missing {', '.join(missing)}")
    shape = _text("manifest_shape", raw["manifest_shape"])
    if shape not in MANIFEST_SHAPES:
        raise ValueError(f"manifest_shape must be one of: {', '.join(MANIFEST_SHAPES)}")
    color_space = _text("color_space", raw["color_space"])
    if color_space not in COLOR_SPACES:
        raise ValueError(f"color_space must be one of: {', '.join(COLOR_SPACES)}")
    required = _non_negative_int("required_padding", raw["required_padding"])
    recommended = _non_negative_int("recommended_padding", raw["recommended_padding"])
    if recommended < required:
        raise ValueError("recommended_padding must not be below required_padding")
    return TargetProfile(
        id=_text("id", raw["id"]),
        source=_text("source", raw["source"]),
        note=_text("note", raw["note"]),
        manifest_shape=shape,
        color_space=color_space,
        max_texture=_positive_int("max_texture", raw["max_texture"]),
        power_of_two=_boolean("power_of_two", raw["power_of_two"]),
        square=_boolean("square", raw["square"]),
        required_padding=required,
        recommended_padding=recommended,
        max_pages_per_asset=_positive_int("max_pages_per_asset", raw["max_pages_per_asset"]),
    )


def load_profiles(path: str | Path | None = None) -> list[TargetProfile]:
    payload = json.loads(Path(path or DEFAULT_PROFILES).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError("import target memory schema mismatch")
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("import target memory needs items")
    profiles = [target_profile(item) for item in items]
    seen = set()
    for profile in profiles:
        if profile.id in seen:
            raise ValueError(f"duplicate target profile id: {profile.id}")
        seen.add(profile.id)
    return profiles


def get_profile(profile_id: object, *, profiles: list[TargetProfile] | None = None) -> TargetProfile:
    wanted = _text("profile id", profile_id)
    for profile in profiles if profiles is not None else load_profiles():
        if profile.id == wanted:
            return profile
    raise KeyError(wanted)


def is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def validate_plan(
    frames: tuple[Frame, ...],
    plan: AtlasPlan,
    profile: TargetProfile,
) -> list[Finding]:
    """Compare a plan with a declared contract. Errors block, warnings inform."""
    findings: list[Finding] = []
    violations = check_plan(frames, plan)
    for violation in violations:
        findings.append(Finding("error", "plan_invalid", violation))
    for unplaced in plan.unplaced:
        findings.append(Finding(
            "error", "unplaced_frame",
            f"frame {unplaced.name} is not placed on any page ({unplaced.reason})",
            unplaced.name,
        ))
    if plan.page_width > profile.max_texture or plan.page_height > profile.max_texture:
        findings.append(Finding(
            "error", "page_too_large",
            f"page {plan.page_width} x {plan.page_height} exceeds the declared "
            f"{profile.id} limit of {profile.max_texture}",
        ))
    if profile.power_of_two and not (
        is_power_of_two(plan.page_width) and is_power_of_two(plan.page_height)
    ):
        findings.append(Finding(
            "error", "not_power_of_two",
            f"page {plan.page_width} x {plan.page_height} is not a power of two",
        ))
    if profile.square and plan.page_width != plan.page_height:
        findings.append(Finding(
            "error", "not_square",
            f"page {plan.page_width} x {plan.page_height} is not square",
        ))
    if plan.page_count > profile.max_pages_per_asset:
        findings.append(Finding(
            "error", "too_many_pages",
            f"{plan.page_count} pages exceed the declared {profile.id} limit of "
            f"{profile.max_pages_per_asset}",
        ))
    if plan.padding < profile.required_padding:
        findings.append(Finding(
            "error", "padding_below_required",
            f"padding {plan.padding} is below the required {profile.required_padding}",
        ))
    elif plan.padding < profile.recommended_padding:
        findings.append(Finding(
            "warning", "padding_below_recommended",
            f"padding {plan.padding} is below the recommended {profile.recommended_padding}",
        ))
    return findings


def summarize(findings: list[Finding]) -> dict[str, Any]:
    return {
        "errors": sum(1 for item in findings if item.severity == "error"),
        "warnings": sum(1 for item in findings if item.severity == "warning"),
        "by_code": {
            code: sum(1 for item in findings if item.code == code)
            for code in sorted({item.code for item in findings})
        },
    }

