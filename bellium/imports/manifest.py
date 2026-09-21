"""Import manifests as data: page names, declared shapes and their checks.

A manifest is a JSON mirror of what an importer would need, never a project file.
Godot resources, Unity metadata and TexturePacker JSON have their own on-disk
formats; nothing here writes them. The bottom-left conversion of the Unity sheet
rect is computed and re-checked, because that origin trips importers in practice.
"""

from __future__ import annotations

from typing import Any

from bellium.atlas.packing import AtlasPlan, Frame
from bellium.imports.targets import TargetProfile

ORIGINS = {
    "godot-atlas-textures": "top-left",
    "unity-spritesheet": "bottom-left",
    "phaser-json": "top-left",
    "generic-regions": "top-left",
}
DEFAULT_PATTERN = "atlas_{index:02d}.png"
APP = "bellium-atlas-packing"
APP_VERSION = "0.1.0"


def page_names(count: int, pattern: str = DEFAULT_PATTERN) -> tuple[str, ...]:
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError("page count must be a non-negative integer")
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("page pattern must be a non-empty string")
    if count > 1 and "{index" not in pattern:
        raise ValueError("a multi-page pattern needs an {index} placeholder")
    try:
        names = tuple(pattern.format(index=index) for index in range(count))
    except (KeyError, IndexError, ValueError) as error:
        raise ValueError(f"page pattern cannot be rendered: {error}") from None
    if len(set(names)) != len(names):
        raise ValueError("page pattern produces duplicate names")
    return names


def _region(plan: AtlasPlan, placement, origin: str) -> dict[str, int]:
    y = placement.y
    if origin == "bottom-left":
        y = plan.page_height - (placement.y + placement.height)
    return {"x": placement.x, "y": y, "w": placement.width, "h": placement.height}


def build_manifest(
    frames: tuple[Frame, ...],
    plan: AtlasPlan,
    profile: TargetProfile,
    *,
    pattern: str = DEFAULT_PATTERN,
) -> dict[str, Any]:
    names = page_names(plan.page_count, pattern)
    origin = ORIGINS[profile.manifest_shape]
    placements = [item for page in plan.pages for item in page]
    common: dict[str, Any] = {
        "shape": profile.manifest_shape,
        "profile": profile.id,
        "profile_source": profile.source,
        "color_space": profile.color_space,
        "origin": origin,
        "engine_verified": False,
    }
    if profile.manifest_shape == "generic-regions":
        return {
            **common,
            "padding": plan.padding,
            "pages": [
                {
                    "index": index,
                    "file": names[index],
                    "width": plan.page_width,
                    "height": plan.page_height,
                    "regions": [
                        {
                            "name": item.name,
                            "page": index,
                            "source_size": {"w": item.width, "h": item.height},
                            **_region(plan, item, origin),
                        }
                        for item in page
                    ],
                }
                for index, page in enumerate(plan.pages)
            ],
        }
    if profile.manifest_shape == "godot-atlas-textures":
        return {
            **common,
            "resources": [
                {
                    "name": item.name,
                    "page": item.page,
                    "atlas": names[item.page],
                    "region": _region(plan, item, origin),
                    "margin": {"left": 0, "top": 0, "right": 0, "bottom": 0},
                    "filter_clip": False,
                }
                for item in placements
            ],
        }
    if profile.manifest_shape == "unity-spritesheet":
        return {
            **common,
            "textures": [
                {
                    "file": names[index],
                    "width": plan.page_width,
                    "height": plan.page_height,
                    "sprites": [
                        {
                            "name": item.name,
                            "page": index,
                            "rect": _region(plan, item, origin),
                            "rect_origin": "bottom-left",
                            "alignment": "Custom",
                            "pivot": {"x": 0.5, "y": 0.5},
                            "pixels_per_unit": 100,
                        }
                        for item in page
                    ],
                }
                for index, page in enumerate(plan.pages)
            ],
        }
    frames_out = {
        item.name: {
            "frame": _region(plan, item, origin),
            "page": item.page,
            "source": names[item.page],
            "rotated": False,
            "trimmed": False,
            "spriteSourceSize": {"x": 0, "y": 0, "w": item.width, "h": item.height},
            "sourceSize": {"w": item.width, "h": item.height},
        }
        for item in placements
    }
    meta: dict[str, Any] = {
        "app": APP,
        "version": APP_VERSION,
        "image": names[0] if names else "",
        "format": "RGBA8888",
        "size": {"w": plan.page_width, "h": plan.page_height},
        "scale": 1,
        "origin": origin,
        "padding": plan.padding,
        "engine_verified": False,
    }
    if plan.page_count > 1:
        meta["related_multi_packs"] = list(names[1:])
    return {**common, "frames": frames_out, "meta": meta}


def _entries(manifest: dict[str, Any]) -> list[tuple[Any, Any, dict]]:
    shape = manifest.get("shape")
    entries: list[tuple[Any, Any, dict]] = []
    if shape == "generic-regions":
        for page in manifest.get("pages") or []:
            for region in page.get("regions") or []:
                entries.append((region.get("name"), page.get("index"), region))
        return entries
    if shape == "godot-atlas-textures":
        for resource in manifest.get("resources") or []:
            entries.append((resource.get("name"), resource.get("page"), resource.get("region") or {}))
        return entries
    if shape == "unity-spritesheet":
        for index, texture in enumerate(manifest.get("textures") or []):
            for sprite in texture.get("sprites") or []:
                entries.append((sprite.get("name"), index, sprite.get("rect") or {}))
        return entries
    for name, entry in (manifest.get("frames") or {}).items():
        entries.append((name, entry.get("page"), entry.get("frame") or {}))
    return entries


def check_manifest(
    manifest: object,
    frames: tuple[Frame, ...],
    plan: AtlasPlan,
    profile: TargetProfile,
) -> list[str]:
    """Recompute the manifest against the plan, including the origin conversion."""
    violations: list[str] = []
    if not isinstance(manifest, dict):
        return ["manifest is not an object"]
    if manifest.get("shape") != profile.manifest_shape:
        violations.append(f"shape is {manifest.get('shape')}, expected {profile.manifest_shape}")
    if manifest.get("profile") != profile.id:
        violations.append(f"profile is {manifest.get('profile')}, expected {profile.id}")
    if manifest.get("color_space") != profile.color_space:
        violations.append("color_space does not match the profile")
    if manifest.get("engine_verified") is not False:
        violations.append("engine_verified must stay false: no engine ran")
    origin = ORIGINS[profile.manifest_shape]
    expected = {
        item.name: (item.page, item.x, item.y, item.width, item.height)
        for page in plan.pages
        for item in page
    }
    seen: dict[str, tuple[int, int, int, int, int]] = {}
    for name, page, box in _entries(manifest):
        if not isinstance(name, str) or not name:
            violations.append("a manifest entry has no name")
            continue
        if name in seen:
            violations.append(f"manifest entry repeated: {name}")
            continue
        if not isinstance(page, int) or isinstance(page, bool):
            violations.append(f"manifest entry without a page: {name}")
            page = -1
        missing = [key for key in ("x", "y", "w", "h") if key not in box]
        if missing:
            violations.append(f"manifest entry {name} is missing {', '.join(missing)}")
            continue
        seen[name] = (page, box["x"], box["y"], box["w"], box["h"])
    for name, (page, x, y, width, height) in expected.items():
        if name not in seen:
            violations.append(f"frame missing from the manifest: {name}")
            continue
        got_page, got_x, got_y, got_w, got_h = seen[name]
        if (got_page, got_x, got_w, got_h) != (page, x, width, height):
            violations.append(f"manifest geometry differs from the plan: {name}")
            continue
        if origin == "bottom-left":
            wanted_y = plan.page_height - (y + height)
        else:
            wanted_y = y
        if got_y != wanted_y:
            violations.append(f"manifest y is not {origin} for {name}")
        left, top, right, bottom = got_x, got_y, got_x + got_w, got_y + got_h
        if left < 0 or top < 0 or right > plan.page_width or bottom > plan.page_height:
            violations.append(f"manifest entry out of bounds: {name}")
    for name in seen:
        if name not in expected:
            violations.append(f"manifest entry has no frame: {name}")
    files: list[str] = []
    if profile.manifest_shape == "generic-regions":
        files = [page.get("file") for page in manifest.get("pages") or []]
    elif profile.manifest_shape == "unity-spritesheet":
        files = [texture.get("file") for texture in manifest.get("textures") or []]
    elif profile.manifest_shape == "phaser-json":
        meta = manifest.get("meta") or {}
        files = [meta.get("image")] + list(meta.get("related_multi_packs") or [])
    else:
        files = list(dict.fromkeys(
            resource.get("atlas") for resource in manifest.get("resources") or []
        ))
    if plan.page_count and any(not isinstance(name, str) or not name for name in files):
        violations.append("a page file name is missing")
    if len(set(files)) != len(files):
        violations.append("page file names are not unique")
    return violations
