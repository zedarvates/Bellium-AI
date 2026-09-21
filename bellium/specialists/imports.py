"""Engine-import validation against a declared contract.

The specialist packs the declared frames, checks the plan against a target
profile and returns the findings plus a manifest mirror. No engine is executed and
no file is written: a profile is a contract the caller declares, so a ready
verdict means the geometry satisfies that contract, not that an engine accepted it.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from bellium.atlas.packing import pack, parse_frames
from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.imports.manifest import DEFAULT_PATTERN, build_manifest, check_manifest
from bellium.imports.targets import get_profile, summarize, target_profile, validate_plan

SPECIALIST_ID = "bellium/hybrid/engine-import-validation:v0"
QUERY_KEYS = (
    "frames",
    "width",
    "height",
    "method",
    "padding",
    "max_pages",
    "profile",
    "profile_overrides",
    "page_pattern",
)
READY_CONFIDENCE = 0.9


def validate_import(query: dict[str, Any]) -> SpecialistResult:
    if not isinstance(query, dict):
        raise ValueError("query must be an object")
    unknown = sorted(set(query) - set(QUERY_KEYS))
    if unknown:
        raise ValueError(f"unknown query keys refuse to be ignored: {', '.join(unknown)}")
    profile = get_profile(query.get("profile"))
    overrides: dict = {}
    if "profile_overrides" in query:
        declared = query["profile_overrides"]
        if not isinstance(declared, dict):
            raise ValueError("profile_overrides must be an object")
        overrides = declared
    if overrides:
        merged = {**asdict(profile), **overrides}
        if merged.get("id") != profile.id:
            raise ValueError("profile_overrides must not change the profile id")
        profile = target_profile(merged)
    frames = parse_frames(query.get("frames"))
    pattern = query.get("page_pattern", DEFAULT_PATTERN)
    if not isinstance(pattern, str):
        raise ValueError("page_pattern must be a string")
    plan = pack(
        frames,
        width=query.get("width"),
        height=query.get("height"),
        method=query.get("method", "skyline"),
        padding=query.get("padding", 0),
        max_pages=query.get("max_pages", 1),
    )
    findings = validate_plan(frames, plan, profile)
    manifest = build_manifest(frames, plan, profile, pattern=pattern)
    violations = check_manifest(manifest, frames, plan, profile)
    shared: dict[str, Any] = {
        "profile": profile.id,
        "profile_source": profile.source,
        "profile_note": profile.note,
        "manifest_shape": profile.manifest_shape,
        "findings": [
            {"severity": item.severity, "code": item.code, "message": item.message, "frame": item.frame}
            for item in findings
        ],
        "summary": summarize(findings),
        "manifest": manifest,
        "manifest_violations": violations,
        "pages": plan.page_count,
        "page_width": plan.page_width,
        "page_height": plan.page_height,
        "padding": plan.padding,
        "complete": plan.complete,
        "imported": False,
        "written_files": False,
        "certified": False,
    }
    if violations:
        return SpecialistResult(
            SPECIALIST_ID,
            {**shared, "status": "abstain", "reason": "manifest_failed_verification"},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=("A manifest that fails its own checks is reported, never returned as usable.",),
        )
    if shared["summary"]["errors"]:
        return SpecialistResult(
            SPECIALIST_ID,
            {**shared, "status": "abstain", "reason": "import_contract_violated"},
            0.0, True, AuthorityMode.CONSULTATIVE,
            notes=(
                "The plan violates the declared profile; nothing was repaired, scaled or cropped.",
                "Fix the declared page, padding or profile and validate again.",
            ),
        )
    return SpecialistResult(
        SPECIALIST_ID,
        {**shared, "status": "ready", "reason": None},
        READY_CONFIDENCE,
        False,
        AuthorityMode.CONSULTATIVE,
        notes=(
            "Checked against a declared contract, not against a running engine.",
            "No file was written; the manifest is returned as data for the caller.",
        ),
    )
