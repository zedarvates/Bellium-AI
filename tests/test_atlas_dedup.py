"""Atlas dedup: one stored copy per distinct image, aliases for the animation."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json

import pytest

from bellium.atlas.dedup import (
    analyze_frames,
    apply_transform,
    check_aliases,
    content_key,
    unique_frames,
)
from bellium.atlas.packing import parse_frames
from bellium.atlas.png import decode_png, inspect_png
from bellium.atlas.raster import check_sources, source_shape
from bellium.specialists import atlas_dedup as specialist
from bellium.specialists.atlas_dedup import dedup_atlas

SPEC = [("walk_0", 8, 8), ("walk_1", 8, 8), ("walk_2", 8, 8), ("walk_3", 8, 8)]


def _region_image(page, region, channels):
    return [
        [tuple(page["pixels"][region["y"] + row][region["x"] + column])
         for column in range(region["width"])]
        for row in range(region["height"])
    ]


def test_flipped_frames_are_stored_as_a_mirror() -> None:
    frames = _frames()
    base = _sprite(8, 8, 6, 4)
    sources = {
        "walk_0": base,
        "walk_1": apply_transform(base, "flip-x"),
        "walk_2": apply_transform(base, "flip-y"),
        "walk_3": apply_transform(base, "rotate-180"),
    }
    report = analyze_frames(frames, sources)
    assert report.unique == ("walk_0",)
    assert [alias.transform for alias in report.aliases] == \
        ["none", "flip-x", "flip-y", "rotate-180"]
    assert [alias.delta for alias in report.aliases] == [0, 0, 0, 0]
    assert report.worst_delta == 0
    assert report.saved_bytes == 3 * 8 * 8 * 4
    assert report.groups[0].transforms == ("none", "flip-x", "flip-y", "rotate-180")
    assert report.to_dict()["transforms"] == {"none": 1, "flip-x": 1, "flip-y": 1, "rotate-180": 1}
    assert check_aliases(frames, sources, report) == []


def test_a_symmetric_frame_keeps_its_own_copy() -> None:
    frames = _frames([("a", 4, 4), ("b", 4, 4)])
    symmetric = [[(1, 2, 3, 255)] * 4 for _ in range(4)]
    report = analyze_frames(frames, {"a": symmetric, "b": _sprite(4, 4, 2, 4)})
    assert report.unique == ("a", "b")
    assert all(alias.transform == "none" for alias in report.aliases)


def test_transform_moves_pixels_and_keeps_the_size() -> None:
    image = [[(1, 2, 3, 4), (5, 6, 7, 8)], [(9, 10, 11, 12), (13, 14, 15, 16)]]
    flipped_x = apply_transform(image, "flip-x")
    flipped_y = apply_transform(image, "flip-y")
    turned = apply_transform(image, "rotate-180")
    assert source_shape(flipped_x) == source_shape(image) == (2, 2, 4)
    assert flipped_x[0] == [(5, 6, 7, 8), (1, 2, 3, 4)]
    assert flipped_y[0] == [(9, 10, 11, 12), (13, 14, 15, 16)]
    assert turned[0] == [(13, 14, 15, 16), (9, 10, 11, 12)]
    assert apply_transform(image, "none") == [[tuple(p) for p in row] for row in image]
    for bad in ("rotate-90", "rotate-270", "", None, 3):
        with pytest.raises(ValueError):
            apply_transform(image, bad)


def test_mirror_matching_is_deterministic_under_reordering() -> None:
    frames = _frames()
    base = _sprite(8, 8, 9, 4)
    sources = {"walk_0": base, "walk_1": apply_transform(base, "flip-x"),
               "walk_2": _sprite(8, 8, 3, 4), "walk_3": _sprite(8, 8, 3, 4)}
    forward = analyze_frames(frames, sources)
    backward = analyze_frames(tuple(reversed(frames)),
                              {frame.name: sources[frame.name] for frame in reversed(frames)})
    assert len(forward.unique) == len(backward.unique) == 2
    assert check_aliases(frames, sources, forward) == []
    reversed_frames = tuple(reversed(frames))
    assert check_aliases(reversed_frames, sources, backward) == []


def test_check_aliases_catches_transform_tampering() -> None:
    frames = _frames()
    base = _sprite(8, 8, 6, 4)
    sources = {"walk_0": base, "walk_1": apply_transform(base, "flip-x"),
               "walk_2": base, "walk_3": _sprite(8, 8, 4, 4)}
    report = analyze_frames(frames, sources)
    assert report.canonical_of("walk_1") == "walk_0"
    unknown = replace(report, aliases=tuple(
        replace(alias, transform="rotate-90") if alias.name == "walk_1" else alias
        for alias in report.aliases
    ))
    assert any("unknown transform" in item for item in check_aliases(frames, sources, unknown))
    wrong_way = replace(report, aliases=tuple(
        replace(alias, transform="flip-y") if alias.name == "walk_1" else alias
        for alias in report.aliases
    ))
    assert any("differs from walk_0" in item
               for item in check_aliases(frames, sources, wrong_way))
    wrong_group = replace(report, groups=tuple(
        replace(group, transforms=("none", "none")) for group in report.groups
    ))
    assert any("wrong transforms" in item
               for item in check_aliases(frames, sources, wrong_group))
    loose = analyze_frames(frames, sources, max_delta=255)
    assert loose.unique == ("walk_0",)
    shifted = replace(loose, aliases=tuple(
        replace(alias, transform="flip-x") if alias.name == "walk_2" else alias
        for alias in loose.aliases
    ))
    assert any("skipped the earlier stored frame walk_0 with transform none" in item
               for item in check_aliases(frames, sources, shifted))


def test_specialist_redraws_mirrored_frames_exactly() -> None:
    base = _sprite(8, 8, 7, 4)
    sources = {
        "walk_0": base,
        "walk_1": apply_transform(base, "flip-x"),
        "walk_2": _sprite(8, 8, 5, 4),
        "walk_3": apply_transform(_sprite(8, 8, 5, 4), "flip-y"),
    }
    result = dedup_atlas(_query(sources))
    output = result.output
    assert output["status"] == "ready"
    assert output["frames_stored"] == 2
    assert output["transforms"] == {"none": 2, "flip-x": 1, "flip-y": 1}
    assert output["saved_bytes"] == 2 * 8 * 8 * 4
    regions = {item["name"]: item for item in output["placements"]}
    aliases = {item["name"]: item for item in output["aliases"]}
    for name, source in sources.items():
        alias = aliases[name]
        region = regions[alias["canonical"]]
        drawn = apply_transform(_region_image(output["pages"][region["page"]], region, 4),
                                alias["transform"])
        assert drawn == [[tuple(pixel) for pixel in row] for row in source], name
    assert json.dumps(output)

def _sprite(width, height, seed, channels=4):
    rows = []
    for row in range(height):
        line = []
        for column in range(width):
            base = (row * 11 + column * 7 + seed * 29) % 180
            pixel = (base, 255 - base, (seed * 40 + row * 3) % 256)
            line.append(pixel + (255,) if channels == 4 else pixel)
        rows.append(line)
    return rows


def _frames(spec=SPEC):
    return parse_frames([{"name": name, "width": w, "height": h} for name, w, h in spec])


def _sources(spec=SPEC, seeds=None, channels=4):
    seeds = seeds if seeds is not None else list(range(len(spec)))
    return {
        name: _sprite(w, h, seeds[index], channels)
        for index, (name, w, h) in enumerate(spec)
    }


def test_identical_frames_share_one_stored_copy() -> None:
    frames = _frames()
    sources = _sources(seeds=[1, 2, 1, 3])
    report = analyze_frames(frames, sources)
    assert report.unique == ("walk_0", "walk_1", "walk_3")
    assert report.canonical_of("walk_2") == "walk_0"
    assert report.canonical_of("walk_0") == "walk_0"
    assert [group.canonical for group in report.groups] == ["walk_0"]
    assert report.groups[0].members == ("walk_0", "walk_2")
    assert report.groups[0].saved_bytes == 8 * 8 * 4
    assert report.original_bytes == 4 * 8 * 8 * 4
    assert report.stored_bytes == 3 * 8 * 8 * 4
    assert report.saved_bytes == 8 * 8 * 4
    assert round(report.saved_ratio, 4) == 0.25
    assert check_aliases(frames, sources, report) == []


def test_every_declared_frame_resolves_to_identical_pixels() -> None:
    frames = _frames()
    sources = _sources(seeds=[4, 4, 4, 4])
    report = analyze_frames(frames, sources)
    assert report.unique == ("walk_0",)
    assert len(report.groups) == 1
    assert report.saved_bytes == 3 * 8 * 8 * 4
    for alias in report.aliases:
        assert sources[alias.canonical] == sources[alias.name]


def test_canonical_is_the_first_declared_occurrence() -> None:
    frames = _frames()
    sources = _sources(seeds=[9, 9, 7, 7])
    report = analyze_frames(frames, sources)
    assert report.unique == ("walk_0", "walk_2")
    assert report.canonical_of("walk_1") == "walk_0"
    assert report.canonical_of("walk_3") == "walk_2"


def test_differences_prevent_merging() -> None:
    frames = _frames()
    near = _sources(seeds=[5, 5, 5, 5])
    near["walk_1"][0][0] = tuple(list(near["walk_1"][0][0])[:-1] + [7])
    report = analyze_frames(frames, near)
    assert report.unique == ("walk_0", "walk_1")
    assert report.groups[0].members == ("walk_0", "walk_2", "walk_3")
    # Same bytes, different shape or channel count: never merged.
    reshaped = parse_frames([{"name": "a", "width": 2, "height": 3},
                             {"name": "b", "width": 3, "height": 2}])
    flat = [[(1, 2, 3), (4, 5, 6)], [(7, 8, 9), (10, 11, 12)], [(13, 14, 15), (16, 17, 18)]]
    other = [[(1, 2, 3), (4, 5, 6), (7, 8, 9)], [(10, 11, 12), (13, 14, 15), (16, 17, 18)]]
    report = analyze_frames(reshaped, {"a": flat, "b": other})
    assert report.unique == ("a", "b")
    # A page has one channel count, so a mixed RGB and RGBA set is refused outright
    # instead of being converted or merged.
    mixed = parse_frames([{"name": "a", "width": 2, "height": 2},
                          {"name": "b", "width": 2, "height": 2}])
    rgb = [[(1, 2, 3), (4, 5, 6)], [(7, 8, 9), (10, 11, 12)]]
    rgba = [pixel + (255,) for row in rgb for pixel in row]
    rgba_rows = [rgba[:2], rgba[2:]]
    with pytest.raises(ValueError):
        analyze_frames(mixed, {"a": rgb, "b": rgba_rows})


def test_no_duplicates_reports_the_full_size() -> None:
    frames = _frames()
    report = analyze_frames(frames, _sources(seeds=[1, 2, 3, 4]))
    assert report.unique == ("walk_0", "walk_1", "walk_2", "walk_3")
    assert report.groups == ()
    assert report.saved_bytes == 0
    assert report.saved_ratio == 0.0
    assert all(alias.canonical == alias.name for alias in report.aliases)
    assert check_aliases(frames, _sources(seeds=[1, 2, 3, 4]), report) == []


def test_unique_frames_keep_the_declaration_order() -> None:
    frames = _frames()
    report = analyze_frames(frames, _sources(seeds=[2, 1, 1, 2]))
    stored = unique_frames(frames, report)
    assert [frame.name for frame in stored] == ["walk_0", "walk_1"]
    assert unique_frames(frames, report)[0].width == 8


def test_content_key_separates_size_and_channels() -> None:
    rgb = _sprite(4, 4, 1, 3)
    rgba = _sprite(4, 4, 1, 4)
    assert content_key(rgb) != content_key(rgba)
    assert content_key(rgb) == content_key(deepcopy(rgb))
    assert content_key(rgb)[2] == 3 and content_key(rgba)[2] == 4


def test_analysis_refuses_broken_sources() -> None:
    frames = _frames()
    sources = _sources()
    missing = {name: image for name, image in sources.items() if name != "walk_1"}
    with pytest.raises(ValueError):
        analyze_frames(frames, missing)
    extra = dict(sources)
    extra["ghost"] = _sprite(8, 8, 5)
    with pytest.raises(ValueError):
        analyze_frames(frames, extra)
    wrong = dict(sources)
    wrong["walk_0"] = _sprite(8, 7, 1)
    with pytest.raises(ValueError):
        analyze_frames(frames, wrong)
    assert check_sources(frames, sources) == []


def test_analysis_does_not_mutate_the_sources() -> None:
    frames = _frames()
    sources = _sources(seeds=[1, 1, 2, 2])
    before = deepcopy(sources)
    report = analyze_frames(frames, sources)
    assert check_aliases(frames, sources, report) == []
    assert sources == before


def test_check_aliases_catches_tampering() -> None:
    frames = _frames()
    sources = _sources(seeds=[1, 2, 1, 2])
    report = analyze_frames(frames, sources)
    assert check_aliases(frames, sources, "report") == ["report must be a DedupReport"]
    dropped = replace(report, aliases=report.aliases[:-1])
    assert any("without an alias" in item for item in check_aliases(frames, sources, dropped))
    ghost = replace(report, aliases=report.aliases + (
        replace(report.aliases[0], name="ghost"),))
    assert any("undeclared frame" in item for item in check_aliases(frames, sources, ghost))
    crossed = replace(report, aliases=tuple(
        replace(alias, canonical="walk_1") if alias.name == "walk_0" else alias
        for alias in report.aliases
    ))
    assert any("does not match the pixels" in item
               for item in check_aliases(frames, sources, crossed))
    unstored = replace(report, unique=("walk_1", "walk_3"))
    assert any("not stored" in item or "not in declaration order" in item
               for item in check_aliases(frames, sources, unstored))
    unreported = replace(report, groups=())
    assert any("missing duplicate group" in item
               for item in check_aliases(frames, sources, unreported))
    inflated = replace(report, groups=(replace(report.groups[0], saved_bytes=1),))
    assert any("wrong savings" in item for item in check_aliases(frames, sources, inflated))
    miscounted = replace(report, original_bytes=report.original_bytes + 4)
    assert any("does not add up" in item for item in check_aliases(frames, sources, miscounted))
    misplaced = replace(report, groups=report.groups + (
        _group("walk_3", ("walk_0", "walk_2")),))
    assert any("has no aliases" in item for item in check_aliases(frames, sources, misplaced))


def test_check_aliases_catches_a_phantom_group() -> None:
    frames = _frames()
    sources = _sources(seeds=[1, 2, 3, 4])
    report = analyze_frames(frames, sources)
    assert report.groups == ()
    phantom = replace(report, groups=(_group("walk_0", ("walk_0", "walk_1")),))
    assert any("not a duplicate group" in item
               for item in check_aliases(frames, sources, phantom))


def _group(canonical, members):
    from bellium.atlas.dedup import DuplicateGroup
    return DuplicateGroup(canonical=canonical, members=members, pixels=64, stored_bytes=256,
                          saved_bytes=256 * (len(members) - 1))


def _query(sources, **overrides):
    query = {"frames": [{"name": name, "width": w, "height": h} for name, w, h in SPEC],
             "sources": sources, "width": 32, "height": 32}
    query.update(overrides)
    return query


def test_specialist_packs_only_the_distinct_frames() -> None:
    sources = _sources(seeds=[1, 2, 1, 4])
    result = dedup_atlas(_query(sources))
    assert result.abstained is False
    output = result.output
    assert output["status"] == "ready"
    assert output["frames_declared"] == 4 and output["frames_stored"] == 3
    assert output["unique"] == ["walk_0", "walk_1", "walk_3"]
    assert output["saved_bytes"] == 8 * 8 * 4
    assert output["placed"] == 3
    assert len(output["placements"]) == 3
    assert output["page_count"] == 1
    assert output["certified"] is False and output["written_files"] is False
    assert result.confidence == 0.9
    assert json.dumps(output)


def test_alias_map_lets_every_declared_frame_be_drawn() -> None:
    sources = _sources(seeds=[3, 3, 5, 8])
    output = dedup_atlas(_query(sources)).output
    regions = {item["name"]: item for item in output["placements"]}
    pages = output["pages"]
    aliases = {item["name"]: item["canonical"] for item in output["aliases"]}
    for name, source in sources.items():
        region = regions[aliases[name]]
        page = pages[region["page"]]
        for row in range(len(source)):
            for column in range(len(source[0])):
                drawn = tuple(page["pixels"][region["y"] + row][region["x"] + column])
                assert drawn == tuple(source[row][column]), (name, row, column)
    assert aliases["walk_1"] == "walk_0"


def test_specialist_encodes_verified_pages() -> None:
    sources = _sources(seeds=[1, 1, 2, 2])
    result = dedup_atlas(_query(sources, encode="png"))
    output = result.output
    assert output["status"] == "ready"
    assert output["frames_stored"] == 2
    data = output["page_bytes"][0]
    assert isinstance(data, bytes)
    assert output["png"][0]["sha256"] == hashlib.sha256(data).hexdigest()
    info = inspect_png(data)
    assert info.ok and info.channels == 4
    assert decode_png(data).digest == output["page_digests"][0]


def test_specialist_abstains_on_sources_and_fit() -> None:
    sources = _sources()
    missing = dedup_atlas(_query({name: image for name, image in sources.items()
                                  if name != "walk_2"}))
    assert missing.abstained is True
    assert missing.output["reason"] == "source_contract_violated"
    assert missing.output["source_violations"] == ["missing source image for frame walk_2"]
    tight = dedup_atlas(_query(sources, width=8, height=8))
    assert tight.output["reason"] == "frames_do_not_fit"
    assert tight.output["unplaced"]


def test_specialist_abstains_when_its_own_checks_fail(monkeypatch) -> None:
    sources = _sources(seeds=[1, 1, 2, 2])
    monkeypatch.setattr(specialist, "check_aliases", lambda *args, **kwargs: ["boom"])
    alias_failure = specialist.dedup_atlas(_query(sources))
    assert alias_failure.output["reason"] == "alias_failed_verification"
    assert alias_failure.output["alias_violations"] == ["boom"]
    monkeypatch.undo()
    monkeypatch.setattr(specialist, "check_raster", lambda *args, **kwargs: ["boom"])
    raster_failure = specialist.dedup_atlas(_query(sources))
    assert raster_failure.output["reason"] == "raster_failed_verification"
    monkeypatch.undo()
    from bellium.atlas.png import inspect_png as real_inspect
    encoded = specialist.dedup_atlas(_query(sources, encode="png"))
    info = real_inspect(encoded.output["page_bytes"][0])
    assert info.ok
    monkeypatch.setattr(specialist, "inspect_png",
                        lambda data: replace(info, violations=("boom",)))
    png_failure = specialist.dedup_atlas(_query(sources, encode="png"))
    assert png_failure.output["reason"] == "png_failed_verification"
    assert png_failure.output["png_violations"] == ["boom"]


@pytest.mark.parametrize("query", [
    {"frames": SPEC, "sources": {}, "width": 32, "height": 32, "profile": "godot-4"},
    {"frames": SPEC, "sources": {}, "width": 32, "height": 32, "encode": "jpeg"},
    {"frames": SPEC, "width": 32, "height": 32},
    {"frames": SPEC, "sources": None, "width": 32, "height": 32},
    ["frames"],
])
def test_specialist_refuses_invalid_queries(query) -> None:
    with pytest.raises(ValueError):
        dedup_atlas(query)


def test_specialist_does_not_mutate_its_inputs() -> None:
    sources = _sources(seeds=[1, 1, 2, 2])
    query = _query(sources)
    before_query = deepcopy(query)
    before_sources = deepcopy(sources)
    assert dedup_atlas(query).output["status"] == "ready"
    assert query == before_query
    assert sources == before_sources


def _jitter(image, *, amount, channel=0, at=(0, 0)):
    rows = [list(row) for row in image]
    pixel = list(rows[at[1]][at[0]])
    pixel[channel] = max(0, min(255, pixel[channel] + amount))
    rows[at[1]][at[0]] = tuple(pixel)
    return [[tuple(value) for value in row] for row in rows]


def test_bounded_tolerance_merges_within_the_declared_delta() -> None:
    frames = _frames()
    sources = _sources(seeds=[1, 1, 1, 1])
    sources["walk_1"] = _jitter(sources["walk_1"], amount=2, channel=1)
    exact = analyze_frames(frames, sources)
    assert exact.unique == ("walk_0", "walk_1")
    assert exact.worst_delta == 0
    bounded = analyze_frames(frames, sources, max_delta=3)
    assert bounded.unique == ("walk_0",)
    assert bounded.max_delta == 3
    assert bounded.worst_delta == 2
    assert bounded.canonical_of("walk_1") == "walk_0"
    assert [alias.delta for alias in bounded.aliases] == [0, 2, 0, 0]
    assert bounded.groups[0].max_delta == 2
    assert bounded.saved_bytes == 3 * 8 * 8 * 4
    assert check_aliases(frames, sources, bounded) == []
    assert check_aliases(frames, sources, exact) == []


def test_tolerance_beyond_the_bound_does_not_merge() -> None:
    frames = _frames()
    sources = _sources(seeds=[1, 1, 1, 1])
    sources["walk_1"] = _jitter(sources["walk_1"], amount=4)
    report = analyze_frames(frames, sources, max_delta=3)
    assert report.unique == ("walk_0", "walk_1")
    assert report.worst_delta == 0
    assert all("walk_1" not in group.members for group in report.groups)


def test_zero_tolerance_is_exact_only() -> None:
    frames = _frames()
    sources = _sources(seeds=[1, 1, 1, 1])
    sources["walk_1"] = _jitter(sources["walk_1"], amount=1)
    report = analyze_frames(frames, sources)
    assert report.max_delta == 0
    assert report.unique == ("walk_0", "walk_1")
    assert analyze_frames(frames, sources, max_delta=1).unique == ("walk_0",)


def test_alpha_always_counts() -> None:
    frames = _frames()
    sources = _sources(seeds=[2, 2, 2, 2])
    sources["walk_2"] = _jitter(sources["walk_2"], amount=-1, channel=3)
    assert analyze_frames(frames, sources).unique == ("walk_0", "walk_2")
    bounded = analyze_frames(frames, sources, max_delta=1)
    assert bounded.unique == ("walk_0",)
    assert bounded.worst_delta == 1


def test_neighbours_do_not_chain_transitively() -> None:
    frames = _frames([("a", 8, 8), ("b", 8, 8), ("c", 8, 8)])
    base = _sprite(8, 8, 1)
    sources = {
        "a": base,
        "b": _jitter(base, amount=1, channel=0, at=(0, 0)),
        "c": _jitter(base, amount=3, channel=0, at=(0, 0)),
    }
    report = analyze_frames(frames, sources, max_delta=1)
    assert report.canonical_of("b") == "a"
    assert report.canonical_of("c") == "c"
    assert report.unique == ("a", "c")
    assert report.worst_delta == 1
    assert check_aliases(frames, sources, report) == []


def test_check_aliases_catches_delta_tampering() -> None:
    frames = _frames()
    sources = _sources(seeds=[1, 1, 1, 1])
    sources["walk_1"] = _jitter(sources["walk_1"], amount=2, channel=1)
    report = analyze_frames(frames, sources, max_delta=3)
    understated = replace(report, aliases=tuple(
        replace(alias, delta=0) if alias.name == "walk_1" else alias for alias in report.aliases
    ))
    assert any("reports delta 0, measured 2" in item
               for item in check_aliases(frames, sources, understated))
    tightened = replace(report, max_delta=1)
    assert any("beyond the declared 1" in item
               for item in check_aliases(frames, sources, tightened))
    assert any("outside the declared bound" in item
               for item in check_aliases(frames, sources, tightened))
    bad_group = replace(report, groups=(replace(report.groups[0], max_delta=0),))
    assert any("reports max_delta 0, measured 2" in item
               for item in check_aliases(frames, sources, bad_group))
    bad_worst = replace(report, worst_delta=0)
    assert any("does not match the measured" in item
               for item in check_aliases(frames, sources, bad_worst))
    non_zero_canonical = replace(report, aliases=tuple(
        replace(alias, delta=1) if alias.name == "walk_0" else alias for alias in report.aliases
    ))
    assert any("non-zero delta" in item
               for item in check_aliases(frames, sources, non_zero_canonical))
    invalid = replace(report, max_delta=999)
    assert check_aliases(frames, sources, invalid) == ["report max_delta is invalid: max_delta must be between 0 and 255"]


def test_check_aliases_catches_a_skipped_earlier_frame() -> None:
    frames = _frames()
    sources = _sources(seeds=[3, 4, 3, 4])
    wide = analyze_frames(frames, sources, max_delta=200)
    assert wide.unique == ("walk_0",)
    skipped = replace(wide, aliases=tuple(
        replace(alias, canonical="walk_1", delta=2) if alias.name == "walk_2" else alias
        for alias in wide.aliases
    ), unique=("walk_0", "walk_1"), stored_bytes=wide.stored_bytes + 8 * 8 * 4)
    assert any("skipped the earlier stored frame" in item
               for item in check_aliases(frames, sources, skipped))


@pytest.mark.parametrize("value", [True, -1, 256, 2.5, "3", None])
def test_the_declared_bound_is_validated(value) -> None:
    frames = _frames()
    sources = _sources()
    with pytest.raises(ValueError):
        analyze_frames(frames, sources, max_delta=value)
    with pytest.raises(ValueError):
        dedup_atlas(_query(sources, max_delta=value))


def test_specialist_exposes_the_declared_bound() -> None:
    sources = _sources(seeds=[1, 1, 1, 1])
    sources["walk_1"] = _jitter(sources["walk_1"], amount=2, channel=1)
    exact = dedup_atlas(_query(sources)).output
    assert exact["max_delta"] == 0 and exact["frames_stored"] == 2
    bounded = dedup_atlas(_query(sources, max_delta=2)).output
    assert bounded["status"] == "ready"
    assert bounded["max_delta"] == 2 and bounded["worst_delta"] == 2
    assert bounded["frames_stored"] == 1
    assert bounded["saved_bytes"] == 3 * 8 * 8 * 4
    assert all(alias["delta"] <= 2 for alias in bounded["aliases"])
    assert bounded["groups"][0]["max_delta"] == 2
    assert json.dumps(bounded)
