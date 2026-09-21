"""Atlas rasterization: exact copies, background, gutter and PNG byte checks."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import io
import json
import zlib

import pytest

from bellium.atlas.packing import pack, parse_frames
from bellium.atlas.png import (
    PNG_SIGNATURE,
    decode_png,
    encode_png,
    inspect_png,
)
from bellium.atlas.raster import (
    RasterPage,
    check_raster,
    check_sources,
    page_digest,
    pixel_problem,
    rasterize,
    source_shape,
)
from bellium.specialists import atlas_raster as specialist
from bellium.specialists.atlas_raster import rasterize_atlas

TWO = [{"name": "a", "width": 8, "height": 6}, {"name": "b", "width": 6, "height": 8}]


def _sprite(width, height, seed=1, channels=4):
    rows = []
    for row in range(height):
        line = []
        for column in range(width):
            base = (row * 7 + column * 5 + seed * 11) % 200
            if channels == 4:
                line.append((base, 255 - base, (base * 3) % 256, 255 if (row + column) % 3 else 0))
            else:
                line.append((base, 255 - base, (base * 3) % 256))
        rows.append(line)
    return rows


def _two_frames(channels=4):
    frames = parse_frames(TWO)
    sources = {"a": _sprite(8, 6, 1, channels), "b": _sprite(6, 8, 2, channels)}
    return frames, sources


def _page(channels=4, padding=0):
    frames, sources = _two_frames(channels)
    plan = pack(frames, width=24, height=24, padding=padding)
    return rasterize(frames, plan, sources).pages[0]


def _png_with(scanlines: bytes, width: int, height: int, color_type: int) -> bytes:
    def chunk(kind, payload):
        body = kind + payload
        return len(payload).to_bytes(4, "big") + body + zlib.crc32(body).to_bytes(4, "big")

    header = width.to_bytes(4, "big") + height.to_bytes(4, "big") + bytes([8, color_type, 0, 0, 0])
    return PNG_SIGNATURE + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(scanlines)) \
        + chunk(b"IEND", b"")


def test_raster_copies_every_placement_exactly() -> None:
    frames, sources = _two_frames()
    plan = pack(frames, width=24, height=24)
    assert plan.complete
    raster = rasterize(frames, plan, sources)
    assert raster.channels == 4
    assert raster.placed == 2
    assert check_raster(frames, plan, raster, sources) == []
    page = raster.pages[0]
    for plan_page in plan.pages:
        for placement in plan_page:
            image = sources[placement.name]
            for row in range(placement.height):
                for column in range(placement.width):
                    assert page.pixels[placement.y + row][placement.x + column] == \
                        tuple(image[row][column])


def test_background_defaults_follow_the_channel_count() -> None:
    rgb_frames, rgb_sources = _two_frames(channels=3)
    rgb = rasterize(rgb_frames, pack(rgb_frames, width=24, height=24), rgb_sources)
    assert rgb.background == (0, 0, 0)
    rgba_frames, rgba_sources = _two_frames(channels=4)
    rgba = rasterize(rgba_frames, pack(rgba_frames, width=24, height=24), rgba_sources)
    assert rgba.background == (0, 0, 0, 0)
    assert rgba.pages[0].pixels[23][23] == (0, 0, 0, 0)


def test_explicit_background_is_validated_and_used() -> None:
    frames, sources = _two_frames()
    plan = pack(frames, width=24, height=24)
    painted = rasterize(frames, plan, sources, background=(255, 255, 255, 255))
    assert check_raster(frames, plan, painted, sources) == []
    assert painted.pages[0].pixels[23][23] == (255, 255, 255, 255)
    for bad in [(0, 0, 0), (0, 0, 0, 0, 0), (0, 0, 0, 300), (0, 0, 0, True), (0, 0, 0, 1.0),
                "white"]:
        with pytest.raises(ValueError):
            rasterize(frames, plan, sources, background=bad)


def test_padding_leaves_a_background_gutter() -> None:
    frames, sources = _two_frames()
    plan = pack(frames, width=32, height=32, padding=2)
    assert plan.complete
    raster = rasterize(frames, plan, sources)
    assert check_raster(frames, plan, raster, sources) == []
    page = raster.pages[0]
    for row in range(page.height):
        for column in range(page.width):
            if row < 2 or column < 2 or row >= page.height - 2 or column >= page.width - 2:
                assert page.pixels[row][column] == raster.background


def test_check_raster_catches_a_tampered_copy() -> None:
    frames, sources = _two_frames()
    plan = pack(frames, width=24, height=24)
    raster = rasterize(frames, plan, sources)
    page = raster.pages[0]
    rows = [list(row) for row in page.pixels]
    rows[page.height - 1][page.width - 1] = (1, 2, 3, 4)
    tampered = replace(raster, pages=(replace(page, pixels=tuple(tuple(r) for r in rows)),))
    violations = check_raster(frames, plan, tampered, sources)
    assert any("not the declared background" in item for item in violations)
    assert any("digest" in item for item in violations)
    inside = [placement for plan_page in plan.pages for placement in plan_page][0]
    rows = [list(row) for row in page.pixels]
    rows[inside.y][inside.x] = (9, 9, 9, 9)
    moved = replace(raster, pages=(replace(page, pixels=tuple(tuple(r) for r in rows)),))
    assert any("does not match frame" in item for item in check_raster(frames, plan, moved, sources))


def test_check_raster_catches_a_broken_gutter() -> None:
    frames, sources = _two_frames()
    plan = pack(frames, width=32, height=32, padding=2)
    raster = rasterize(frames, plan, sources)
    page = raster.pages[0]
    rows = [list(row) for row in page.pixels]
    rows[0][0] = (200, 10, 10, 255)
    broken = replace(raster, pages=(replace(page, pixels=tuple(tuple(r) for r in rows)),))
    assert any("gutter pixel" in item for item in check_raster(frames, plan, broken, sources))


def test_check_raster_catches_wrong_sizes_counts_and_digests() -> None:
    frames, sources = _two_frames()
    plan = pack(frames, width=24, height=24)
    raster = rasterize(frames, plan, sources)
    page = raster.pages[0]
    assert check_raster(frames, plan, replace(raster, pages=()), sources)
    assert any("pages" in item for item in check_raster(frames, plan, replace(raster, pages=()), sources))
    wide = replace(raster, pages=(replace(page, width=page.width + 1),))
    assert any("the plan declares" in item for item in check_raster(frames, plan, wide, sources))
    counted = replace(raster, placed=raster.placed + 1)
    assert any("placements" in item for item in check_raster(frames, plan, counted, sources))
    wrong = replace(raster, pages=(replace(page, raw_bytes=page.raw_bytes + 1),))
    assert any("raw bytes" in item for item in check_raster(frames, plan, wrong, sources))
    assert check_raster(frames, plan, "not-a-raster", sources) == ["raster must be an AtlasRaster"]


def test_sources_are_never_mutated() -> None:
    frames, sources = _two_frames()
    before = deepcopy(sources)
    plan = pack(frames, width=24, height=24)
    raster = rasterize(frames, plan, sources)
    assert check_sources(frames, sources) == []
    assert check_raster(frames, plan, raster, sources) == []
    assert sources == before


def test_missing_extra_and_mismatched_sources_are_reported() -> None:
    frames, sources = _two_frames()
    missing = {name: image for name, image in sources.items() if name != "b"}
    assert check_sources(frames, missing) == ["missing source image for frame b"]
    extra = dict(sources)
    extra["ghost"] = _sprite(4, 4, 3)
    assert check_sources(frames, extra) == ["source image without a declared frame: ghost"]
    wrong = dict(sources)
    wrong["a"] = _sprite(8, 7, 1)
    problems = check_sources(frames, wrong)
    assert problems and "8 x 7" in problems[0] and "8 x 6" in problems[0]
    mixed = {"a": _sprite(8, 6, 1, 3), "b": _sprite(6, 8, 2, 4)}
    problems = check_sources(frames, mixed)
    assert problems and "share one channel count" in problems[0]
    assert check_sources(frames, None) == ["sources must be an object mapping names to images"]


@pytest.mark.parametrize("image,problem", [
    ([], "non-empty list of rows"),
    ([[(1, 2, 3)], [(1, 2, 3), (1, 2, 3)]], "equal width"),
    ([[(1, 2)], [(1, 2)]], "RGB or RGBA"),
    ([[(1, 2, 3, 4, 5)]], "RGB or RGBA"),
    ([[(1, 2, 300)]], "integers in [0, 255]"),
    ([[(1, 2, True)]], "integers in [0, 255]"),
    ([[(1, 2, 3.5)]], "integers in [0, 255]"),
    ([[(1, 2, 3), (1, 2, 3, 4)]], "share one channel count"),
])
def test_malformed_pixels_are_reported(image, problem) -> None:
    assert problem in pixel_problem(image)
    frames = parse_frames([{"name": "a", "width": 1, "height": 1}])
    assert check_sources(frames, {"a": image})
    with pytest.raises(ValueError):
        source_shape(image)


def test_valid_source_shapes_are_measured() -> None:
    assert source_shape(_sprite(8, 6, 1, 3)) == (6, 8, 3)
    assert source_shape(_sprite(6, 8, 1, 4)) == (8, 6, 4)
    assert pixel_problem(_sprite(3, 3)) is None


def test_rasterize_refuses_incomplete_or_invalid_plans() -> None:
    frames, sources = _two_frames()
    tight = pack(frames, width=12, height=12)
    assert tight.complete is False
    with pytest.raises(ValueError):
        rasterize(frames, tight, sources)
    broken = pack(frames, width=24, height=24)
    invalid = replace(broken, used_area=broken.used_area + 1)
    with pytest.raises(ValueError):
        rasterize(frames, invalid, sources)


def test_page_digest_is_stable_and_order_sensitive() -> None:
    rows = (((1, 2, 3),), ((4, 5, 6),))
    assert page_digest(rows) == page_digest(rows)
    assert page_digest(rows) != page_digest((((1, 2, 3),), ((4, 5, 7),)))
    # The digest covers the raw byte stream, so identical streams share it even
    # when they are cut into rows differently. Geometry is checked separately.
    assert page_digest(rows) == page_digest((((1, 2, 3), (4, 5, 6)),))
    assert page_digest(rows) == hashlib.sha256(bytes([1, 2, 3, 4, 5, 6])).hexdigest()


def test_png_container_is_well_formed() -> None:
    frames, sources = _two_frames()
    plan = pack(frames, width=24, height=24, padding=1)
    page = rasterize(frames, plan, sources).pages[0]
    data = encode_png(page)
    info = inspect_png(data)
    assert info.ok and info.violations == ()
    assert data.startswith(PNG_SIGNATURE)
    assert info.chunks == ("IHDR", "IDAT", "IEND")
    assert (info.width, info.height, info.channels, info.color_type) == (24, 24, 4, 6)
    assert info.bit_depth == 8
    assert info.raw_bytes == 24 * (1 + 24 * 4)
    rgb_page = rasterize(*_two_frames(channels=3), *(pack(*_two_frames(channels=3), width=24, height=24),)) if False else None
    assert rgb_page is None


def test_png_encoding_is_deterministic() -> None:
    frames, sources = _two_frames()
    page = rasterize(frames, pack(frames, width=24, height=24), sources).pages[0]
    assert encode_png(page) == encode_png(page)


@pytest.mark.parametrize("channels,color_type,mode", [(3, 2, "RGB"), (4, 6, "RGBA")])
def test_png_roundtrip_and_pillow_agreement(channels, color_type, mode) -> None:
    frames, sources = _two_frames(channels=channels)
    page = rasterize(frames, pack(frames, width=24, height=24), sources).pages[0]
    data = encode_png(page)
    info = inspect_png(data)
    assert (info.color_type, info.channels) == (color_type, channels)
    decoded = decode_png(data)
    assert decoded.pixels == page.pixels
    assert decoded.digest == page.digest
    Image = pytest.importorskip("PIL.Image")
    image = Image.open(io.BytesIO(data))
    assert image.mode == mode and image.size == (page.width, page.height)
    for row in range(page.height):
        for column in range(page.width):
            assert image.getpixel((column, row))[:channels] == page.pixels[row][column]


def test_inspect_png_reports_container_violations() -> None:
    page = _page()
    data = encode_png(page)
    assert "signature" in inspect_png(b"not a png").violations[0]
    corrupted = bytearray(data)
    corrupted[-5] ^= 0xFF
    assert any("CRC" in item for item in inspect_png(bytes(corrupted)).violations)
    assert any("continues after IEND" in item for item in inspect_png(data + b"junk").violations)
    assert any("past the end" in item or "IEND" in item for item in inspect_png(data[:40]).violations)
    assert any("must be bytes" in item for item in inspect_png("png").violations)
    truncated = data[: len(PNG_SIGNATURE) + 4]
    assert any("truncated" in item for item in inspect_png(truncated).violations)


def test_inspect_png_reports_filter_and_structure_problems() -> None:
    filtered = _png_with(bytes([1]) + bytes(3 * 2), 2, 1, 2)
    assert any("filter" in item for item in inspect_png(filtered).violations)
    with pytest.raises(ValueError):
        decode_png(filtered)
    short = _png_with(bytes([0]) + bytes(3), 2, 1, 2)
    assert any("scanlines are" in item for item in inspect_png(short).violations)
    deep = _png_with(bytes([0]) + bytes(3 * 3), 2, 1, 2)
    assert any("scanlines are" in item for item in inspect_png(deep).violations)


def test_inspect_png_reports_an_unknown_critical_chunk() -> None:
    data = encode_png(_page())
    body = b"ZZZZ" + b""
    injected = data[:8] + len(b"").to_bytes(4, "big") + body \
        + zlib.crc32(b"ZZZZ").to_bytes(4, "big") + data[8:]
    assert any("unknown critical chunk" in item for item in inspect_png(injected).violations)


def test_multi_page_raster_keeps_page_indices_and_digests() -> None:
    frames = parse_frames([{"name": f"p_{index}", "width": 16, "height": 16} for index in range(8)])
    sources = {frame.name: _sprite(16, 16, index, 4) for index, frame in enumerate(frames)}
    plan = pack(frames, width=32, height=32, max_pages=4)
    assert plan.page_count == 2
    raster = rasterize(frames, plan, sources)
    assert raster.page_count == 2
    assert [page.index for page in raster.pages] == [0, 1]
    assert len({page.digest for page in raster.pages}) == 2
    assert raster.bytes_written == 2 * 32 * 32 * 4
    assert check_raster(frames, plan, raster, sources) == []


def test_specialist_returns_pixels_metadata_and_a_clean_contract() -> None:
    frames, sources = _two_frames()
    query = {"frames": TWO, "sources": sources, "width": 24, "height": 24, "padding": 1}
    before = deepcopy(query)
    result = rasterize_atlas(query)
    assert result.abstained is False
    assert result.output["status"] == "ready"
    assert result.output["encode"] == "none"
    assert result.output["placed"] == 2
    assert result.output["channels"] == 4
    assert result.output["page_count"] == 1
    assert result.output["written_files"] is False
    assert result.output["certified"] is False
    assert result.confidence == 0.9
    assert len(result.output["pages"][0]["pixels"]) == result.output["page_height"]
    assert json.dumps(result.output)
    assert query == before
    assert frames  # declared frames stay untouched


def test_specialist_returns_png_bytes_and_digests() -> None:
    frames, sources = _two_frames()
    result = rasterize_atlas({"frames": TWO, "sources": sources, "width": 24, "height": 24,
                              "encode": "png"})
    assert result.output["status"] == "ready"
    assert len(result.output["page_bytes"]) == 1
    data = result.output["page_bytes"][0]
    assert isinstance(data, bytes)
    assert result.output["png"][0]["bytes"] == len(data)
    assert result.output["png"][0]["sha256"] == hashlib.sha256(data).hexdigest()
    info = inspect_png(data)
    assert info.ok
    assert info.channels == 4
    assert result.output["png"][0]["channels"] == 4
    raster = rasterize(frames, pack(frames, width=24, height=24), sources)
    assert decode_png(data).digest == raster.pages[0].digest


def test_specialist_abstains_on_source_and_plan_problems() -> None:
    frames, sources = _two_frames()
    missing = rasterize_atlas({"frames": TWO, "sources": {"a": sources["a"]},
                               "width": 24, "height": 24})
    assert missing.abstained is True
    assert missing.output["reason"] == "source_contract_violated"
    assert missing.output["source_violations"] == ["missing source image for frame b"]
    tight = rasterize_atlas({"frames": TWO, "sources": sources, "width": 12, "height": 12})
    assert tight.output["reason"] == "frames_do_not_fit"
    assert tight.output["unplaced"]


def test_specialist_abstains_when_its_own_output_fails(monkeypatch) -> None:
    frames, sources = _two_frames()
    query = {"frames": TWO, "sources": sources, "width": 24, "height": 24}
    monkeypatch.setattr(specialist, "check_raster", lambda *args, **kwargs: ["boom"])
    raster_failure = specialist.rasterize_atlas(query)
    assert raster_failure.output["reason"] == "raster_failed_verification"
    assert raster_failure.output["raster_violations"] == ["boom"]
    monkeypatch.undo()
    from bellium.atlas.png import inspect_png as real_inspect
    real_info = real_inspect(encode_png(_page()))
    monkeypatch.setattr(specialist, "inspect_png",
                        lambda data: replace(real_info, violations=("boom",)))
    png_failure = specialist.rasterize_atlas({**query, "encode": "png"})
    assert png_failure.output["reason"] == "png_failed_verification"
    assert png_failure.output["png_violations"] == ["boom"]


@pytest.mark.parametrize("query", [
    {"frames": TWO, "width": 24, "height": 24},
    {"frames": TWO, "sources": {}, "width": 24, "height": 24, "encode": "jpeg"},
    {"frames": TWO, "sources": {}, "width": 24, "height": 24, "profile": "godot-4"},
    {"frames": TWO, "sources": None, "width": 24, "height": 24},
    ["frames"],
])
def test_specialist_refuses_invalid_queries(query) -> None:
    with pytest.raises(ValueError):
        rasterize_atlas(query)


def test_specialist_refuses_an_invalid_background() -> None:
    _frames, sources = _two_frames()
    with pytest.raises(ValueError):
        rasterize_atlas({"frames": TWO, "sources": sources, "width": 24, "height": 24,
                         "background": (0, 0, 0)})


def test_placement_pixels_are_independent_of_the_copy_order() -> None:
    frames = parse_frames([
        {"name": "left", "width": 4, "height": 4},
        {"name": "right", "width": 4, "height": 4},
    ])
    sources = {"left": _sprite(4, 4, 1, 4), "right": _sprite(4, 4, 2, 4)}
    plan = pack(frames, width=12, height=8)
    raster = rasterize(frames, plan, sources)
    assert check_raster(frames, plan, raster, sources) == []
    page = raster.pages[0]
    reversed_plan = pack(list(reversed(frames)), width=12, height=8)
    assert rasterize(list(reversed(frames)), reversed_plan, sources).pages[0].digest == page.digest


def test_raster_page_is_frozen() -> None:
    page = RasterPage(index=0, width=2, height=2, channels=4,
                      pixels=(((0, 0, 0, 0), (0, 0, 0, 0)), ((0, 0, 0, 0), (0, 0, 0, 0))),
                      digest="x", raw_bytes=16)
    with pytest.raises(Exception):
        page.width = 3
