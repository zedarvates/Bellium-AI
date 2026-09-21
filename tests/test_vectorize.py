from pathlib import Path

from bellium.editing.compare import compare_images
from bellium.editing.vectorize import make_pelican_bicycle, rasterize_regions, vectorize_regions
from bellium.knn.vector_region import classify_vector_region
from bellium.magick.cli import main
from bellium.magick.io import save_image
from bellium.specialists.vectorize import raster_to_svg


def test_pelican_bicycle_becomes_svg_paths() -> None:
    scene = make_pelican_bicycle()
    result = raster_to_svg({"image": scene, "max_colors": 8, "min_area": 3, "epsilon": 0.4})
    assert not result.abstained
    svg = result.output["svg"]
    assert svg.startswith("<svg ")
    assert "<path d=" in svg
    assert "rgb(28,28,32)" in svg or "rgb(236,132,36)" in svg
    assert result.output["region_count"] >= 3
    preview = result.output["preview"]
    metrics = compare_images(scene, preview)
    assert metrics["mismatch_ratio"] < 0.35


def test_quantized_flat_art_roundtrips_without_simplify() -> None:
    image = [
        [(255, 0, 0), (255, 0, 0), (0, 0, 255)],
        [(255, 0, 0), (255, 0, 0), (0, 0, 255)],
        [(0, 255, 0), (0, 255, 0), (0, 255, 0)],
    ]
    payload = vectorize_regions(image, max_colors=3, min_area=1, epsilon=0.0, skip_background=False)
    rebuilt = rasterize_regions(payload)
    assert rebuilt == image


def test_noisy_photo_region_is_not_traced() -> None:
    noisy = []
    seed = 17
    for r in range(12):
        row = []
        for c in range(12):
            seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
            v = seed % 256
            row.append((v, (v * 3) % 256, (v * 7) % 256))
        noisy.append(row)
    result = raster_to_svg({"image": noisy, "max_colors": 12, "min_area": 1})
    assert result.abstained
    assert result.output["reason"] in {"no_vector_safe_regions", "photographic_unique_colors"}


def test_vector_region_knn_splits_fill_and_photo() -> None:
    fill = classify_vector_region(
        {"luma_std": 0.02, "unique_ratio": 0.03, "compactness": 0.8, "fill_ratio": 0.8, "chroma": 0.2}
    )
    assert not fill.abstained
    assert fill.output["label"] == "flat_fill"
    photo = classify_vector_region(
        {"luma_std": 0.4, "unique_ratio": 0.5, "compactness": 0.3, "fill_ratio": 0.5, "chroma": 0.2}
    )
    assert photo.abstained


def test_cli_vectorize_and_mogrify(tmp_path: Path) -> None:
    scene = make_pelican_bicycle()
    src = tmp_path / "pelican.ppm"
    save_image(scene, src)
    svg_path = tmp_path / "pelican.svg"
    assert main(["vectorize", str(src), str(svg_path), "--colors", "8"]) == 0
    text = svg_path.read_text(encoding="utf-8")
    assert "<svg" in text and "<path" in text
    out_dir = tmp_path / "batch"
    assert main(["mogrify", str(src), "--output-dir", str(out_dir), "--format", "ppm", "--resize", "50%"]) == 0
    assert (out_dir / "pelican.ppm").is_file()
    sheet = tmp_path / "sheet.ppm"
    save_image([[(10, 10, 10), (20, 20, 20)], [(30, 30, 30), (40, 40, 40)]], sheet)
    frames = tmp_path / "frames"
    assert main(["spritesheet-slice", str(sheet), str(frames), "--frame-width", "1", "--frame-height", "1"]) == 0
    assert len(list(frames.glob("frame_*.ppm"))) == 4
