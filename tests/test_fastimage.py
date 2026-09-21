from pathlib import Path
import struct

from bellium.knn.hash_matcher import match_hash_knn
from bellium.magick.auto_orient import auto_orient
from bellium.magick.cli import main
from bellium.magick.fastimage import fast_info
from bellium.magick.geometry_parser import parse_geometry
from bellium.magick.hashes import compute_ahash, compute_dhash, compute_phash, hamming_distance
from bellium.magick.io import save_image

def _sample_image():
    return [
        [(10, 10, 10), (100, 100, 100), (200, 200, 200), (250, 250, 250)],
        [(10, 10, 10), (100, 100, 100), (200, 200, 200), (250, 250, 250)],
        [(20, 20, 20), (110, 110, 110), (210, 210, 210), (255, 255, 255)],
        [(20, 20, 20), (110, 110, 110), (210, 210, 210), (255, 255, 255)],
    ]

def test_fastimage_sniffing_synthetic_headers():
    # PNG header
    png_bytes = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR' + struct.pack('>IIBB', 1920, 1080, 8, 2) + b'\x00\x00\x00' + b'\x00' * 50
    info_png = fast_info(png_bytes)
    assert info_png.format == 'PNG'
    assert info_png.width == 1920 and info_png.height == 1080
    assert info_png.channels == 3
    
    # GIF header
    gif_bytes = b'GIF89a' + struct.pack('<HH', 320, 240) + b'\x80\x00\x00' + b'\x00' * 50
    info_gif = fast_info(gif_bytes)
    assert info_gif.format == 'GIF'
    assert info_gif.width == 320 and info_gif.height == 240
    
    # BMP header
    bmp_bytes = b'BM\x00\x00\x00\x00\x00\x00\x00\x00\x36\x00\x00\x00\x28\x00\x00\x00' + struct.pack('<iiH', 640, 480, 24) + b'\x00' * 50
    info_bmp = fast_info(bmp_bytes)
    assert info_bmp.format == 'BMP'
    assert info_bmp.width == 640 and info_bmp.height == 480
    
    # PPM header
    ppm_bytes = b'P6\n800 600\n255\n' + b'\x00' * 100
    info_ppm = fast_info(ppm_bytes)
    assert info_ppm.format == 'PPM'
    assert info_ppm.width == 800 and info_ppm.height == 600

def test_geometry_parser():
    w, h, crop = parse_geometry('800x600', 1600, 1200)
    assert w == 800 and h == 600 and crop is None
    
    w, h, crop = parse_geometry('800x600!', 1600, 900)
    assert w == 800 and h == 600
    
    w, h, crop = parse_geometry('50%', 1000, 800)
    assert w == 500 and h == 400
    
    w, h, crop = parse_geometry('800x600#', 1000, 500)
    assert crop is not None
    assert (crop[2] - crop[0]) == 600 and (crop[3] - crop[1]) == 800

def test_perceptual_hashes_and_knn_matching():
    img1 = _sample_image()
    img2 = _sample_image()
    # Identical images must yield 0 distance
    a1 = compute_ahash(img1)
    a2 = compute_ahash(img2)
    assert hamming_distance(a1, a2) == 0
    d1 = compute_dhash(img1)
    d2 = compute_dhash(img2)
    assert hamming_distance(d1, d2) == 0
    p1 = compute_phash(img1)
    p2 = compute_phash(img2)
    assert hamming_distance(p1, p2) == 0
    
    # Gallery k-NN matching
    gallery = [
        {'id': 'asset_01', 'hash': p1},
        {'id': 'asset_random', 'hash': 0x0123456789abcdef},
    ]
    res = match_hash_knn(p1, gallery, k=1)
    assert not res.abstained
    assert res.output['best_match_id'] == 'asset_01'
    assert res.output['best_distance'] == 0
    assert res.output['is_duplicate'] is True

def test_auto_orient():
    img = _sample_image()
    assert auto_orient(img, 1) == img
    o6 = auto_orient(img, 6)  # 90 CW
    assert len(o6) == len(img[0]) and len(o6[0]) == len(img)

def test_cli_fastimage_and_hashes(tmp_path: Path):
    img = _sample_image()
    src = tmp_path / 'test.ppm'
    save_image(img, src)
    
    # fastimage command
    ret = main(['fastimage', str(src)])
    assert ret == 0
    
    # phash command
    ret = main(['phash', str(src)])
    assert ret == 0
    
    # auto-orient command
    out_orient = tmp_path / 'oriented.ppm'
    ret = main(['auto-orient', str(src), str(out_orient)])
    assert ret == 0
    assert out_orient.is_file()
    
    # resize with geometry 50%
    out_geom = tmp_path / 'geom.ppm'
    ret = main(['resize', str(src), str(out_geom), '--geometry', '50%'])
    assert ret == 0
    assert out_geom.is_file()
