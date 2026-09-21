from pathlib import Path

from bellium.editing.quantize import quantize_ordered
from bellium.magick.cli import main
from bellium.magick.io import load_ppm, save_ppm, load_image, save_image

def _sample_image():
    return [
        [(10, 20, 30), (40, 50, 60)],
        [(70, 80, 90), (100, 110, 120)],
    ]

def test_ppm_io_roundtrip():
    img = _sample_image()
    p6 = save_ppm(img, 'P6')
    assert p6.startswith(b'P6\n2 2\n255\n')
    loaded_p6 = load_ppm(p6)
    assert loaded_p6 == img
    p3 = save_ppm(img, 'P3')
    assert p3.startswith(b'P3\n2 2\n255')
    loaded_p3 = load_ppm(p3)
    assert loaded_p3 == img

def test_quantize_ordered():
    img = [
        [(i * 10, i * 10, i * 10) for i in range(8)]
        for _ in range(8)
    ]
    palette = [(0, 0, 0), (70, 70, 70)]
    ordered = quantize_ordered(img, palette)
    assert len(ordered) == 8 and len(ordered[0]) == 8
    for row in ordered:
        for px in row:
            assert px in palette

def test_cli_end_to_end(tmp_path: Path):
    img = [
        [(10, 10, 10), (100, 100, 100), (200, 200, 200), (250, 250, 250)],
        [(10, 10, 10), (100, 100, 100), (200, 200, 200), (250, 250, 250)],
        [(20, 20, 20), (110, 110, 110), (210, 210, 210), (255, 255, 255)],
        [(20, 20, 20), (110, 110, 110), (210, 210, 210), (255, 255, 255)],
    ]
    src_ppm = tmp_path / 'input.ppm'
    save_image(img, src_ppm)
    
    # info
    ret = main(['info', str(src_ppm)])
    assert ret == 0
    
    # resize
    out_resize = tmp_path / 'resized.ppm'
    ret = main(['resize', str(src_ppm), str(out_resize), '--width', '8', '--height', '8', '--method', 'nano_edge'])
    assert ret == 0
    loaded = load_image(out_resize)
    assert len(loaded) == 8 and len(loaded[0]) == 8
    
    # quantize
    out_quant = tmp_path / 'quant.ppm'
    ret = main(['quantize', str(src_ppm), str(out_quant), '--colors', '4', '--dither', 'nano_tone'])
    assert ret == 0
    assert out_quant.is_file()
    
    # filter
    out_filt = tmp_path / 'filtered.ppm'
    ret = main(['filter', str(src_ppm), str(out_filt), '--type', 'nano_adaptive'])
    assert ret == 0
    assert out_filt.is_file()
    
    # tone
    out_tone = tmp_path / 'tone.ppm'
    ret = main(['tone', str(src_ppm), str(out_tone), '--mode', 'micro_nn'])
    assert ret == 0
    assert out_tone.is_file()
    
    # threshold
    out_thresh = tmp_path / 'thresh.ppm'
    ret = main(['threshold', str(src_ppm), str(out_thresh), '--mode', 'otsu'])
    assert ret == 0
    assert out_thresh.is_file()
    
    # morphology
    out_morph = tmp_path / 'morph.ppm'
    ret = main(['morphology', str(out_thresh), str(out_morph), '--op', 'dilate', '--radius', '1'])
    assert ret == 0
    assert out_morph.is_file()
    
    # compare
    diff_path = tmp_path / 'diff.ppm'
    ret = main(['compare', str(src_ppm), str(out_filt), '--diff-output', str(diff_path)])
    assert ret == 0
    assert diff_path.is_file()
    
    # transform
    out_rot = tmp_path / 'rotated.ppm'
    ret = main(['transform', str(src_ppm), str(out_rot), '--op', 'rotate90'])
    assert ret == 0
    assert out_rot.is_file()
    
    # composite
    out_comp = tmp_path / 'composite.ppm'
    ret = main(['composite', str(src_ppm), str(out_resize), str(out_comp), '--mode', 'multiply'])
    assert ret == 0
    assert out_comp.is_file()
    
    # channel extract
    out_ch = tmp_path / 'channel_r.ppm'
    ret = main(['channel', str(src_ppm), str(out_ch), '--action', 'extract', '--channel', 'r'])
    assert ret == 0
    assert out_ch.is_file()
    
    # montage
    out_montage = tmp_path / 'montage.ppm'
    ret = main(['montage', str(src_ppm), str(out_ch), '-o', str(out_montage), '--columns', '2'])
    assert ret == 0
    assert out_montage.is_file()
