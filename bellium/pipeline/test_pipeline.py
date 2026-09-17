import os, sys
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.abspath('temp_bellium_repo'))
from bellium.pipeline import AssetPrepPipeline, AssetSpec

def run_tests():
    # Test 1: Full pipeline on a character sprite with scratch and transparent export
    # 100x100 white bg with a blue circle character (radius 30 around (50, 50))
    im = Image.new('RGB', (100, 100), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    draw.ellipse([20, 20, 80, 80], fill=(20, 80, 200))
    
    # Defect: yellow scratch inside the circle
    draw.rectangle([45, 45, 55, 55], fill=(255, 255, 0))
    
    defect_mask = Image.new('L', (100, 100), 0)
    draw_m = ImageDraw.Draw(defect_mask)
    draw_m.rectangle([45, 45, 55, 55], fill=255)
    
    pipe = AssetPrepPipeline(AssetSpec(
        target_width=128,
        target_height=128,
        bg_mode='transparent',
        auto_inpaint_defects=True,
    ))
    
    report = pipe.process_asset(im, defect_mask=defect_mask)
    print('Pipeline report:')
    print(f'  verdict: {report.verdict}, conf: {report.confidence}')
    print(f'  inpaint_applied: {report.inpaint_applied}, repaired_pixels: {report.inpaint_pixels_filled}')
    for line in report.log:
        print('   -', line)
        
    assert report.verdict == 'needs_review'
    assert report.inpaint_applied is True
    assert report.output_image.size == (128, 128)
    assert report.output_image.mode == 'RGBA'
    # Center pixel should have been repaired and not be yellow (255, 255, 0)
    center_pixel = report.output_image.getpixel((64, 64))
    print('Center pixel in output:', center_pixel)
    assert center_pixel[3] > 200  # Opaque foreground
    assert center_pixel[:3] != (255, 255, 0)  # Defect successfully healed
    
    # Test 2: Solid white export mode
    pipe_white = AssetPrepPipeline(AssetSpec(
        target_width=64,
        target_height=64,
        bg_mode='white',
    ))
    rep_white = pipe_white.process_asset(im)
    assert rep_white.output_image.size == (64, 64)
    assert rep_white.output_image.mode == 'RGB'
    print('Solid white mode test: OK')
    
    print('\nAll Bellium Asset Pipeline tests PASSED!')

if __name__ == '__main__':
    run_tests()
