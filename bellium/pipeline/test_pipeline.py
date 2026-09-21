import os
import sys

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from bellium.pipeline import AssetPrepPipeline, AssetSpec


def make_asset(defect_colour, box):
    """A blue disc on white, plus a rectangular defect and its mask."""
    image = Image.new('RGB', (100, 100), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.ellipse([20, 20, 80, 80], fill=(20, 80, 200))
    draw.rectangle(box, fill=defect_colour)
    mask = Image.new('L', (100, 100), 0)
    ImageDraw.Draw(mask).rectangle(box, fill=255)
    return image, mask


def run_tests():
    # Test 1: a bounded defect whose known-context probes pass is repaired.
    image, defect_mask = make_asset((150, 150, 150), [48, 48, 52, 52])

    pipe = AssetPrepPipeline(AssetSpec(
        target_width=128,
        target_height=128,
        bg_mode='transparent',
        auto_inpaint_defects=True,
    ))

    report = pipe.process_asset(image, defect_mask=defect_mask)
    print('Pipeline report:')
    print(f'  verdict: {report.verdict}, conf: {report.confidence}')
    print(f'  inpaint_applied: {report.inpaint_applied}, repaired_pixels: {report.inpaint_pixels_filled}')
    for line in report.log:
        print('   -', line)

    assert report.verdict == 'ready_production'
    assert report.inpaint_applied is True
    assert report.inpaint_pixels_filled > 0
    assert report.output_image.size == (128, 128)
    assert report.output_image.mode == 'RGBA'
    # Center pixel should have been repaired and not be the defect colour
    center_pixel = report.output_image.getpixel((64, 64))
    print('Center pixel in output:', center_pixel)
    assert center_pixel[3] > 200  # Opaque foreground
    assert center_pixel[:3] != (150, 150, 150)  # Defect successfully healed

    # Test 2: a saturated defect the known-context probes cannot verify is refused.
    # The patch k-NN abstains and escalates, so the pipeline must report that instead
    # of claiming a repair. This is the safety property measured in VISUAL_QUALITY_V3/V4.
    image_big, mask_big = make_asset((255, 255, 0), [45, 45, 55, 55])
    report_big = pipe.process_asset(image_big, defect_mask=mask_big)
    print(f'Abstention report: inpaint_applied={report_big.inpaint_applied}, '
          f'repaired_pixels={report_big.inpaint_pixels_filled}')
    assert report_big.inpaint_applied is False
    assert report_big.inpaint_pixels_filled == 0
    assert any('abstained' in line for line in report_big.log)

    # Test 3: Solid white export mode
    pipe_white = AssetPrepPipeline(AssetSpec(
        target_width=64,
        target_height=64,
        bg_mode='white',
    ))
    rep_white = pipe_white.process_asset(image)
    assert rep_white.output_image.size == (64, 64)
    assert rep_white.output_image.mode == 'RGB'
    print('Solid white mode test: OK')

    print('\nAll Bellium Asset Pipeline tests PASSED!')


if __name__ == '__main__':
    run_tests()
