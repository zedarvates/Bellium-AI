import os, sys
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.abspath('temp_bellium_repo'))
from bellium.cutout import extract_foreground, normalize_background, compute_mask_metrics

def run_tests():
    # Test 1: Simple red square on white background
    im = Image.new('RGB', (100, 100), (255, 255, 255))
    draw = ImageDraw.Draw(im)
    draw.rectangle([25, 25, 75, 75], fill=(255, 0, 0))
    
    res = extract_foreground(im, tolerance=20)
    m = res.metrics
    print('Test 1 - Red square on white:')
    print(f'  coverage: {m.coverage_ratio:.3f}, confidence: {m.confidence}, recommendation: {m.recommendation}')
    assert m.recommendation == 'confident'
    assert m.bbox[0] >= 24 and m.bbox[2] <= 76
    
    # Test 2: Normalize to green background
    norm = normalize_background(im, target_bg=(0, 255, 0))
    assert norm.size == (100, 100)
    # Center pixel should be red
    assert norm.getpixel((50, 50)) == (255, 0, 0)
    # Corner pixel should be green
    assert norm.getpixel((5, 5)) == (0, 255, 0)
    print('Test 2 - Normalize background: OK')
    
    # Test 3: Ambiguous noisy background should escalate / review
    ambig = Image.new('RGB', (100, 100), (120, 120, 120))
    draw_a = ImageDraw.Draw(ambig)
    draw_a.rectangle([0, 0, 99, 99], fill=(125, 125, 125)) # Touching edges with almost zero contrast
    res_a = extract_foreground(ambig, tolerance=10)
    print(f'Test 3 - Low contrast / edge bleed: recommendation={res_a.metrics.recommendation}, conf={res_a.metrics.confidence}')
    assert res_a.metrics.recommendation in ('review', 'escalate')
    
    print('\nAll Cutout tests PASSED successfully!')

if __name__ == '__main__':
    run_tests()
