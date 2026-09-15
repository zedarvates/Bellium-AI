import os, sys
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.abspath('temp_bellium_repo'))
from bellium.inpaint import inpaint_patch_knn, route_inpaint_request

def run_tests():
    # 1. Routing tests
    # Empty mask -> patch_knn ratio 0.0
    m_zero = Image.new('L', (50, 50), 0)
    v0 = route_inpaint_request(m_zero)
    assert v0.method == 'patch_knn' and v0.mask_ratio == 0.0
    
    # Small hole (e.g. 5x5 in 50x50 -> 25/2500 = 1%)
    m_small = Image.new('L', (50, 50), 0)
    d_s = ImageDraw.Draw(m_small)
    d_s.rectangle([20, 20, 24, 24], fill=255)
    v_small = route_inpaint_request(m_small)
    assert v_small.method == 'patch_knn' and v_small.confidence >= 0.9
    
    # Large hole (e.g. 40x40 in 50x50 -> 1600/2500 = 64%) -> escalate
    m_large = Image.new('L', (50, 50), 0)
    d_l = ImageDraw.Draw(m_large)
    d_l.rectangle([5, 5, 45, 45], fill=255)
    v_large = route_inpaint_request(m_large)
    assert v_large.method == 'escalate_diffusion' and v_large.mask_ratio > 0.5
    print('Routing tests passed.')
    
    # 2. Inpainting synthesis test
    # Blue canvas with a red scratch hole
    base_img = Image.new('RGB', (60, 60), (0, 100, 220))
    draw = ImageDraw.Draw(base_img)
    draw.rectangle([25, 25, 34, 34], fill=(255, 0, 0)) # Red scratch to remove
    
    mask_img = Image.new('L', (60, 60), 0)
    draw_m = ImageDraw.Draw(mask_img)
    draw_m.rectangle([25, 25, 34, 34], fill=255) # Hole to fill
    
    res = inpaint_patch_knn(base_img, mask_img, patch_size=5, search_radius=15)
    assert res.metrics.filled_pixels == 100
    # The center pixel should no longer be red (255, 0, 0), it should be filled with blue (0, 100, 220)
    filled_color = res.image.getpixel((30, 30))
    print('Filled center pixel color:', filled_color)
    assert filled_color == (0, 100, 220)
    assert res.metrics.elapsed_ms >= 0
    print('Inpainting synthesis test passed.')
    print('\nAll Bellium inpaint tests PASSED!')

if __name__ == '__main__':
    run_tests()
