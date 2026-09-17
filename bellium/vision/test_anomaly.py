import os, sys
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.abspath('temp_bellium_repo'))
from bellium.vision import detect_visual_anomalies, VisualAnomalyDetector

def run_tests():
    # Test 1: Normal sharp checkerboard frame
    im_normal = Image.new('RGB', (100, 100), (200, 200, 200))
    draw = ImageDraw.Draw(im_normal)
    for y in range(0, 100, 20):
        for x in range(0, 100, 20):
            if (x // 20 + y // 20) % 2 == 0:
                draw.rectangle([x, y, x + 19, y + 19], fill=(50, 50, 50))
                
    rep_normal = detect_visual_anomalies(im_normal)
    print('Normal frame report:')
    print(f'  is_anomalous: {rep_normal.is_anomalous}, severity: {rep_normal.severity}, blur: {rep_normal.blur_score}')
    assert rep_normal.is_anomalous is False
    assert rep_normal.severity == 'normal'
    
    # Test 2: Blackout / lens cap -> critical_emergency
    im_blackout = Image.new('RGB', (100, 100), (5, 5, 5))
    rep_blackout = detect_visual_anomalies(im_blackout)
    print('Blackout frame report:')
    print(f'  severity: {rep_blackout.severity}, reasons: {rep_blackout.reasons}')
    assert rep_blackout.is_anomalous is True
    assert rep_blackout.severity == 'critical_emergency'
    assert any('blackout' in r.lower() for r in rep_blackout.reasons)
    
    # Test 3: Major sudden occlusion against reference frame
    im_occluded = im_normal.copy()
    draw_occ = ImageDraw.Draw(im_occluded)
    draw_occ.rectangle([10, 10, 80, 80], fill=(255, 0, 0)) # 70x70 = 4900/10000 = 49% occlusion
    
    detector = VisualAnomalyDetector()
    rep_occ = detector.inspect_frame(im_occluded, reference_frame=im_normal)
    print('Occluded frame report:')
    print(f'  severity: {rep_occ.severity}, occlusion_ratio: {rep_occ.occlusion_ratio}')
    assert rep_occ.is_anomalous is True
    assert rep_occ.occlusion_ratio >= 0.40
    assert any('occlusion' in r.lower() for r in rep_occ.reasons)
    
    print('\nAll Bellium Vision Anomaly tests PASSED!')

if __name__ == '__main__':
    run_tests()
