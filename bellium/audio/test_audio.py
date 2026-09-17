import os, sys, math

sys.path.insert(0, os.path.abspath('temp_bellium_repo'))
from bellium.audio import detect_voice_activity, VoiceActivityDetector

def run_tests():
    sr = 16000
    # 1. Synthesize 1 sec audio: 0.3s silence + 0.4s speech-like harmonic tone (250Hz) + 0.3s silence
    samples = []
    for i in range(sr):
        t = i / sr
        if 0.3 <= t < 0.7:
            # Speech-like tone (fundamental + 2nd harmonic) with amplitude ~0.6
            val = 0.45 * math.sin(2 * math.pi * 250 * t) + 0.15 * math.sin(2 * math.pi * 500 * t)
        else:
            # Ambient background noise floor (~ -60dB)
            val = 0.001 * math.sin(2 * math.pi * 100 * t)
        samples.append(val)
        
    rep = detect_voice_activity(samples, sample_rate=sr)
    print('VAD Report on synthetic utterance:')
    print(f'  duration: {rep.duration_ms} ms, speech_ratio: {rep.speech_ratio:.1%}, SNR: {rep.snr_db_estimate:.1f} dB, quality: {rep.overall_quality}')
    assert rep.overall_quality == 'good'
    assert 0.35 <= rep.speech_ratio <= 0.45
    assert rep.is_clipped is False
    assert rep.snr_db_estimate > 20.0
    
    # 2. Test clipping detection
    clipped_samples = [1.0 if (i % 2 == 0) else -1.0 for i in range(1600)]
    rep_clip = detect_voice_activity(clipped_samples, sample_rate=sr)
    print('Clipping test:')
    print(f'  is_clipped: {rep_clip.is_clipped}, clipping_ratio: {rep_clip.clipping_ratio:.1%}, quality: {rep_clip.overall_quality}')
    assert rep_clip.is_clipped is True
    assert rep_clip.overall_quality == 'clipped'
    
    # 3. Test absolute silence
    silent_samples = [0.0] * 3200
    rep_silent = detect_voice_activity(silent_samples, sample_rate=sr)
    print('Silence test:')
    print(f'  speech_ratio: {rep_silent.speech_ratio}, quality: {rep_silent.overall_quality}')
    assert rep_silent.speech_ratio == 0.0
    assert rep_silent.overall_quality == 'silent'
    
    print('\nAll Bellium Audio VAD tests PASSED successfully!')

if __name__ == '__main__':
    run_tests()
