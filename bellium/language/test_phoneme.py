import os
import sys

sys.path.insert(0, os.path.abspath('temp_bellium_repo'))
from bellium.language import PhonemeMemory, EvidenceLevel

def run_tests():
    mem = PhonemeMemory()
    
    # Add attested Latin / French phonemes
    # [syllabic, high, low, back, round, tense, duration]
    # /a/ -> [1.0, 0.0, 1.0, 0.5, 0.0, 1.0, 0.5]
    # /i/ -> [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.5]
    # /u/ -> [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.5]
    mem.register_phoneme('a', 'lat', [1.0, 0.0, 1.0, 0.5, 0.0, 1.0, 0.5], EvidenceLevel.ATTESTED, 'classical-corpus')
    mem.register_phoneme('i', 'lat', [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.5], EvidenceLevel.ATTESTED, 'classical-corpus')
    mem.register_phoneme('u', 'lat', [1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.5], EvidenceLevel.ATTESTED, 'classical-corpus')
    
    # Add reconstructed Proto-Indo-European laryngeal
    mem.register_phoneme('h2', 'ine-pro', [0.0, 0.2, 0.8, 0.9, 0.0, 0.5, 0.7], EvidenceLevel.RECONSTRUCTED, 'comparative-method')
    
    assert mem.count() == 4
    print('Registration count test: OK')
    
    # Query nearest to an open low central vowel close to /a/
    q_a = [1.0, 0.05, 0.95, 0.5, 0.0, 0.95, 0.5]
    matches = mem.find_nearest_neighbors(q_a, k=1)
    assert len(matches) == 1
    top = matches[0]
    print(f'Query /a/: top={top.record.symbol}, dist={top.distance}, conf={top.confidence}')
    assert top.record.symbol == 'a'
    assert top.confidence > 0.95
    
    # Reconstruct test
    sym, ev, conf = mem.reconstruct_phoneme(q_a, language_filter='lat')
    print(f'Reconstruct /a/: sym={sym}, evidence={ev}, conf={conf}')
    assert sym == 'a'
    assert ev in (EvidenceLevel.ATTESTED, EvidenceLevel.INFERRED)
    
    # Filtering by evidence test
    # Filter to only reconstructed
    rec_matches = mem.find_nearest_neighbors(q_a, k=2, min_evidence=EvidenceLevel.RECONSTRUCTED)
    # All returned must be ATTESTED or RECONSTRUCTED
    for m in rec_matches:
        assert m.record.evidence in (EvidenceLevel.ATTESTED, EvidenceLevel.RECONSTRUCTED)
    print('Evidence filtering test: OK')
    
    print('\nAll Bellium Language / Phoneme k-NN tests PASSED!')

if __name__ == '__main__':
    run_tests()
