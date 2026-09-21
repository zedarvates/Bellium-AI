import os
import sys

sys.path.insert(0, os.path.abspath('temp_bellium_repo'))
from bellium.adapters import repair_and_validate_json, SchemaRule

def run_tests():
    # Define strict schema for a bounding box / classification prediction
    schema = {
        'label': SchemaRule(expected_type=str, required=True, allowed_values={'player', 'npc', 'prop'}),
        'confidence': SchemaRule(expected_type=float, required=True, min_value=0.0, max_value=1.0),
        'count': SchemaRule(expected_type=int, required=False, default=1, min_value=0),
        'is_active': SchemaRule(expected_type=bool, required=True),
    }
    
    # Test 1: Markdown backtick fence + trailing comma + type coercion ('0.92' -> float, 'yes' -> True)
    dirty_payload = '''```json
    {
        "label": "npc",
        "confidence": "0.95",
        "is_active": "true",
        "unwanted_extra": 42,
    }
    ```'''
    
    rep1 = repair_and_validate_json(dirty_payload, schema, allow_extra_keys=False)
    print('Report 1:')
    print(f'  valid: {rep1.is_valid}, repaired: {rep1.repaired_syntax}, data: {rep1.data}')
    print(f'  coerced: {rep1.coerced_fields}, dropped: {rep1.dropped_fields}')
    assert rep1.is_valid is True
    assert rep1.repaired_syntax is True
    assert rep1.data['label'] == 'npc'
    assert rep1.data['confidence'] == 0.95
    assert rep1.data['is_active'] is True
    assert rep1.data['count'] == 1  # Assigned default
    assert 'unwanted_extra' not in rep1.data
    
    # Test 2: Out of bounds clamping (confidence = 1.45 -> clamped to 1.0)
    payload_clamp = {
        'label': 'player',
        'confidence': 1.45,
        'is_active': True,
    }
    rep2 = repair_and_validate_json(payload_clamp, schema)
    print('Report 2:')
    print(f'  confidence: {rep2.data["confidence"]}, clamped: {rep2.clamped_fields}')
    assert rep2.is_valid is True
    assert rep2.data['confidence'] == 1.0
    assert len(rep2.clamped_fields) == 1
    
    # Test 3: Invalid label value -> validation error
    payload_invalid = {
        'label': 'alien_boss',
        'confidence': 0.8,
        'is_active': True,
    }
    rep3 = repair_and_validate_json(payload_invalid, schema)
    print('Report 3:')
    print(f'  valid: {rep3.is_valid}, errors: {rep3.errors}')
    assert rep3.is_valid is False
    assert any('not in allowed set' in err for err in rep3.errors)
    
    print('\nAll Bellium JSON Adapter tests PASSED successfully!')

if __name__ == '__main__':
    run_tests()
