from __future__ import annotations

from typing import Any
from bellium.contracts.schema import AuthorityMode, SpecialistResult
from bellium.magick.hashes import hamming_distance, hash_to_hex

SPECIALIST_ID = 'bellium/knn/hash-matcher:v0'

def match_hash_knn(
    query_hash: int,
    gallery: list[dict[str, Any]],
    *,
    k: int = 1,
    max_distance: int = 10,
) -> SpecialistResult:
    if not gallery:
        return SpecialistResult(
            SPECIALIST_ID,
            {'status': 'abstain', 'reason': 'empty_gallery', 'query_hash': hash_to_hex(query_hash)},
            0.0,
            True,
            AuthorityMode.CONSULTATIVE,
        )
    scored = []
    for item in gallery:
        item_hash = item.get('hash')
        if isinstance(item_hash, str):
            h_val = int(item_hash, 16)
        else:
            h_val = int(item_hash)
        dist = hamming_distance(query_hash, h_val)
        scored.append((dist, item))
    scored.sort(key=lambda x: x[0])
    top_matches = scored[:k]
    best_dist, best_item = top_matches[0]
    is_duplicate = best_dist <= 3
    is_similar = best_dist <= max_distance
    conf = max(0.0, min(1.0, 1.0 - best_dist / 32.0))
    return SpecialistResult(
        SPECIALIST_ID,
        {
            'query_hash': hash_to_hex(query_hash),
            'best_match_id': best_item.get('id'),
            'best_distance': best_dist,
            'is_duplicate': is_duplicate,
            'is_similar': is_similar,
            'matches': [
                {'id': it.get('id'), 'distance': d, 'hash': hash_to_hex(it.get('hash'))}
                for d, it in top_matches
            ],
        },
        round(conf, 4),
        False,
        AuthorityMode.CONSULTATIVE,
        notes=('k-NN perceptual hash matching for near-duplicate and asset reuse retrieval.',),
    )
