"""Provenance-grounded Phoneme k-NN and reconstruction engine.

Every phoneme record explicitly tracks its epistemic status:
- ATTESTED: directly documented in primary linguistic corpus or native recording
- RECONSTRUCTED: historically reconstructed via comparative method (e.g. Proto-Indo-European, Old French)
- INFERRED: deduced by deterministic rule or high-confidence nearest exemplars
- SPECULATIVE: hypothesis under test or low-confidence extrapolation

Zero external dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import List, Dict, Optional, Tuple


class EvidenceLevel(str, Enum):
    ATTESTED = "attested"
    RECONSTRUCTED = "reconstructed"
    INFERRED = "inferred"
    SPECULATIVE = "speculative"


@dataclass(frozen=True)
class PhonemicVector:
    # Articulatory feature vector (values normalized in [0.0, 1.0])
    # Consonant features: [sonorant, consonantal, voice, nasal, continuant, place_front_back, place_height]
    # Vowel features: [syllabic, high, low, back, round, tense, duration]
    features: Tuple[float, ...]

    def distance(self, other: PhonemicVector) -> float:
        if len(self.features) != len(other.features):
            raise ValueError("Mismatched feature dimensions")
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(self.features, other.features)))


def phonetic_distance(v1: PhonemicVector, v2: PhonemicVector) -> float:
    return v1.distance(v2)


@dataclass
class PhonemeRecord:
    symbol: str  # IPA symbol or grapheme representation
    language_code: str  # ISO 639 code or family tag (e.g. 'fra', 'lat', 'ine-pro')
    vector: PhonemicVector
    evidence: EvidenceLevel
    provenance_source: str
    notes: str = ""


@dataclass
class RetrievalMatch:
    record: PhonemeRecord
    distance: float
    confidence: float


class PhonemeMemory:
    """Family-isolated and provenance-aware k-NN memory for phonetic exemplars."""
    
    def __init__(self):
        self._records: List[PhonemeRecord] = []
        
    def register_phoneme(
        self,
        symbol: str,
        language_code: str,
        features: Tuple[float, ...] | List[float],
        evidence: EvidenceLevel | str,
        provenance_source: str,
        notes: str = "",
    ) -> PhonemeRecord:
        if isinstance(evidence, str):
            evidence = EvidenceLevel(evidence.lower())
        rec = PhonemeRecord(
            symbol=symbol,
            language_code=language_code,
            vector=PhonemicVector(tuple(float(x) for x in features)),
            evidence=evidence,
            provenance_source=provenance_source,
            notes=notes,
        )
        self._records.append(rec)
        return rec
        
    def count(self) -> int:
        return len(self._records)
        
    def find_nearest_neighbors(
        self,
        query_vector: PhonemicVector | Tuple[float, ...],
        k: int = 3,
        *,
        min_evidence: Optional[EvidenceLevel] = None,
        language_filter: Optional[str] = None,
    ) -> List[RetrievalMatch]:
        if isinstance(query_vector, (tuple, list)):
            query_vector = PhonemicVector(tuple(float(x) for x in query_vector))
            
        candidates = self._records
        if language_filter:
            candidates = [r for r in candidates if r.language_code == language_filter]
            
        if min_evidence:
            # Rank hierarchy
            hierarchy = {
                EvidenceLevel.ATTESTED: 3,
                EvidenceLevel.RECONSTRUCTED: 2,
                EvidenceLevel.INFERRED: 1,
                EvidenceLevel.SPECULATIVE: 0,
            }
            min_rank = hierarchy[min_evidence]
            candidates = [r for r in candidates if hierarchy[r.evidence] >= min_rank]
            
        if not candidates:
            return []
            
        scored = []
        for r in candidates:
            d = r.vector.distance(query_vector)
            # Confidence decays with distance
            conf = max(0.0, min(1.0, 1.0 - (d / 2.0)))
            scored.append(RetrievalMatch(record=r, distance=round(d, 4), confidence=round(conf, 4)))
            
        scored.sort(key=lambda m: m.distance)
        return scored[:k]
        
    def reconstruct_phoneme(
        self,
        query_vector: PhonemicVector | Tuple[float, ...],
        *,
        k: int = 3,
    ) -> Tuple[str, EvidenceLevel, float]:
        """Infer most probable phoneme and its epistemic status based on neighbor evidence."""
        neighbors = self.find_nearest_neighbors(query_vector, k=k)
        if not neighbors:
            return ("?\恢", EvidenceLevel.SPECULATIVE, 0.0)
            
        top = neighbors[0]
        if top.distance < 0.05 and top.record.evidence == EvidenceLevel.ATTESTED:
            return (top.record.symbol, EvidenceLevel.ATTESTED, top.confidence)
        elif top.confidence > 0.85:
            return (top.record.symbol, EvidenceLevel.INFERRED, top.confidence)
        else:
            return (top.record.symbol, EvidenceLevel.SPECULATIVE, top.confidence)
