from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any


class AuthorityMode(str, Enum):
    OBSERVE = "observe"
    CONSULTATIVE = "consultative"
    SHADOW = "shadow"
    ACTIVE = "active"


@dataclass(frozen=True)
class SpecialistResult:
    specialist_id: str
    output: Any
    confidence: float | None
    abstained: bool
    authority_mode: AuthorityMode
    evidence_ref: str | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.authority_mode, AuthorityMode):
            raise ValueError("authority_mode must be an AuthorityMode")
        if not isinstance(self.abstained, bool):
            raise ValueError("abstained must be boolean")
        if self.confidence is not None:
            if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
                raise ValueError("confidence must be numeric")
            value = float(self.confidence)
            if not math.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError("confidence must be between 0 and 1")
        if self.authority_mode is AuthorityMode.ACTIVE and self.abstained:
            raise ValueError("active specialists cannot abstain silently; fail closed first")
