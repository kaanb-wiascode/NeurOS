from dataclasses import dataclass
from enum import StrEnum
from time import time


class Intent(StrEnum):
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    SELECT = "SELECT"
    BACK = "BACK"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class DecodedIntent:
    intent: Intent
    confidence: float
    source: str
    timestamp: float

    @classmethod
    def create(cls, intent: Intent, confidence: float, source: str) -> "DecodedIntent":
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        return cls(intent=intent, confidence=confidence, source=source, timestamp=time())
