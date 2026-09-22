from dataclasses import dataclass
from enum import StrEnum

from neuro_os.domain import DecodedIntent, Intent


class Decision(StrEnum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    REQUIRE_CONFIRMATION = "REQUIRE_CONFIRMATION"


@dataclass(frozen=True, slots=True)
class SafetyDecision:
    decision: Decision
    reason: str


@dataclass(frozen=True, slots=True)
class SafetyPolicy:
    minimum_confidence: float = 0.70

    def evaluate(self, event: DecodedIntent, *, high_impact_action: bool = False) -> SafetyDecision:
        if event.intent is Intent.UNKNOWN:
            return SafetyDecision(Decision.REJECT, "decoder returned UNKNOWN")
        if event.confidence < self.minimum_confidence:
            return SafetyDecision(
                Decision.REJECT,
                f"confidence {event.confidence:.3f} below {self.minimum_confidence:.3f}",
            )
        if high_impact_action:
            return SafetyDecision(
                Decision.REQUIRE_CONFIRMATION,
                "high-impact action requires explicit non-neural confirmation",
            )
        return SafetyDecision(Decision.ACCEPT, "intent passed confidence and safety gates")
