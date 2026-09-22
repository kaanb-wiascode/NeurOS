from neuro_os.domain import DecodedIntent, Intent
from neuro_os.intent.safety import Decision, SafetyPolicy


def test_low_confidence_intent_is_rejected() -> None:
    event = DecodedIntent.create(Intent.SELECT, 0.51, "test")
    assert SafetyPolicy().evaluate(event).decision is Decision.REJECT


def test_high_impact_action_requires_confirmation() -> None:
    event = DecodedIntent.create(Intent.SELECT, 0.95, "test")
    assert (
        SafetyPolicy().evaluate(event, high_impact_action=True).decision
        is Decision.REQUIRE_CONFIRMATION
    )


def test_safe_high_confidence_intent_is_accepted() -> None:
    event = DecodedIntent.create(Intent.RIGHT, 0.92, "test")
    assert SafetyPolicy().evaluate(event).decision is Decision.ACCEPT
