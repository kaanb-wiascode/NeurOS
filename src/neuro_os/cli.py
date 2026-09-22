import json

import typer

from neuro_os.decoders.ssvep import SSVEPDecoder
from neuro_os.domain import Intent
from neuro_os.intent.safety import SafetyPolicy
from neuro_os.sources.synthetic import generate_ssvep

app = typer.Typer(help="NeurOS Phase 0 command line tools.")


@app.command()
def simulate(
    intent: Intent = typer.Option(Intent.SELECT, case_sensitive=False),
    high_impact_action: bool = typer.Option(False),
) -> None:
    """Generate synthetic SSVEP, decode it, and pass the event through the safety gate."""
    signal = generate_ssvep(intent)
    event = SSVEPDecoder().decode(signal)
    safety = SafetyPolicy().evaluate(event, high_impact_action=high_impact_action)
    typer.echo(json.dumps({
        "requested_intent": intent.value,
        "decoded_intent": event.intent.value,
        "confidence": round(event.confidence, 4),
        "source": event.source,
        "safety_decision": safety.decision.value,
        "reason": safety.reason,
    }, indent=2))


if __name__ == "__main__":
    app()
