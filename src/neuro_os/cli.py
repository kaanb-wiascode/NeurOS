import json
from typing import Annotated

import typer

from neuro_os.decoders.ssvep import SSVEPDecoder
from neuro_os.domain import Intent
from neuro_os.intent.safety import SafetyPolicy
from neuro_os.sources.brainflow import (
    BrainFlowSource,
    BrainFlowSourceConfig,
    BrainFlowUnavailableError,
)
from neuro_os.sources.synthetic import SyntheticSSVEPSource

app = typer.Typer(help="NeurOS non-invasive BCI research tools.")


@app.command()
def simulate(
    intent: Annotated[Intent, typer.Option(case_sensitive=False)] = Intent.SELECT,
    high_impact_action: Annotated[bool, typer.Option()] = False,
) -> None:
    """Run the synthetic EEG -> decoder -> safety-gate pipeline."""
    source = SyntheticSSVEPSource(intent)
    source.open()
    try:
        frame = source.read(2.0)
    finally:
        source.close()

    event = SSVEPDecoder().decode(frame.data[0])
    safety = SafetyPolicy().evaluate(event, high_impact_action=high_impact_action)
    typer.echo(
        json.dumps(
            {
                "requested_intent": intent.value,
                "decoded_intent": event.intent.value,
                "confidence": round(event.confidence, 4),
                "source": frame.source,
                "safety_decision": safety.decision.value,
                "reason": safety.reason,
            },
            indent=2,
        )
    )


@app.command("brainflow-smoke")
def brainflow_smoke(
    duration_seconds: Annotated[float, typer.Option(min=0.1, max=10.0)] = 1.0,
    board_id: Annotated[int, typer.Option()] = -1,
    serial_port: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Verify BrainFlow acquisition; board -1 uses BrainFlow's synthetic board."""
    source = BrainFlowSource(
        BrainFlowSourceConfig(board_id=board_id, serial_port=serial_port)
    )
    try:
        source.open()
        frame = source.read(duration_seconds)
    except BrainFlowUnavailableError as exc:
        raise typer.BadParameter(str(exc)) from exc
    finally:
        source.close()

    typer.echo(
        json.dumps(
            {
                "source": frame.source,
                "sample_rate_hz": frame.sample_rate_hz,
                "channels": frame.channel_names,
                "sample_count": frame.sample_count,
                "duration_seconds": round(frame.duration_seconds, 4),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    app()
