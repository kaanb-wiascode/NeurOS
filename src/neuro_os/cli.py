import json
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from neuro_os.calibration import collect_calibration_profile
from neuro_os.decoders.ssvep import SSVEPDecoder
from neuro_os.domain import Intent
from neuro_os.intent.safety import SafetyPolicy
from neuro_os.preprocessing.filters import preprocess_eeg
from neuro_os.preprocessing.quality import assess_signal_quality
from neuro_os.sources.brainflow import (
    BrainFlowSource,
    BrainFlowSourceConfig,
    BrainFlowUnavailableError,
)
from neuro_os.sources.synthetic import SyntheticSSVEPSource
from neuro_os.stimulus_server import serve_stimulus

app = typer.Typer(help="NeurOS non-invasive BCI research tools.")


@app.command()
def simulate(
    intent: Annotated[Intent, typer.Option(case_sensitive=False)] = Intent.SELECT,
    high_impact_action: Annotated[bool, typer.Option()] = False,
) -> None:
    """Run synthetic EEG through quality, preprocessing, decoding, and safety."""
    source = SyntheticSSVEPSource(intent)
    source.open()
    try:
        frame = source.read(2.0)
    finally:
        source.close()

    quality = assess_signal_quality(frame)
    if not quality.acceptable:
        typer.echo(
            json.dumps(
                {
                    "requested_intent": intent.value,
                    "decoded_intent": Intent.UNKNOWN.value,
                    "source": frame.source,
                    "safety_decision": "REJECT",
                    "reason": "signal-quality gate rejected the frame",
                    "rejected_channels": quality.rejected_channels,
                },
                indent=2,
            )
        )
        raise typer.Exit(code=2)

    filtered = preprocess_eeg(frame)
    event = SSVEPDecoder().decode(filtered.data[0])
    safety = SafetyPolicy().evaluate(event, high_impact_action=high_impact_action)
    typer.echo(
        json.dumps(
            {
                "requested_intent": intent.value,
                "decoded_intent": event.intent.value,
                "confidence": round(event.confidence, 4),
                "source": filtered.source,
                "quality_acceptable": quality.acceptable,
                "safety_decision": safety.decision.value,
                "reason": safety.reason,
            },
            indent=2,
        )
    )


@app.command("calibrate-synthetic")
def calibrate_synthetic(
    intent: Annotated[Intent, typer.Option(case_sensitive=False)] = Intent.SELECT,
    frame_count: Annotated[int, typer.Option(min=1, max=100)] = 5,
    duration_seconds: Annotated[float, typer.Option(min=0.5, max=30.0)] = 2.0,
    output: Annotated[Path, typer.Option()] = Path(".neuros/calibration/synthetic.json"),
) -> None:
    """Create a local calibration-profile example without EEG hardware."""
    profile = collect_calibration_profile(
        SyntheticSSVEPSource(intent),
        frame_count=frame_count,
        duration_seconds=duration_seconds,
    )
    profile.save(output)
    typer.echo(
        json.dumps(
            {
                "output": str(output),
                "sample_rate_hz": profile.sample_rate_hz,
                "channels": profile.channel_names,
                "frame_count": profile.frame_count,
                "duration_seconds": profile.duration_seconds,
            },
            indent=2,
        )
    )


@app.command("brainflow-smoke")
def brainflow_smoke(
    board_id: Annotated[int, typer.Option()] = -1,
    serial_port: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Verify BrainFlow acquisition and marker-channel alignment."""
    source = BrainFlowSource(
        BrainFlowSourceConfig(board_id=board_id, serial_port=serial_port)
    )
    marker_value = 999.0
    try:
        source.open()
        source.clear_buffer()
        source.insert_marker(marker_value)
        frame = source.drain_marked(settle_seconds=0.10)
    except BrainFlowUnavailableError as exc:
        raise typer.BadParameter(str(exc)) from exc
    finally:
        source.close()

    marker_indices = tuple(
        int(index) for index in np.flatnonzero(np.isclose(frame.markers, marker_value))
    )
    if not marker_indices:
        raise RuntimeError("BrainFlow marker smoke test did not observe the inserted marker")

    typer.echo(
        json.dumps(
            {
                "source": frame.source,
                "sample_rate_hz": frame.sample_rate_hz,
                "channels": frame.channel_names,
                "sample_count": frame.sample_count,
                "marker_value": marker_value,
                "marker_indices": marker_indices,
            },
            indent=2,
        )
    )


@app.command("serve-stimulus")
def serve_stimulus_command(
    board_id: Annotated[int, typer.Option()] = -1,
    serial_port: Annotated[str | None, typer.Option()] = None,
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option(min=1, max=65535)] = 8080,
    static_dir: Annotated[Path, typer.Option()] = Path("apps/ssvep-stimulus"),
    storage_dir: Annotated[Path, typer.Option()] = Path(".neuros/sessions"),
) -> None:
    """Serve the SSVEP UI and bridge its trials to BrainFlow markers."""
    source = BrainFlowSource(
        BrainFlowSourceConfig(board_id=board_id, serial_port=serial_port)
    )
    typer.echo(
        json.dumps(
            {
                "url": f"http://{host}:{port}",
                "board_id": board_id,
                "static_dir": str(static_dir),
                "storage_dir": str(storage_dir),
            },
            indent=2,
        )
    )
    try:
        serve_stimulus(
            source,
            host=host,
            port=port,
            static_dir=static_dir,
            storage_dir=storage_dir,
        )
    except BrainFlowUnavailableError as exc:
        raise typer.BadParameter(str(exc)) from exc


if __name__ == "__main__":
    app()
