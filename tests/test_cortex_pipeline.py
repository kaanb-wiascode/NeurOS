import json
from pathlib import Path

import numpy as np

from neuro_os.ai.audit import AuditLog
from neuro_os.ai.router import ActionStatus, create_navigation_router
from neuro_os.analysis import analyze_trial
from neuro_os.domain import DecodedIntent, Intent
from neuro_os.sources.synthetic import SyntheticEEGConfig, generate_ssvep


def test_marker_aligned_eeg_trial_routes_through_cortex(tmp_path: Path) -> None:
    intent = Intent.SELECT
    sample_rate_hz = 250
    signal = generate_ssvep(
        intent,
        SyntheticEEGConfig(
            sample_rate_hz=sample_rate_hz,
            duration_seconds=4.0,
            noise_std=0.10,
            seed=23,
        ),
    )
    eeg = np.concatenate(([0.0], signal, [0.0]))[np.newaxis, :]
    markers = np.zeros(eeg.shape[1], dtype=float)
    markers[0] = 103.0
    markers[-1] = 203.0

    eeg_path = tmp_path / "trial.npz"
    metadata_path = tmp_path / "trial.json"
    np.savez_compressed(
        eeg_path,
        eeg=eeg,
        markers=markers,
        sample_rate_hz=np.asarray([sample_rate_hz], dtype=np.int64),
        channel_names=np.asarray(["Oz"], dtype="U"),
    )
    metadata_path.write_text(
        json.dumps(
            {
                "trial_id": "trial",
                "intent": intent.value,
                "started_at": 1.0,
                "start_marker": 103.0,
                "stop_marker": 203.0,
                "eeg_file": str(eeg_path),
            }
        ),
        encoding="utf-8",
    )

    analysis = analyze_trial(metadata_path)
    event = DecodedIntent.create(
        Intent(analysis.predicted_intent),
        analysis.confidence,
        "trial:test",
    )
    router = create_navigation_router(
        audit_log=AuditLog(tmp_path / "audit.jsonl"),
    )
    action = router.route(event, context={"trial_id": analysis.trial_id})

    assert analysis.predicted_intent == Intent.SELECT.value
    assert analysis.correct
    assert action.status is ActionStatus.EXECUTED
    assert action.tool_name == "ui.select"
    assert action.result["action"] == "select"
    assert (tmp_path / "audit.jsonl").exists()
