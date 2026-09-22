import json
from pathlib import Path

import numpy as np
import pytest

from neuro_os.analysis import (
    analyze_session,
    analyze_trial,
    summarize_analyses,
    wolpaw_bits_per_trial,
)
from neuro_os.domain import Intent
from neuro_os.sources.synthetic import SyntheticEEGConfig, generate_ssvep


def _write_trial(
    root: Path,
    intent: Intent,
    *,
    trial_id: str,
    signal: np.ndarray | None = None,
) -> Path:
    sample_rate_hz = 250
    body = (
        signal
        if signal is not None
        else generate_ssvep(
            intent,
            SyntheticEEGConfig(
                sample_rate_hz=sample_rate_hz,
                duration_seconds=4.0,
                noise_std=0.15,
                seed=7,
            ),
        )
    )
    eeg = np.concatenate(([0.0], body, [0.0]))[np.newaxis, :]
    markers = np.zeros(eeg.shape[1], dtype=float)
    marker_index = {
        Intent.LEFT: 1,
        Intent.RIGHT: 2,
        Intent.SELECT: 3,
        Intent.BACK: 4,
    }[intent]
    markers[0] = 100 + marker_index
    markers[-1] = 200 + marker_index

    eeg_path = root / f"{trial_id}.npz"
    metadata_path = root / f"{trial_id}.json"
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
                "trial_id": trial_id,
                "intent": intent.value,
                "started_at": 1.0,
                "start_marker": float(100 + marker_index),
                "stop_marker": float(200 + marker_index),
                "eeg_file": str(eeg_path),
            }
        ),
        encoding="utf-8",
    )
    return metadata_path


@pytest.mark.parametrize(
    "intent",
    [Intent.LEFT, Intent.RIGHT, Intent.SELECT, Intent.BACK],
)
def test_analyze_trial_decodes_marker_delimited_ssvep(tmp_path: Path, intent: Intent) -> None:
    metadata_path = _write_trial(tmp_path, intent, trial_id=intent.value.lower())

    result = analyze_trial(metadata_path)

    assert result.expected_intent == intent.value
    assert result.predicted_intent == intent.value
    assert result.correct
    assert result.quality_acceptable
    assert result.confidence >= 0.55
    assert result.epoch_sample_count == 1000


def test_analyze_trial_marks_unsupported_frequency_unknown(tmp_path: Path) -> None:
    sample_rate_hz = 250
    t = np.arange(sample_rate_hz * 4) / sample_rate_hz
    unsupported = np.sin(2 * np.pi * 40.0 * t)
    metadata_path = _write_trial(
        tmp_path,
        Intent.LEFT,
        trial_id="unsupported",
        signal=unsupported,
    )

    result = analyze_trial(metadata_path)

    assert result.predicted_intent == Intent.UNKNOWN.value
    assert not result.correct


def test_analyze_session_builds_confusion_matrix(tmp_path: Path) -> None:
    for intent in (Intent.LEFT, Intent.RIGHT, Intent.SELECT, Intent.BACK):
        _write_trial(tmp_path, intent, trial_id=intent.value.lower())

    analyses, metrics = analyze_session(tmp_path)

    assert len(analyses) == 4
    assert metrics.total_trials == 4
    assert metrics.correct_trials == 4
    assert metrics.accuracy == 1.0
    assert metrics.classified_accuracy == 1.0
    assert metrics.confusion_matrix["SELECT"]["SELECT"] == 1
    assert metrics.itr_estimate_bits_per_trial == 2.0
    assert metrics.itr_estimate_bits_per_minute == pytest.approx(30.0, rel=0.02)


def test_wolpaw_itr_is_zero_at_or_below_chance() -> None:
    assert wolpaw_bits_per_trial(4, 0.25) == 0.0
    assert wolpaw_bits_per_trial(4, 0.10) == 0.0
    assert wolpaw_bits_per_trial(4, 1.0) == 2.0


def test_summarize_empty_analysis_is_safe() -> None:
    metrics = summarize_analyses(())

    assert metrics.total_trials == 0
    assert metrics.accuracy == 0.0
    assert metrics.itr_estimate_bits_per_minute == 0.0
