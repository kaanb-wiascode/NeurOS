import dataclasses
import json
import math
import pathlib
import time
import typing

import numpy as np

from neuro_os.decoders.ssvep import SSVEPDecoder, SSVEPDecoderConfig
from neuro_os.domain import Intent
from neuro_os.preprocessing.channels import average_channels, select_channels
from neuro_os.preprocessing.filters import FilterConfig, preprocess_eeg
from neuro_os.preprocessing.quality import SignalQualityConfig, assess_signal_quality
from neuro_os.sources.base import EEGFrame


@dataclasses.dataclass(frozen=True, slots=True)
class TrialAnalysis:
    trial_id: str
    expected_intent: str
    predicted_intent: str
    confidence: float
    correct: bool
    quality_acceptable: bool
    rejected_channels: tuple[str, ...]
    selected_channels: tuple[str, ...]
    sample_rate_hz: int
    epoch_start_index: int
    epoch_stop_index: int
    epoch_sample_count: int
    epoch_duration_seconds: float
    decoder_source: str


@dataclasses.dataclass(frozen=True, slots=True)
class SessionMetrics:
    total_trials: int
    classified_trials: int
    rejected_or_unknown_trials: int
    correct_trials: int
    accuracy: float
    classified_accuracy: float
    average_selection_seconds: float
    itr_estimate_bits_per_trial: float
    itr_estimate_bits_per_minute: float
    confusion_matrix: dict[str, dict[str, int]]


def analyze_trial(
    metadata_path: str | pathlib.Path,
    *,
    channel_names: tuple[str, ...] | None = None,
    mains_hz: float = 50.0,
) -> TrialAnalysis:
    metadata_file = pathlib.Path(metadata_path)
    metadata = _load_metadata(metadata_file)
    eeg_file = _resolve_eeg_file(metadata_file, str(metadata["eeg_file"]))

    with np.load(eeg_file, allow_pickle=False) as stored:
        eeg = np.asarray(stored["eeg"], dtype=float)
        markers = np.asarray(stored["markers"], dtype=float)
        sample_rate_hz = int(np.asarray(stored["sample_rate_hz"]).reshape(-1)[0])
        stored_channel_names = tuple(str(value) for value in stored["channel_names"].tolist())

    if eeg.ndim != 2:
        raise ValueError("stored EEG must have shape (channels, samples)")
    if markers.ndim != 1 or markers.size != eeg.shape[1]:
        raise ValueError("stored marker array must align with EEG samples")
    if len(stored_channel_names) != eeg.shape[0]:
        raise ValueError("stored channel names do not match EEG channel count")

    expected = Intent(str(metadata["intent"]))
    start_marker = float(metadata["start_marker"])
    stop_marker = float(metadata["stop_marker"])
    start_index, stop_index = _epoch_bounds(markers, start_marker, stop_marker)

    epoch = eeg[:, start_index + 1 : stop_index]
    if epoch.shape[1] < 8:
        raise ValueError("marker-delimited EEG epoch is too short")

    frame = EEGFrame(
        data=epoch,
        sample_rate_hz=sample_rate_hz,
        channel_names=stored_channel_names,
        source=f"trial:{metadata['trial_id']}",
        timestamp=float(metadata.get("started_at", time.time())),
    )

    selected = (
        select_channels(frame, channel_names)
        if channel_names is not None
        else frame
    )
    nyquist_hz = selected.sample_rate_hz / 2
    quality = assess_signal_quality(
        selected,
        SignalQualityConfig(
            mains_hz=mains_hz,
            analysis_high_hz=min(100.0, max(2.0, nyquist_hz - 1.0)),
        ),
    )

    if not quality.acceptable:
        return TrialAnalysis(
            trial_id=str(metadata["trial_id"]),
            expected_intent=expected.value,
            predicted_intent=Intent.UNKNOWN.value,
            confidence=0.0,
            correct=False,
            quality_acceptable=False,
            rejected_channels=quality.rejected_channels,
            selected_channels=selected.channel_names,
            sample_rate_hz=sample_rate_hz,
            epoch_start_index=start_index,
            epoch_stop_index=stop_index,
            epoch_sample_count=selected.sample_count,
            epoch_duration_seconds=selected.duration_seconds,
            decoder_source="quality_gate",
        )

    high_hz = min(45.0, nyquist_hz - 1.0)
    if high_hz <= 6.0:
        raise ValueError("sample rate is too low for the configured SSVEP preprocessing band")

    filtered = preprocess_eeg(
        selected,
        FilterConfig(
            low_hz=6.0,
            high_hz=high_hz,
            notch_hz=mains_hz if mains_hz < nyquist_hz else None,
        ),
    )
    decoding_frame = (
        average_channels(
            filtered,
            filtered.channel_names,
            output_name="ssvep_average",
        )
        if len(filtered.channel_names) > 1
        else filtered
    )

    event = SSVEPDecoder(
        SSVEPDecoderConfig(sample_rate_hz=sample_rate_hz)
    ).decode(decoding_frame.data[0])

    return TrialAnalysis(
        trial_id=str(metadata["trial_id"]),
        expected_intent=expected.value,
        predicted_intent=event.intent.value,
        confidence=event.confidence,
        correct=event.intent is expected,
        quality_acceptable=True,
        rejected_channels=(),
        selected_channels=selected.channel_names,
        sample_rate_hz=sample_rate_hz,
        epoch_start_index=start_index,
        epoch_stop_index=stop_index,
        epoch_sample_count=decoding_frame.sample_count,
        epoch_duration_seconds=decoding_frame.duration_seconds,
        decoder_source=event.source,
    )


def analyze_session(
    storage_dir: str | pathlib.Path,
    *,
    channel_names: tuple[str, ...] | None = None,
    mains_hz: float = 50.0,
) -> tuple[tuple[TrialAnalysis, ...], SessionMetrics]:
    directory = pathlib.Path(storage_dir)
    analyses: list[TrialAnalysis] = []

    for metadata_path in sorted(directory.glob("*.json")):
        if metadata_path.name.endswith(".analysis.json") or metadata_path.name == "summary.json":
            continue
        try:
            metadata = _load_metadata(metadata_path)
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            continue
        if not {"trial_id", "intent", "eeg_file", "start_marker", "stop_marker"} <= metadata.keys():
            continue
        analyses.append(
            analyze_trial(
                metadata_path,
                channel_names=channel_names,
                mains_hz=mains_hz,
            )
        )

    metrics = summarize_analyses(tuple(analyses))
    return tuple(analyses), metrics


def summarize_analyses(analyses: tuple[TrialAnalysis, ...]) -> SessionMetrics:
    total = len(analyses)
    classified = sum(item.predicted_intent != Intent.UNKNOWN.value for item in analyses)
    correct = sum(item.correct for item in analyses)
    accuracy = correct / total if total else 0.0
    classified_accuracy = correct / classified if classified else 0.0
    average_seconds = (
        sum(item.epoch_duration_seconds for item in analyses) / total
        if total
        else 0.0
    )

    labels = tuple(intent.value for intent in Intent if intent is not Intent.UNKNOWN)
    predicted_labels = labels + (Intent.UNKNOWN.value,)
    matrix = {
        expected: {predicted: 0 for predicted in predicted_labels}
        for expected in labels
    }
    for item in analyses:
        if item.expected_intent in matrix and item.predicted_intent in matrix[item.expected_intent]:
            matrix[item.expected_intent][item.predicted_intent] += 1

    bits_per_trial = wolpaw_bits_per_trial(len(labels), accuracy)
    bits_per_minute = (
        bits_per_trial * 60.0 / average_seconds
        if average_seconds > 0
        else 0.0
    )
    return SessionMetrics(
        total_trials=total,
        classified_trials=classified,
        rejected_or_unknown_trials=total - classified,
        correct_trials=correct,
        accuracy=accuracy,
        classified_accuracy=classified_accuracy,
        average_selection_seconds=average_seconds,
        itr_estimate_bits_per_trial=bits_per_trial,
        itr_estimate_bits_per_minute=bits_per_minute,
        confusion_matrix=matrix,
    )


def wolpaw_bits_per_trial(number_of_classes: int, accuracy: float) -> float:
    if number_of_classes < 2:
        raise ValueError("number_of_classes must be at least 2")
    if not 0.0 <= accuracy <= 1.0:
        raise ValueError("accuracy must be between 0 and 1")
    chance = 1.0 / number_of_classes
    if accuracy <= chance:
        return 0.0
    if accuracy == 1.0:
        return math.log2(number_of_classes)

    error = 1.0 - accuracy
    return (
        math.log2(number_of_classes)
        + accuracy * math.log2(accuracy)
        + error * math.log2(error / (number_of_classes - 1))
    )


def write_trial_analysis(
    analysis: TrialAnalysis,
    destination: str | pathlib.Path,
) -> None:
    pathlib.Path(destination).write_text(
        json.dumps(dataclasses.asdict(analysis), indent=2) + "\n",
        encoding="utf-8",
    )


def write_session_metrics(
    metrics: SessionMetrics,
    destination: str | pathlib.Path,
) -> None:
    pathlib.Path(destination).write_text(
        json.dumps(dataclasses.asdict(metrics), indent=2) + "\n",
        encoding="utf-8",
    )


def _epoch_bounds(
    markers: np.ndarray,
    start_marker: float,
    stop_marker: float,
) -> tuple[int, int]:
    start_candidates = np.flatnonzero(np.isclose(markers, start_marker))
    if start_candidates.size == 0:
        raise ValueError("start marker is missing from the stored trial")
    start_index = int(start_candidates[0])

    stop_candidates = np.flatnonzero(
        np.isclose(markers, stop_marker)
        & (np.arange(markers.size) > start_index)
    )
    if stop_candidates.size == 0:
        raise ValueError("stop marker after start marker is missing from the stored trial")
    return start_index, int(stop_candidates[0])


def _load_metadata(path: pathlib.Path) -> dict[str, typing.Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("trial metadata must be a JSON object")
    return value


def _resolve_eeg_file(metadata_path: pathlib.Path, eeg_file: str) -> pathlib.Path:
    configured = pathlib.Path(eeg_file)
    if configured.exists():
        return configured

    sibling = metadata_path.parent / configured.name
    if sibling.exists():
        return sibling
    raise FileNotFoundError(f"trial EEG file does not exist: {configured}")
