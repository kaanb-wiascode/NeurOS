import json
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from time import time
from typing import Any
from uuid import uuid4

import numpy as np

from neuro_os.domain import Intent
from neuro_os.sources.base import MarkerEEGSource


_INTENT_INDEX = {
    Intent.LEFT: 1,
    Intent.RIGHT: 2,
    Intent.SELECT: 3,
    Intent.BACK: 4,
}


def start_marker_code(intent: Intent) -> float:
    if intent not in _INTENT_INDEX:
        raise ValueError(f"unsupported trial intent: {intent}")
    return float(100 + _INTENT_INDEX[intent])


def stop_marker_code(intent: Intent) -> float:
    if intent not in _INTENT_INDEX:
        raise ValueError(f"unsupported trial intent: {intent}")
    return float(200 + _INTENT_INDEX[intent])


@dataclass(frozen=True, slots=True)
class ActiveTrial:
    trial_id: str
    intent: Intent
    started_at: float
    start_marker: float


@dataclass(frozen=True, slots=True)
class TrialArtifact:
    trial_id: str
    intent: str
    started_at: float
    ended_at: float
    completion: str
    sample_rate_hz: int
    channel_names: tuple[str, ...]
    sample_count: int
    duration_seconds: float
    start_marker: float
    stop_marker: float
    start_marker_indices: tuple[int, ...]
    stop_marker_indices: tuple[int, ...]
    eeg_file: str
    metadata_file: str
    client_metadata: dict[str, Any]


class TrialRecorder:
    """Record marker-aligned local EEG trials without exposing raw EEG to the browser."""

    def __init__(
        self,
        source: MarkerEEGSource,
        storage_dir: str | Path = ".neuros/sessions",
    ) -> None:
        self.source = source
        self.storage_dir = Path(storage_dir)
        self._active: ActiveTrial | None = None
        self._lock = Lock()

    @property
    def active_trial(self) -> ActiveTrial | None:
        return self._active

    def start(self, intent: Intent) -> ActiveTrial:
        with self._lock:
            if not self.source.is_open:
                raise RuntimeError("EEG source must be open before starting a trial")
            if self._active is not None:
                raise RuntimeError("a trial is already active")

            marker = start_marker_code(intent)
            self.source.clear_buffer()
            self.source.insert_marker(marker)
            active = ActiveTrial(
                trial_id=uuid4().hex,
                intent=intent,
                started_at=time(),
                start_marker=marker,
            )
            self._active = active
            return active

    def stop(
        self,
        *,
        completion: str = "completed",
        client_metadata: dict[str, Any] | None = None,
    ) -> TrialArtifact:
        with self._lock:
            active = self._active
            if active is None:
                raise RuntimeError("no trial is active")
            if completion not in {"completed", "stopped"}:
                raise ValueError("completion must be 'completed' or 'stopped'")

            stop_marker = stop_marker_code(active.intent)
            self.source.insert_marker(stop_marker)
            settle_seconds = max(0.02, 2 / self.source.sample_rate_hz)
            frame = self.source.drain_marked(settle_seconds=settle_seconds)
            ended_at = time()

            start_indices = tuple(
                int(index)
                for index in np.flatnonzero(np.isclose(frame.markers, active.start_marker))
            )
            stop_indices = tuple(
                int(index)
                for index in np.flatnonzero(np.isclose(frame.markers, stop_marker))
            )
            if not start_indices:
                raise RuntimeError("start marker was not observed in the acquired EEG window")
            if not stop_indices:
                raise RuntimeError("stop marker was not observed in the acquired EEG window")

            self.storage_dir.mkdir(parents=True, exist_ok=True)
            eeg_path = self.storage_dir / f"{active.trial_id}.npz"
            metadata_path = self.storage_dir / f"{active.trial_id}.json"
            np.savez_compressed(
                eeg_path,
                eeg=frame.data,
                markers=frame.markers,
                sample_rate_hz=np.asarray([frame.sample_rate_hz], dtype=np.int64),
                channel_names=np.asarray(frame.channel_names, dtype="U"),
            )

            artifact = TrialArtifact(
                trial_id=active.trial_id,
                intent=active.intent.value,
                started_at=active.started_at,
                ended_at=ended_at,
                completion=completion,
                sample_rate_hz=frame.sample_rate_hz,
                channel_names=frame.channel_names,
                sample_count=frame.sample_count,
                duration_seconds=frame.duration_seconds,
                start_marker=active.start_marker,
                stop_marker=stop_marker,
                start_marker_indices=start_indices,
                stop_marker_indices=stop_indices,
                eeg_file=str(eeg_path),
                metadata_file=str(metadata_path),
                client_metadata=dict(client_metadata or {}),
            )
            metadata_path.write_text(
                json.dumps(asdict(artifact), indent=2) + "\n",
                encoding="utf-8",
            )
            self._active = None
            return artifact
