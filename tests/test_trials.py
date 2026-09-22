from time import time

import numpy as np

from neuro_os.domain import Intent
from neuro_os.sources.base import MarkedEEGFrame
from neuro_os.trials import TrialRecorder, start_marker_code, stop_marker_code


class FakeMarkerSource:
    def __init__(self) -> None:
        self._is_open = True
        self._inserted: list[float] = []

    @property
    def sample_rate_hz(self) -> int:
        return 250

    @property
    def channel_names(self) -> tuple[str, ...]:
        return ("Oz",)

    @property
    def is_open(self) -> bool:
        return self._is_open

    def open(self) -> None:
        self._is_open = True

    def close(self) -> None:
        self._is_open = False

    def read(self, duration_seconds: float):
        raise NotImplementedError

    def clear_buffer(self) -> None:
        self._inserted.clear()

    def insert_marker(self, value: float) -> None:
        self._inserted.append(value)

    def drain_marked(self, *, settle_seconds: float = 0.0) -> MarkedEEGFrame:
        markers = np.zeros(100, dtype=float)
        markers[10] = self._inserted[0]
        markers[90] = self._inserted[-1]
        return MarkedEEGFrame(
            data=np.sin(np.linspace(0, 10, 100))[np.newaxis, :],
            markers=markers,
            sample_rate_hz=self.sample_rate_hz,
            channel_names=self.channel_names,
            source="fake",
            timestamp=time(),
        )


def test_marker_codes_are_stable_and_distinct() -> None:
    assert start_marker_code(Intent.LEFT) == 101.0
    assert start_marker_code(Intent.SELECT) == 103.0
    assert stop_marker_code(Intent.LEFT) == 201.0
    assert stop_marker_code(Intent.SELECT) == 203.0


def test_trial_recorder_writes_marker_aligned_local_artifacts(tmp_path) -> None:
    source = FakeMarkerSource()
    recorder = TrialRecorder(source, storage_dir=tmp_path)

    active = recorder.start(Intent.SELECT)
    artifact = recorder.stop(
        client_metadata={"measured_display_fps": 60.0},
    )

    assert artifact.trial_id == active.trial_id
    assert artifact.intent == "SELECT"
    assert artifact.start_marker_indices == (10,)
    assert artifact.stop_marker_indices == (90,)
    assert artifact.client_metadata["measured_display_fps"] == 60.0
    assert (tmp_path / f"{active.trial_id}.npz").exists()
    assert (tmp_path / f"{active.trial_id}.json").exists()

    stored = np.load(tmp_path / f"{active.trial_id}.npz")
    assert stored["eeg"].shape == (1, 100)
    assert stored["markers"][10] == 103.0
    assert stored["markers"][90] == 203.0
