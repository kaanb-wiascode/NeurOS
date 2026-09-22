from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True, slots=True)
class EEGFrame:
    data: np.ndarray
    sample_rate_hz: int
    channel_names: tuple[str, ...]
    source: str
    timestamp: float

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ValueError("EEGFrame.data must have shape (channels, samples)")
        if self.data.shape[0] != len(self.channel_names):
            raise ValueError("channel_names must match the channel dimension")
        if self.data.shape[0] == 0 or self.data.shape[1] == 0:
            raise ValueError("EEGFrame cannot be empty")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if not np.all(np.isfinite(self.data)):
            raise ValueError("EEGFrame contains non-finite samples")
        if not self.source.strip():
            raise ValueError("source must be non-empty")

    @property
    def sample_count(self) -> int:
        return int(self.data.shape[1])

    @property
    def duration_seconds(self) -> float:
        return self.sample_count / self.sample_rate_hz


@dataclass(frozen=True, slots=True)
class MarkedEEGFrame(EEGFrame):
    markers: np.ndarray

    def __post_init__(self) -> None:
        EEGFrame.__post_init__(self)
        if self.markers.ndim != 1:
            raise ValueError("markers must be a 1-D sample-aligned array")
        if self.markers.size != self.sample_count:
            raise ValueError("markers must have one value per EEG sample")
        if not np.all(np.isfinite(self.markers)):
            raise ValueError("markers contain non-finite values")

    @property
    def marker_sample_indices(self) -> tuple[int, ...]:
        return tuple(int(index) for index in np.flatnonzero(self.markers))


@runtime_checkable
class EEGSource(Protocol):
    @property
    def sample_rate_hz(self) -> int: ...

    @property
    def channel_names(self) -> tuple[str, ...]: ...

    @property
    def is_open(self) -> bool: ...

    def open(self) -> None: ...

    def close(self) -> None: ...

    def read(self, duration_seconds: float) -> EEGFrame: ...


@runtime_checkable
class MarkerEEGSource(EEGSource, Protocol):
    def insert_marker(self, value: float) -> None: ...

    def clear_buffer(self) -> None: ...

    def drain_marked(self, *, settle_seconds: float = 0.0) -> MarkedEEGFrame: ...
