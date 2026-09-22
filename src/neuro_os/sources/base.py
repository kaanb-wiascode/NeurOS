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
