from dataclasses import dataclass
from time import time

import numpy as np

from neuro_os.domain import Intent
from neuro_os.sources.base import EEGFrame

TARGET_FREQUENCIES: dict[Intent, float] = {
    Intent.LEFT: 8.0,
    Intent.RIGHT: 10.0,
    Intent.SELECT: 12.0,
    Intent.BACK: 15.0,
}


@dataclass(frozen=True, slots=True)
class SyntheticEEGConfig:
    sample_rate_hz: int = 250
    duration_seconds: float = 2.0
    amplitude: float = 1.0
    noise_std: float = 0.35
    harmonic_ratio: float = 0.35
    seed: int = 42


def generate_ssvep(intent: Intent, config: SyntheticEEGConfig | None = None) -> np.ndarray:
    """Generate a deterministic SSVEP-like single-channel signal for tests and demos."""
    cfg = config or SyntheticEEGConfig()
    if intent not in TARGET_FREQUENCIES:
        raise ValueError(f"Unsupported synthetic intent: {intent}")
    if cfg.sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if cfg.duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")

    frequency = TARGET_FREQUENCIES[intent]
    count = int(cfg.sample_rate_hz * cfg.duration_seconds)
    if count < 1:
        raise ValueError("duration_seconds is too short for the configured sample rate")

    t = np.arange(count, dtype=float) / cfg.sample_rate_hz
    fundamental = cfg.amplitude * np.sin(2 * np.pi * frequency * t)
    harmonic = cfg.amplitude * cfg.harmonic_ratio * np.sin(2 * np.pi * frequency * 2 * t)
    rng = np.random.default_rng(cfg.seed)
    noise = rng.normal(0.0, cfg.noise_std, size=count)
    return fundamental + harmonic + noise


class SyntheticSSVEPSource:
    """EEGSource implementation for deterministic, hardware-free integration tests."""

    def __init__(
        self,
        intent: Intent = Intent.SELECT,
        *,
        sample_rate_hz: int = 250,
        amplitude: float = 1.0,
        noise_std: float = 0.35,
        harmonic_ratio: float = 0.35,
        seed: int = 42,
    ) -> None:
        self.intent = intent
        self._sample_rate_hz = sample_rate_hz
        self.amplitude = amplitude
        self.noise_std = noise_std
        self.harmonic_ratio = harmonic_ratio
        self.seed = seed
        self._is_open = False

    @property
    def sample_rate_hz(self) -> int:
        return self._sample_rate_hz

    @property
    def channel_names(self) -> tuple[str, ...]:
        return ("synthetic_ssvep",)

    @property
    def is_open(self) -> bool:
        return self._is_open

    def open(self) -> None:
        self._is_open = True

    def close(self) -> None:
        self._is_open = False

    def read(self, duration_seconds: float) -> EEGFrame:
        if not self._is_open:
            raise RuntimeError("source must be opened before read()")

        signal = generate_ssvep(
            self.intent,
            SyntheticEEGConfig(
                sample_rate_hz=self.sample_rate_hz,
                duration_seconds=duration_seconds,
                amplitude=self.amplitude,
                noise_std=self.noise_std,
                harmonic_ratio=self.harmonic_ratio,
                seed=self.seed,
            ),
        )
        return EEGFrame(
            data=signal[np.newaxis, :],
            sample_rate_hz=self.sample_rate_hz,
            channel_names=self.channel_names,
            source="synthetic_ssvep",
            timestamp=time(),
        )
