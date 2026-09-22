from dataclasses import dataclass

import numpy as np

from neuro_os.domain import Intent


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
    """Generate a deterministic SSVEP-like single-channel signal for Phase 0 tests."""
    cfg = config or SyntheticEEGConfig()
    if intent not in TARGET_FREQUENCIES:
        raise ValueError(f"Unsupported synthetic intent: {intent}")

    frequency = TARGET_FREQUENCIES[intent]
    count = int(cfg.sample_rate_hz * cfg.duration_seconds)
    t = np.arange(count, dtype=float) / cfg.sample_rate_hz
    fundamental = cfg.amplitude * np.sin(2 * np.pi * frequency * t)
    harmonic = cfg.amplitude * cfg.harmonic_ratio * np.sin(2 * np.pi * frequency * 2 * t)
    rng = np.random.default_rng(cfg.seed)
    noise = rng.normal(0.0, cfg.noise_std, size=count)
    return fundamental + harmonic + noise
