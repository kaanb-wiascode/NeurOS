from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt

from neuro_os.sources.base import EEGFrame


@dataclass(frozen=True, slots=True)
class FilterConfig:
    low_hz: float = 6.0
    high_hz: float = 45.0
    order: int = 4
    notch_hz: float | None = 50.0
    notch_quality_factor: float = 30.0


def preprocess_eeg(
    frame: EEGFrame,
    config: FilterConfig | None = None,
) -> EEGFrame:
    """Apply zero-phase band-pass and optional mains-notch filters."""
    cfg = config or FilterConfig()
    _validate_config(frame.sample_rate_hz, cfg)

    filtered = np.asarray(frame.data, dtype=float).copy()
    sos = butter(
        cfg.order,
        [cfg.low_hz, cfg.high_hz],
        btype="bandpass",
        fs=frame.sample_rate_hz,
        output="sos",
    )
    bandpass_padlen = min(_sos_padlen(sos), frame.sample_count - 1)
    filtered = sosfiltfilt(sos, filtered, axis=-1, padlen=bandpass_padlen)

    if cfg.notch_hz is not None:
        b, a = iirnotch(
            cfg.notch_hz,
            cfg.notch_quality_factor,
            fs=frame.sample_rate_hz,
        )
        notch_padlen = min(3 * max(len(a), len(b)), frame.sample_count - 1)
        filtered = filtfilt(b, a, filtered, axis=-1, padlen=notch_padlen)

    return EEGFrame(
        data=filtered,
        sample_rate_hz=frame.sample_rate_hz,
        channel_names=frame.channel_names,
        source=f"{frame.source}|filtered",
        timestamp=frame.timestamp,
    )


def _validate_config(sample_rate_hz: int, config: FilterConfig) -> None:
    nyquist_hz = sample_rate_hz / 2
    if config.order < 1:
        raise ValueError("filter order must be positive")
    if not 0 < config.low_hz < config.high_hz < nyquist_hz:
        raise ValueError(
            "band-pass frequencies must satisfy 0 < low_hz < high_hz < Nyquist"
        )
    if config.notch_quality_factor <= 0:
        raise ValueError("notch_quality_factor must be positive")
    if config.notch_hz is not None and not 0 < config.notch_hz < nyquist_hz:
        raise ValueError("notch_hz must be between 0 and Nyquist")


def _sos_padlen(sos: np.ndarray) -> int:
    sections = int(sos.shape[0])
    return 3 * (2 * sections + 1)
