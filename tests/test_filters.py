from time import time

import numpy as np
import pytest

from neuro_os.preprocessing.filters import FilterConfig, preprocess_eeg
from neuro_os.sources.base import EEGFrame


def _frame(signal: np.ndarray, sample_rate_hz: int = 250) -> EEGFrame:
    return EEGFrame(
        data=signal[np.newaxis, :],
        sample_rate_hz=sample_rate_hz,
        channel_names=("Oz",),
        source="test",
        timestamp=time(),
    )


def _amplitude_at(signal: np.ndarray, sample_rate_hz: int, frequency_hz: float) -> float:
    centered = signal - float(np.mean(signal))
    spectrum = np.abs(np.fft.rfft(centered))
    freqs = np.fft.rfftfreq(signal.size, d=1 / sample_rate_hz)
    index = int(np.argmin(np.abs(freqs - frequency_hz)))
    return float(spectrum[index])


def test_preprocess_preserves_ssvep_and_suppresses_mains() -> None:
    sample_rate_hz = 250
    duration_seconds = 4
    t = np.arange(sample_rate_hz * duration_seconds) / sample_rate_hz
    signal = np.sin(2 * np.pi * 10.0 * t) + 0.8 * np.sin(2 * np.pi * 50.0 * t)

    filtered = preprocess_eeg(_frame(signal, sample_rate_hz))

    ten_before = _amplitude_at(signal, sample_rate_hz, 10.0)
    ten_after = _amplitude_at(filtered.data[0], sample_rate_hz, 10.0)
    fifty_before = _amplitude_at(signal, sample_rate_hz, 50.0)
    fifty_after = _amplitude_at(filtered.data[0], sample_rate_hz, 50.0)

    assert ten_after > ten_before * 0.75
    assert fifty_after < fifty_before * 0.10
    assert filtered.channel_names == ("Oz",)
    assert filtered.source.endswith("|filtered")


def test_preprocess_rejects_frequency_above_nyquist() -> None:
    frame = _frame(np.zeros(500, dtype=float), sample_rate_hz=100)

    with pytest.raises(ValueError, match="Nyquist"):
        preprocess_eeg(frame, FilterConfig(high_hz=50.0))


def test_preprocess_can_disable_notch() -> None:
    sample_rate_hz = 250
    t = np.arange(500) / sample_rate_hz
    frame = _frame(np.sin(2 * np.pi * 12.0 * t), sample_rate_hz)

    filtered = preprocess_eeg(frame, FilterConfig(notch_hz=None))

    assert filtered.data.shape == frame.data.shape
    assert np.all(np.isfinite(filtered.data))
