from time import time

import numpy as np

from neuro_os.preprocessing.quality import SignalQualityConfig, assess_signal_quality
from neuro_os.sources.base import EEGFrame


def _frame(signal: np.ndarray, sample_rate_hz: int = 250) -> EEGFrame:
    return EEGFrame(
        data=signal[np.newaxis, :],
        sample_rate_hz=sample_rate_hz,
        channel_names=("Oz",),
        source="test",
        timestamp=time(),
    )


def test_quality_accepts_clean_non_mains_signal() -> None:
    sample_rate_hz = 250
    t = np.arange(sample_rate_hz * 2) / sample_rate_hz
    signal = np.sin(2 * np.pi * 10.0 * t)

    report = assess_signal_quality(_frame(signal))

    assert report.acceptable
    assert report.rejected_channels == ()
    assert report.channels[0].mains_power_fraction < 0.05


def test_quality_rejects_flatline() -> None:
    signal = np.ones(500, dtype=float)

    report = assess_signal_quality(_frame(signal))

    assert not report.acceptable
    assert report.channels[0].is_flatline
    assert report.rejected_channels == ("Oz",)


def test_quality_rejects_mains_dominated_signal() -> None:
    sample_rate_hz = 250
    t = np.arange(sample_rate_hz * 2) / sample_rate_hz
    signal = np.sin(2 * np.pi * 50.0 * t)

    report = assess_signal_quality(_frame(signal))

    assert not report.acceptable
    assert report.channels[0].is_mains_noise_dominated
    assert report.channels[0].mains_power_fraction > 0.90


def test_quality_supports_configurable_60_hz_mains() -> None:
    sample_rate_hz = 250
    t = np.arange(sample_rate_hz * 2) / sample_rate_hz
    signal = np.sin(2 * np.pi * 60.0 * t)

    report = assess_signal_quality(
        _frame(signal),
        SignalQualityConfig(mains_hz=60.0),
    )

    assert not report.acceptable
    assert report.channels[0].is_mains_noise_dominated
