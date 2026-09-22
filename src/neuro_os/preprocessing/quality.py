from dataclasses import dataclass

import numpy as np

from neuro_os.sources.base import EEGFrame


@dataclass(frozen=True, slots=True)
class SignalQualityConfig:
    mains_hz: float = 50.0
    mains_band_half_width_hz: float = 1.0
    flatline_std_threshold: float = 1e-8
    max_mains_power_fraction: float = 0.35
    analysis_low_hz: float = 1.0
    analysis_high_hz: float = 100.0


@dataclass(frozen=True, slots=True)
class ChannelQuality:
    channel_name: str
    rms: float
    standard_deviation: float
    peak_to_peak: float
    mains_power_fraction: float
    is_flatline: bool
    is_mains_noise_dominated: bool

    @property
    def acceptable(self) -> bool:
        return not self.is_flatline and not self.is_mains_noise_dominated


@dataclass(frozen=True, slots=True)
class SignalQualityReport:
    channels: tuple[ChannelQuality, ...]

    @property
    def acceptable(self) -> bool:
        return bool(self.channels) and all(channel.acceptable for channel in self.channels)

    @property
    def rejected_channels(self) -> tuple[str, ...]:
        return tuple(
            channel.channel_name for channel in self.channels if not channel.acceptable
        )


def assess_signal_quality(
    frame: EEGFrame,
    config: SignalQualityConfig | None = None,
) -> SignalQualityReport:
    cfg = config or SignalQualityConfig()
    if cfg.mains_hz <= 0:
        raise ValueError("mains_hz must be positive")
    if cfg.mains_band_half_width_hz <= 0:
        raise ValueError("mains_band_half_width_hz must be positive")
    if not 0.0 <= cfg.max_mains_power_fraction <= 1.0:
        raise ValueError("max_mains_power_fraction must be between 0 and 1")

    reports: list[ChannelQuality] = []
    for name, samples in zip(frame.channel_names, frame.data, strict=True):
        centered = samples - float(np.mean(samples))
        standard_deviation = float(np.std(centered))
        rms = float(np.sqrt(np.mean(np.square(centered))))
        peak_to_peak = float(np.ptp(samples))
        mains_fraction = _mains_power_fraction(
            centered,
            sample_rate_hz=frame.sample_rate_hz,
            config=cfg,
        )
        reports.append(
            ChannelQuality(
                channel_name=name,
                rms=rms,
                standard_deviation=standard_deviation,
                peak_to_peak=peak_to_peak,
                mains_power_fraction=mains_fraction,
                is_flatline=standard_deviation <= cfg.flatline_std_threshold,
                is_mains_noise_dominated=mains_fraction >= cfg.max_mains_power_fraction,
            )
        )

    return SignalQualityReport(channels=tuple(reports))


def _mains_power_fraction(
    samples: np.ndarray,
    *,
    sample_rate_hz: int,
    config: SignalQualityConfig,
) -> float:
    if samples.size < 2:
        return 0.0

    windowed = samples * np.hanning(samples.size)
    spectrum = np.abs(np.fft.rfft(windowed)) ** 2
    freqs = np.fft.rfftfreq(samples.size, d=1 / sample_rate_hz)

    nyquist_hz = sample_rate_hz / 2
    analysis_high_hz = min(config.analysis_high_hz, nyquist_hz)
    analysis_mask = (freqs >= config.analysis_low_hz) & (freqs <= analysis_high_hz)
    analysis_power = float(np.sum(spectrum[analysis_mask]))
    if analysis_power <= 0:
        return 0.0

    mains_mask = (
        (freqs >= config.mains_hz - config.mains_band_half_width_hz)
        & (freqs <= config.mains_hz + config.mains_band_half_width_hz)
        & analysis_mask
    )
    mains_power = float(np.sum(spectrum[mains_mask]))
    return min(max(mains_power / analysis_power, 0.0), 1.0)
