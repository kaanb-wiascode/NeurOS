from dataclasses import dataclass

import numpy as np

from neuro_os.domain import DecodedIntent, Intent
from neuro_os.sources.synthetic import TARGET_FREQUENCIES


@dataclass(frozen=True, slots=True)
class SSVEPDecoderConfig:
    sample_rate_hz: int = 250
    band_half_width_hz: float = 0.75
    minimum_confidence: float = 0.55
    minimum_target_energy_fraction: float = 0.10
    analysis_low_hz: float = 4.0
    analysis_high_hz: float = 45.0


class SSVEPDecoder:
    """FFT baseline decoder with explicit resolution and signal-quality gates."""

    def __init__(self, config: SSVEPDecoderConfig | None = None) -> None:
        self.config = config or SSVEPDecoderConfig()

    def decode(self, samples: np.ndarray) -> DecodedIntent:
        if samples.ndim != 1 or samples.size < 8:
            raise ValueError("samples must be a 1-D signal with at least 8 samples")

        fft_bin_width_hz = self.config.sample_rate_hz / samples.size
        if fft_bin_width_hz > self.config.band_half_width_hz:
            minimum_samples = int(
                np.ceil(self.config.sample_rate_hz / self.config.band_half_width_hz)
            )
            raise ValueError(
                "signal window is too short for configured SSVEP bands: "
                f"need at least {minimum_samples} samples"
            )

        centered = samples - float(np.mean(samples))
        windowed = centered * np.hanning(centered.size)
        spectrum = np.abs(np.fft.rfft(windowed)) ** 2
        freqs = np.fft.rfftfreq(windowed.size, d=1 / self.config.sample_rate_hz)

        scores: dict[Intent, float] = {}
        target_mask = np.zeros(freqs.shape, dtype=bool)
        for intent, target_hz in TARGET_FREQUENCIES.items():
            fundamental_mask = self._band_mask(freqs, target_hz)
            harmonic_mask = self._band_mask(freqs, target_hz * 2)
            fundamental = float(np.sum(spectrum[fundamental_mask]))
            harmonic = float(np.sum(spectrum[harmonic_mask]))
            scores[intent] = fundamental + (0.35 * harmonic)
            target_mask |= fundamental_mask | harmonic_mask

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_intent, best_score = ordered[0]
        candidate_power = sum(max(score, 0.0) for _, score in ordered)
        class_confidence = float(best_score / candidate_power) if candidate_power > 0 else 0.0

        analysis_mask = (freqs >= self.config.analysis_low_hz) & (
            freqs <= self.config.analysis_high_hz
        )
        analysis_power = float(np.sum(spectrum[analysis_mask]))
        target_power = float(np.sum(spectrum[target_mask & analysis_mask]))
        target_energy_fraction = target_power / analysis_power if analysis_power > 0 else 0.0

        passes_quality = target_energy_fraction >= self.config.minimum_target_energy_fraction
        passes_classification = class_confidence >= self.config.minimum_confidence
        if not passes_quality or not passes_classification:
            best_intent = Intent.UNKNOWN

        confidence = class_confidence if passes_quality else 0.0
        return DecodedIntent.create(
            intent=best_intent,
            confidence=min(max(confidence, 0.0), 1.0),
            source="ssvep_fft_v1",
        )

    def _band_mask(self, freqs: np.ndarray, target_hz: float) -> np.ndarray:
        half = self.config.band_half_width_hz
        return (freqs >= target_hz - half) & (freqs <= target_hz + half)
