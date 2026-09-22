from dataclasses import dataclass

import numpy as np

from neuro_os.domain import DecodedIntent, Intent
from neuro_os.sources.synthetic import TARGET_FREQUENCIES


@dataclass(frozen=True, slots=True)
class SSVEPDecoderConfig:
    sample_rate_hz: int = 250
    band_half_width_hz: float = 0.75
    minimum_confidence: float = 0.55


class SSVEPDecoder:
    """Small FFT-based baseline decoder for the Phase 0 synthetic pipeline."""

    def __init__(self, config: SSVEPDecoderConfig | None = None) -> None:
        self.config = config or SSVEPDecoderConfig()

    def decode(self, samples: np.ndarray) -> DecodedIntent:
        if samples.ndim != 1 or samples.size < 8:
            raise ValueError("samples must be a 1-D signal with at least 8 samples")

        centered = samples - float(np.mean(samples))
        windowed = centered * np.hanning(centered.size)
        spectrum = np.abs(np.fft.rfft(windowed)) ** 2
        freqs = np.fft.rfftfreq(windowed.size, d=1 / self.config.sample_rate_hz)

        scores: dict[Intent, float] = {}
        for intent, target_hz in TARGET_FREQUENCIES.items():
            fundamental = self._band_power(freqs, spectrum, target_hz)
            harmonic = self._band_power(freqs, spectrum, target_hz * 2)
            scores[intent] = fundamental + (0.35 * harmonic)

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_intent, best_score = ordered[0]
        total = sum(max(score, 0.0) for _, score in ordered) or 1.0
        confidence = float(best_score / total)

        if confidence < self.config.minimum_confidence:
            best_intent = Intent.UNKNOWN

        return DecodedIntent.create(
            intent=best_intent,
            confidence=min(max(confidence, 0.0), 1.0),
            source="synthetic_ssvep_fft",
        )

    def _band_power(self, freqs: np.ndarray, spectrum: np.ndarray, target_hz: float) -> float:
        half = self.config.band_half_width_hz
        mask = (freqs >= target_hz - half) & (freqs <= target_hz + half)
        if not np.any(mask):
            return 0.0
        return float(np.sum(spectrum[mask]))
