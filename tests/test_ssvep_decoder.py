import numpy as np
import pytest

from neuro_os.decoders.ssvep import SSVEPDecoder
from neuro_os.domain import Intent
from neuro_os.sources.synthetic import SyntheticEEGConfig, generate_ssvep


@pytest.mark.parametrize("intent", [Intent.LEFT, Intent.RIGHT, Intent.SELECT, Intent.BACK])
def test_decoder_recognizes_synthetic_target(intent: Intent) -> None:
    decoded = SSVEPDecoder().decode(generate_ssvep(intent))
    assert decoded.intent is intent
    assert decoded.confidence >= 0.55


def test_decoder_rejects_under_resolved_window() -> None:
    short_signal = generate_ssvep(
        Intent.LEFT,
        SyntheticEEGConfig(duration_seconds=0.1),
    )

    with pytest.raises(ValueError, match="too short"):
        SSVEPDecoder().decode(short_signal)


def test_decoder_marks_out_of_class_signal_unknown() -> None:
    sample_rate_hz = 250
    duration_seconds = 2.0
    t = np.arange(int(sample_rate_hz * duration_seconds)) / sample_rate_hz
    unsupported_signal = np.sin(2 * np.pi * 40.0 * t)

    decoded = SSVEPDecoder().decode(unsupported_signal)

    assert decoded.intent is Intent.UNKNOWN
    assert decoded.confidence == 0.0
