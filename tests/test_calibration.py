from time import time

import numpy as np
import pytest

from neuro_os.calibration import build_calibration_profile
from neuro_os.sources.base import EEGFrame


def _frame(scale: float = 1.0, sample_rate_hz: int = 250) -> EEGFrame:
    t = np.arange(500) / sample_rate_hz
    o1 = scale * np.sin(2 * np.pi * 10.0 * t)
    oz = scale * np.sin(2 * np.pi * 12.0 * t)
    return EEGFrame(
        data=np.vstack((o1, oz)),
        sample_rate_hz=sample_rate_hz,
        channel_names=("O1", "Oz"),
        source="test",
        timestamp=time(),
    )


def test_build_calibration_profile_uses_multiple_frames() -> None:
    profile = build_calibration_profile([_frame(0.8), _frame(1.0), _frame(1.2)])

    assert profile.sample_rate_hz == 250
    assert profile.channel_names == ("O1", "Oz")
    assert profile.frame_count == 3
    assert profile.duration_seconds == 6.0
    assert len(profile.channels) == 2
    assert profile.channels[0].rms_median > 0
    assert '"frame_count": 3' in profile.to_json()


def test_build_calibration_profile_rejects_layout_mismatch() -> None:
    first = _frame()
    second = EEGFrame(
        data=first.data[:1],
        sample_rate_hz=250,
        channel_names=("O1",),
        source="test",
        timestamp=time(),
    )

    with pytest.raises(ValueError, match="channel layout"):
        build_calibration_profile([first, second])


def test_build_calibration_profile_rejects_sample_rate_mismatch() -> None:
    with pytest.raises(ValueError, match="sample rate"):
        build_calibration_profile([_frame(sample_rate_hz=250), _frame(sample_rate_hz=200)])
