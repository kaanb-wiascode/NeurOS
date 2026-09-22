from time import time

import numpy as np
import pytest

from neuro_os.domain import Intent
from neuro_os.sources.base import EEGFrame, EEGSource
from neuro_os.sources.synthetic import SyntheticSSVEPSource


def test_eeg_frame_exposes_duration() -> None:
    frame = EEGFrame(
        data=np.zeros((2, 500), dtype=float),
        sample_rate_hz=250,
        channel_names=("C3", "C4"),
        source="test",
        timestamp=time(),
    )

    assert frame.sample_count == 500
    assert frame.duration_seconds == 2.0


def test_eeg_frame_rejects_channel_mismatch() -> None:
    with pytest.raises(ValueError, match="channel_names"):
        EEGFrame(
            data=np.zeros((2, 100), dtype=float),
            sample_rate_hz=250,
            channel_names=("C3",),
            source="test",
            timestamp=time(),
        )


def test_synthetic_source_implements_source_contract() -> None:
    source = SyntheticSSVEPSource(Intent.RIGHT)

    assert isinstance(source, EEGSource)
    source.open()
    frame = source.read(2.0)
    source.close()

    assert frame.data.shape == (1, 500)
    assert frame.sample_rate_hz == 250
    assert frame.channel_names == ("synthetic_ssvep",)
    assert frame.source == "synthetic_ssvep"
    assert not source.is_open


def test_synthetic_source_requires_open() -> None:
    source = SyntheticSSVEPSource(Intent.SELECT)

    with pytest.raises(RuntimeError, match="opened"):
        source.read(2.0)
