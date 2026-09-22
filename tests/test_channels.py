from time import time

import numpy as np
import pytest

from neuro_os.preprocessing.channels import average_channels, select_channels
from neuro_os.sources.base import EEGFrame


def _frame() -> EEGFrame:
    return EEGFrame(
        data=np.asarray(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        ),
        sample_rate_hz=250,
        channel_names=("O1", "Oz", "O2"),
        source="test",
        timestamp=time(),
    )


def test_select_channels_preserves_requested_order() -> None:
    selected = select_channels(_frame(), ("O2", "O1"))

    assert selected.channel_names == ("O2", "O1")
    assert np.array_equal(selected.data[0], np.asarray([7.0, 8.0, 9.0]))
    assert np.array_equal(selected.data[1], np.asarray([1.0, 2.0, 3.0]))


def test_select_channels_can_ignore_missing_channels() -> None:
    selected = select_channels(_frame(), ("POz", "Oz"), strict=False)

    assert selected.channel_names == ("Oz",)


def test_select_channels_rejects_missing_in_strict_mode() -> None:
    with pytest.raises(ValueError, match="unavailable"):
        select_channels(_frame(), ("POz", "Oz"))


def test_average_channels_creates_virtual_channel() -> None:
    averaged = average_channels(_frame(), ("O1", "O2"), output_name="occipital")

    assert averaged.channel_names == ("occipital",)
    assert np.allclose(averaged.data[0], np.asarray([4.0, 5.0, 6.0]))
