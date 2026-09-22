from collections.abc import Sequence

import numpy as np

from neuro_os.sources.base import EEGFrame


def select_channels(
    frame: EEGFrame,
    channel_names: Sequence[str],
    *,
    strict: bool = True,
) -> EEGFrame:
    """Return a frame containing requested channels in the requested order."""
    requested = tuple(channel_names)
    if not requested:
        raise ValueError("at least one channel must be requested")
    if len(set(requested)) != len(requested):
        raise ValueError("channel_names must not contain duplicates")

    index_by_name = {name: index for index, name in enumerate(frame.channel_names)}
    missing = tuple(name for name in requested if name not in index_by_name)
    if missing and strict:
        raise ValueError(f"requested channels are unavailable: {', '.join(missing)}")

    selected_names = tuple(name for name in requested if name in index_by_name)
    if not selected_names:
        raise ValueError("none of the requested channels are available")

    indices = [index_by_name[name] for name in selected_names]
    return EEGFrame(
        data=np.asarray(frame.data[indices, :], dtype=float).copy(),
        sample_rate_hz=frame.sample_rate_hz,
        channel_names=selected_names,
        source=f"{frame.source}|channels",
        timestamp=frame.timestamp,
    )


def average_channels(
    frame: EEGFrame,
    channel_names: Sequence[str],
    *,
    output_name: str = "channel_average",
) -> EEGFrame:
    """Average selected channels into one virtual channel."""
    if not output_name.strip():
        raise ValueError("output_name must be non-empty")

    selected = select_channels(frame, channel_names)
    averaged = np.mean(selected.data, axis=0, keepdims=True)
    return EEGFrame(
        data=averaged,
        sample_rate_hz=frame.sample_rate_hz,
        channel_names=(output_name,),
        source=f"{frame.source}|average",
        timestamp=frame.timestamp,
    )
