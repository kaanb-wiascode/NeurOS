from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

from neuro_os.sources.base import EEGFrame, EEGSource


@dataclass(frozen=True, slots=True)
class ChannelBaseline:
    channel_name: str
    rms_median: float
    standard_deviation_median: float
    peak_to_peak_median: float


@dataclass(frozen=True, slots=True)
class CalibrationProfile:
    sample_rate_hz: int
    channel_names: tuple[str, ...]
    frame_count: int
    duration_seconds: float
    channels: tuple[ChannelBaseline, ...]
    version: int = 1

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.to_json() + "\n", encoding="utf-8")


def collect_calibration_profile(
    source: EEGSource,
    *,
    frame_count: int = 5,
    duration_seconds: float = 2.0,
) -> CalibrationProfile:
    """Collect local windows from an EEGSource and build a descriptive baseline."""
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")

    frames: list[EEGFrame] = []
    source.open()
    try:
        for _ in range(frame_count):
            frames.append(source.read(duration_seconds))
    finally:
        source.close()

    return build_calibration_profile(frames)


def build_calibration_profile(frames: list[EEGFrame]) -> CalibrationProfile:
    """Build a descriptive baseline from local, consented EEG windows."""
    if not frames:
        raise ValueError("at least one EEG frame is required")

    first = frames[0]
    for frame in frames[1:]:
        if frame.sample_rate_hz != first.sample_rate_hz:
            raise ValueError("all calibration frames must use the same sample rate")
        if frame.channel_names != first.channel_names:
            raise ValueError("all calibration frames must use the same channel layout")

    channels: list[ChannelBaseline] = []
    for channel_index, channel_name in enumerate(first.channel_names):
        rms_values: list[float] = []
        std_values: list[float] = []
        ptp_values: list[float] = []
        for frame in frames:
            samples = np.asarray(frame.data[channel_index], dtype=float)
            centered = samples - float(np.mean(samples))
            rms_values.append(float(np.sqrt(np.mean(np.square(centered)))))
            std_values.append(float(np.std(centered)))
            ptp_values.append(float(np.ptp(samples)))

        channels.append(
            ChannelBaseline(
                channel_name=channel_name,
                rms_median=float(median(rms_values)),
                standard_deviation_median=float(median(std_values)),
                peak_to_peak_median=float(median(ptp_values)),
            )
        )

    return CalibrationProfile(
        sample_rate_hz=first.sample_rate_hz,
        channel_names=first.channel_names,
        frame_count=len(frames),
        duration_seconds=float(sum(frame.duration_seconds for frame in frames)),
        channels=tuple(channels),
    )


def profile_as_dict(profile: CalibrationProfile) -> dict[str, Any]:
    return asdict(profile)
