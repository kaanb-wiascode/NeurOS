from dataclasses import dataclass
from time import sleep, time
from typing import Any

import numpy as np

from neuro_os.sources.base import EEGFrame, MarkedEEGFrame


class BrainFlowUnavailableError(RuntimeError):
    """Raised when the optional BrainFlow dependency is not installed."""


@dataclass(frozen=True, slots=True)
class BrainFlowSourceConfig:
    board_id: int = -1
    serial_port: str | None = None
    mac_address: str | None = None
    ip_address: str | None = None
    ip_port: int | None = None
    serial_number: str | None = None
    timeout_seconds: int | None = None
    ring_buffer_size: int = 45_000


@dataclass(frozen=True, slots=True)
class BrainFlowBoardInfo:
    board_id: int
    device_name: str
    sample_rate_hz: int
    eeg_row_indices: tuple[int, ...]
    channel_names: tuple[str, ...]
    marker_channel_index: int


class BrainFlowSource:
    """Non-invasive BrainFlow acquisition adapter.

    The default board_id (-1) is BrainFlow's synthetic board, so acquisition,
    markers, and storage can be verified before physical EEG hardware is connected.
    """

    def __init__(self, config: BrainFlowSourceConfig | None = None) -> None:
        self.config = config or BrainFlowSourceConfig()
        self._board: Any | None = None
        self._sample_rate_hz: int | None = None
        self._eeg_channels: tuple[int, ...] = ()
        self._channel_names: tuple[str, ...] = ()
        self._marker_channel: int | None = None
        self._master_board_id: int | None = None
        self._device_name: str | None = None
        self._is_open = False

    @property
    def sample_rate_hz(self) -> int:
        if self._sample_rate_hz is None:
            raise RuntimeError("source must be opened before sample_rate_hz is available")
        return self._sample_rate_hz

    @property
    def channel_names(self) -> tuple[str, ...]:
        if not self._channel_names:
            raise RuntimeError("source must be opened before channel_names are available")
        return self._channel_names

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def board_info(self) -> BrainFlowBoardInfo:
        if (
            self._master_board_id is None
            or self._device_name is None
            or self._marker_channel is None
        ):
            raise RuntimeError("source must be opened before board_info is available")
        return BrainFlowBoardInfo(
            board_id=self._master_board_id,
            device_name=self._device_name,
            sample_rate_hz=self.sample_rate_hz,
            eeg_row_indices=self._eeg_channels,
            channel_names=self.channel_names,
            marker_channel_index=self._marker_channel,
        )

    def open(self) -> None:
        if self._is_open:
            return

        try:
            from brainflow.board_shim import (
                BoardShim,
                BrainFlowError,
                BrainFlowInputParams,
            )
        except ImportError as exc:
            raise BrainFlowUnavailableError(
                'BrainFlow is optional. Install it with: pip install -e ".[bci]"'
            ) from exc

        params = BrainFlowInputParams()
        if self.config.serial_port is not None:
            params.serial_port = self.config.serial_port
        if self.config.mac_address is not None:
            params.mac_address = self.config.mac_address
        if self.config.ip_address is not None:
            params.ip_address = self.config.ip_address
        if self.config.ip_port is not None:
            params.ip_port = self.config.ip_port
        if self.config.serial_number is not None:
            params.serial_number = self.config.serial_number
        if self.config.timeout_seconds is not None:
            params.timeout = self.config.timeout_seconds

        board = BoardShim(self.config.board_id, params)
        try:
            board.prepare_session()
            master_board_id = int(board.get_board_id())
            sample_rate_hz = int(BoardShim.get_sampling_rate(master_board_id))
            eeg_channels = tuple(
                int(index) for index in BoardShim.get_eeg_channels(master_board_id)
            )
            marker_channel = int(BoardShim.get_marker_channel(master_board_id))
            device_name = str(BoardShim.get_device_name(master_board_id))
            if not eeg_channels:
                raise RuntimeError(f"board {master_board_id} exposes no EEG channels")

            try:
                fixed_names = tuple(
                    str(name) for name in BoardShim.get_eeg_names(master_board_id)
                )
            except BrainFlowError:
                fixed_names = ()

            channel_names = (
                fixed_names
                if len(fixed_names) == len(eeg_channels)
                else tuple(f"eeg_{index}" for index in eeg_channels)
            )
            board.start_stream(self.config.ring_buffer_size)
        except (BrainFlowError, RuntimeError):
            try:
                board.release_session()
            finally:
                raise

        self._board = board
        self._master_board_id = master_board_id
        self._sample_rate_hz = sample_rate_hz
        self._eeg_channels = eeg_channels
        self._channel_names = channel_names
        self._marker_channel = marker_channel
        self._device_name = device_name
        self._is_open = True

    def close(self) -> None:
        board = self._board
        if board is None:
            self._reset_state()
            return

        try:
            if self._is_open:
                board.stop_stream()
        finally:
            try:
                board.release_session()
            finally:
                self._reset_state()

    def read(self, duration_seconds: float) -> EEGFrame:
        if not self._is_open or self._board is None:
            raise RuntimeError("source must be opened before read()")
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")

        requested_samples = max(1, round(duration_seconds * self.sample_rate_hz))
        sleep(duration_seconds)
        board_data = np.asarray(self._board.get_board_data(requested_samples), dtype=float)
        if board_data.ndim != 2 or board_data.shape[1] == 0:
            raise RuntimeError("BrainFlow returned no samples")

        eeg = board_data[np.asarray(self._eeg_channels), :]
        return EEGFrame(
            data=eeg,
            sample_rate_hz=self.sample_rate_hz,
            channel_names=self.channel_names,
            source=f"brainflow:{self._master_board_id}",
            timestamp=time(),
        )

    def insert_marker(self, value: float) -> None:
        if not self._is_open or self._board is None:
            raise RuntimeError("source must be opened before insert_marker()")
        if not np.isfinite(value) or value == 0:
            raise ValueError("marker value must be finite and non-zero")
        self._board.insert_marker(float(value))

    def clear_buffer(self) -> None:
        if not self._is_open or self._board is None:
            raise RuntimeError("source must be opened before clear_buffer()")
        self._board.get_board_data()

    def drain_marked(self, *, settle_seconds: float = 0.0) -> MarkedEEGFrame:
        if not self._is_open or self._board is None:
            raise RuntimeError("source must be opened before drain_marked()")
        if self._marker_channel is None:
            raise RuntimeError("marker channel is unavailable")
        if settle_seconds < 0:
            raise ValueError("settle_seconds must not be negative")
        if settle_seconds:
            sleep(settle_seconds)

        board_data = np.asarray(self._board.get_board_data(), dtype=float)
        if board_data.ndim != 2 or board_data.shape[1] == 0:
            raise RuntimeError("BrainFlow returned no samples")

        eeg = board_data[np.asarray(self._eeg_channels), :]
        markers = np.asarray(board_data[self._marker_channel, :], dtype=float)
        return MarkedEEGFrame(
            data=eeg,
            markers=markers,
            sample_rate_hz=self.sample_rate_hz,
            channel_names=self.channel_names,
            source=f"brainflow:{self._master_board_id}",
            timestamp=time(),
        )

    def _reset_state(self) -> None:
        self._board = None
        self._sample_rate_hz = None
        self._eeg_channels = ()
        self._channel_names = ()
        self._marker_channel = None
        self._master_board_id = None
        self._device_name = None
        self._is_open = False
