from dataclasses import dataclass
from time import sleep, time
from typing import Any

import numpy as np

from neuro_os.sources.base import EEGFrame


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


class BrainFlowSource:
    """Non-invasive BrainFlow acquisition adapter.

    The default board_id (-1) is BrainFlow's synthetic board, so the acquisition
    path can be verified before physical EEG hardware is connected.
    """

    def __init__(self, config: BrainFlowSourceConfig | None = None) -> None:
        self.config = config or BrainFlowSourceConfig()
        self._board: Any | None = None
        self._sample_rate_hz: int | None = None
        self._eeg_channels: tuple[int, ...] = ()
        self._master_board_id: int | None = None
        self._is_open = False

    @property
    def sample_rate_hz(self) -> int:
        if self._sample_rate_hz is None:
            raise RuntimeError("source must be opened before sample_rate_hz is available")
        return self._sample_rate_hz

    @property
    def channel_names(self) -> tuple[str, ...]:
        if not self._eeg_channels:
            raise RuntimeError("source must be opened before channel_names are available")
        return tuple(f"eeg_{index}" for index in self._eeg_channels)

    @property
    def is_open(self) -> bool:
        return self._is_open

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
            if not eeg_channels:
                raise RuntimeError(f"board {master_board_id} exposes no EEG channels")
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

    def _reset_state(self) -> None:
        self._board = None
        self._sample_rate_hz = None
        self._eeg_channels = ()
        self._master_board_id = None
        self._is_open = False
