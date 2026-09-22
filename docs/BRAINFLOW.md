# BrainFlow Acquisition

NeurOS uses BrainFlow only as an acquisition boundary. Decoder and safety logic remain
independent from a specific EEG vendor.

## Hardware-free smoke test

BrainFlow's synthetic board uses board ID `-1` and exercises the same session lifecycle
used by physical boards.

```bash
pip install -e ".[bci,dev]"
neuro-os brainflow-smoke --board-id -1 --duration-seconds 1
```

The adapter follows this lifecycle:

1. create `BoardShim`;
2. `prepare_session()`;
3. resolve the effective board ID and EEG metadata;
4. `start_stream()`;
5. read a bounded window with `get_board_data()`;
6. `stop_stream()`;
7. `release_session()`.

## Physical hardware

No physical device is hard-coded yet. Once a supported non-invasive board is selected,
device-specific connection values (for example a serial port) will be supplied through
`BrainFlowSourceConfig` or the CLI rather than committed to source control.

Raw EEG recordings must remain local by default and must not be committed to Git.


## Marker alignment

NeurOS uses BrainFlow's marker channel to align labeled browser trials with acquired EEG.

`neuro-os brainflow-smoke --board-id -1` now:

1. starts the BrainFlow synthetic board;
2. clears stale buffered samples;
3. inserts marker `999`;
4. drains EEG and marker data together;
5. fails unless marker `999` is found on the sample-aligned marker array.

The Phase 2 trial bridge uses separate start/stop marker ranges (101–104 / 201–204).


## Board inspection

Use:

```bash
neuro-os inspect-board --board-id -1
```

The command opens the board and reports only metadata:

- effective BrainFlow board ID;
- device name;
- sampling rate;
- EEG row indices;
- EEG channel names;
- marker-channel row index.

When BrainFlow exposes fixed 10–20 electrode names, NeurOS preserves those names in
`EEGFrame.channel_names`. Boards without fixed electrode locations fall back to stable
`eeg_<row>` names so a later user/device profile can assign electrode aliases explicitly.
