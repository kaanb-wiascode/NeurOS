# SSVEP Stimulus Prototype

This browser UI is the Phase 2 visual-stimulus and trial-labeling surface.

## Integrated BrainFlow mode

From the repository root:

```bash
pip install -e ".[bci]"
neuro-os serve-stimulus --board-id -1 --port 8080
```

Then open `http://127.0.0.1:8080`.

The UI checks `/api/status`. When the local acquisition bridge is available:

- trial start sends the selected intent to `/api/trial/start`;
- the server inserts an intent-specific BrainFlow start marker;
- trial completion/stop sends display timing metadata to `/api/trial/stop`;
- the server inserts the matching stop marker;
- EEG and marker samples are drained together;
- raw EEG is stored only under `.neuros/sessions/`;
- the browser receives metadata, never raw EEG.

Marker ranges:

- LEFT: start 101, stop 201
- RIGHT: start 102, stop 202
- SELECT: start 103, stop 203
- BACK: start 104, stop 204

## Static local-only mode

```bash
python -m http.server 8080 --directory apps/ssvep-stimulus
```

In this mode the visual trial still works, but the Acquisition indicator reports
`Local-only` and no BrainFlow markers are recorded.

## Timing model

Targets are driven from elapsed-time phase at 8 / 10 / 12 / 15 Hz and rendered with
`requestAnimationFrame`.

The requested frequency is a software stimulus target, not proof of physical luminance
timing. Before research-quality EEG collection, validate actual timing on the target
display and record the monitor refresh rate.

## Safety

The page intentionally displays repetitive flicker. It includes a visible flashing-light
warning and requires an explicit start action. It is not a medical or diagnostic tool.
