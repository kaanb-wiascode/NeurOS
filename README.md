# NeurOS

NeurOS is a safety-first, non-invasive BCI + AI research prototype.

## Current pipeline

`EEG source -> signal quality -> preprocessing -> decoder -> structured intent -> safety gate -> AI/action adapter`

The project does **not** claim to read arbitrary thoughts. It classifies explicitly
calibrated signal classes and exposes confidence and provenance.

## Synthetic end-to-end demo

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
neuro-os simulate --intent SELECT
pytest
```

The simulation now exercises signal-quality checks, zero-phase filtering, decoding, and
the safety gate.

## Local calibration profile demo

```bash
neuro-os calibrate-synthetic --intent SELECT --frame-count 5
```

Profiles are written under `.neuros/` by default and are excluded from Git. They contain
descriptive baseline statistics rather than claims about thoughts or mental state.

## BrainFlow acquisition

```bash
pip install -e ".[bci,dev]"
neuro-os brainflow-smoke --board-id -1 --duration-seconds 1
```

Board `-1` uses BrainFlow's synthetic board, allowing the acquisition lifecycle to be
tested without physical EEG hardware.

## Phase 2 browser stimulus

```bash
python -m http.server 8080 --directory apps/ssvep-stimulus
```

Open `http://localhost:8080` to run the four-target SSVEP calibration prototype.

The page intentionally produces visual flicker and includes a visible flashing-light
warning. Actual monitor timing must be validated before research-quality EEG collection.

## Principles

1. Non-invasive first.
2. No brain stimulation or implant workflow.
3. No autonomous high-impact actions from neural signals.
4. Every decoded intent carries confidence and provenance.
5. Poor-quality, unsupported, or low-confidence events are rejected.
6. Sensitive actions require explicit non-neural confirmation.
7. Raw EEG and local calibration state remain local by default.

See `docs/ARCHITECTURE.md`, `docs/BRAINFLOW.md`, `docs/SAFETY.md`,
`docs/ROADMAP.md`, and `apps/ssvep-stimulus/README.md`.
