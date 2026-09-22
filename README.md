# NeurOS

NeurOS is a safety-first, non-invasive BCI + AI research prototype.

## Current goal

Build the software path before purchasing EEG hardware:

`EEG source -> signal decoder -> structured intent -> safety gate -> AI/action adapter`

The project does **not** claim to read arbitrary thoughts. It classifies explicitly
calibrated signal classes and exposes confidence and provenance.

## Phase 0 — simulated intent pipeline

Initial SSVEP targets:

- 8 Hz -> LEFT
- 10 Hz -> RIGHT
- 12 Hz -> SELECT
- 15 Hz -> BACK

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
neuro-os simulate --intent SELECT
pytest
```

## Phase 1 — EEG acquisition abstraction

NeurOS now has a source contract (`EEGSource` / `EEGFrame`) and an optional BrainFlow
adapter. BrainFlow is not required for the core test suite.

To exercise BrainFlow without physical hardware, use its synthetic board:

```bash
pip install -e ".[bci,dev]"
neuro-os brainflow-smoke --board-id -1 --duration-seconds 1
```

Physical board configuration will be added only after a supported non-invasive device is
selected and locally calibrated.

## Principles

1. Non-invasive first.
2. No brain stimulation or implant workflow.
3. No autonomous high-impact actions from neural signals.
4. Every decoded intent carries confidence and provenance.
5. Low-confidence or poor-quality events are rejected.
6. Sensitive actions require explicit non-neural confirmation.
7. Raw EEG is treated as sensitive biometric data and is excluded from Git by default.

See `docs/ARCHITECTURE.md`, `docs/BRAINFLOW.md`, `docs/SAFETY.md`, and `docs/ROADMAP.md`.
