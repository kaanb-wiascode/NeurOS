# NeurOS

NeurOS is a safety-first, non-invasive BCI + AI research prototype.

## Phase 0 goal

Build and validate the software path before purchasing EEG hardware:

`synthetic EEG -> signal decoder -> intent event -> safety gate -> AI/action adapter`

Initial SSVEP targets:
- 8 Hz -> LEFT
- 10 Hz -> RIGHT
- 12 Hz -> SELECT
- 15 Hz -> BACK

> NeurOS does not claim to read arbitrary thoughts. It classifies explicitly calibrated signal classes and exposes confidence.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
neuro-os simulate --intent SELECT
pytest
```

## Principles
1. Non-invasive first.
2. No autonomous high-impact actions from neural signals.
3. Every decoded intent carries confidence and provenance.
4. Low-confidence events are rejected.
5. Sensitive actions require explicit non-neural confirmation.
6. Raw EEG is sensitive biometric data.

See `docs/ARCHITECTURE.md`, `docs/SAFETY.md`, and `docs/ROADMAP.md`.
