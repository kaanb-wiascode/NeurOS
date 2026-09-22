# SSVEP Stimulus Prototype

This is a zero-backend browser prototype for Phase 2 calibration trials.

## Run locally

From the repository root:

```bash
python -m http.server 8080 --directory apps/ssvep-stimulus
```

Then open `http://localhost:8080`.

## What it does

- renders LEFT / RIGHT / SELECT / BACK targets;
- drives target luminance from elapsed-time phase at 8 / 10 / 12 / 15 Hz;
- lets the operator label the intended focus target for a calibration trial;
- measures observed browser animation-frame rate;
- exports session metadata as local JSON;
- makes no network/API calls after loading local assets.

## Important timing limitation

`requestAnimationFrame` is synchronized to the display refresh cycle. The requested
frequency is therefore a stimulus target, not proof of physical luminance timing.
Before research-quality EEG collection, validate actual timing on the target display
and record the monitor refresh rate.

## Safety

The page intentionally displays repetitive flicker. It includes a visible warning and
requires an explicit start action. It is not a medical or diagnostic tool.
