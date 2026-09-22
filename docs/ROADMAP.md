# Roadmap

## Phase 0 — Simulated Brain -> Intent
- [x] Repository architecture
- [x] Synthetic SSVEP generator
- [x] FFT baseline decoder
- [x] Confidence-bearing intent event
- [x] Safety gate
- [x] Unit tests
- [x] CI green on GitHub
- [x] Reject under-resolved FFT windows
- [x] Reject target-absent signals with a spectral quality gate

## Phase 1 — Real EEG ingestion
- [x] Hardware-neutral `EEGSource` / `EEGFrame` contract
- [x] Synthetic source implementation behind the same contract
- [x] BrainFlow adapter behind the source interface
- [x] BrainFlow synthetic-board smoke command
- [x] Hardware-free BrainFlow smoke test in GitHub Actions
- [x] Keep BrainFlow optional so core CI does not require hardware libraries
- [x] Never commit raw EEG data or local calibration state
- [x] Baseline signal-quality metrics: flatline + configurable 50/60 Hz mains noise
- [x] Zero-phase band-pass preprocessing
- [x] Configurable 50/60 Hz notch filtering
- [x] Deterministic multi-channel selection and virtual channel averaging
- [x] Local descriptive calibration-profile model
- [x] Hardware-free calibration CLI example
- [ ] Select supported non-invasive EEG hardware
- [ ] Calibrate on a real user/device session
- [ ] Device-specific channel naming/profile
- [ ] Hardware-in-the-loop test on the selected device

## Phase 2 — SSVEP experiment UI
- [ ] Four-target stimulus window
- [ ] Calibration workflow UI
- [ ] Accuracy/confusion matrix/ITR metrics
- [ ] Replay of consented local recordings

## Phase 3 — AI Cortex
- [ ] Structured intent -> tool router
- [ ] Voice fallback
- [ ] Confirmation UX
- [ ] Local audit log
- [ ] Permissioned personal knowledge retrieval

## Phase 4 — Additional paradigms
- [ ] P300 speller research prototype
- [ ] Motor-imagery experiment
- [ ] Per-user decoder training
