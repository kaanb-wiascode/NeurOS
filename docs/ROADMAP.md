# Roadmap

## Phase 0 — Simulated Brain -> Intent
- [x] Repository architecture
- [x] Synthetic SSVEP generator
- [x] FFT baseline decoder
- [x] Confidence-bearing intent event
- [x] Safety gate
- [x] Unit tests
- [ ] CI green on GitHub

## Phase 1 — Real EEG ingestion
- [ ] BrainFlow adapter behind a source interface
- [ ] Select supported non-invasive EEG hardware
- [ ] Local calibration sessions
- [ ] Filtering and artifact-quality metrics
- [ ] Never commit raw EEG data

## Phase 2 — SSVEP experiment UI
- [ ] Four-target stimulus window
- [ ] Calibration workflow
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
