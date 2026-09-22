# NeurOS Architecture

## Phase 0

```text
Synthetic EEG
    |
    v
SSVEP Decoder
    |
    v
DecodedIntent (intent, confidence, source, timestamp)
    |
    v
Safety Policy
    |
    +--> REJECT
    +--> REQUIRE_CONFIRMATION
    `--> ACCEPT --> future AI Cortex
```

## Boundaries
- `sources/`: synthetic first; BrainFlow hardware adapters later.
- `decoders/`: SSVEP baseline first; P300 and motor imagery remain isolated experiments.
- `intent/`: confidence gating, calibration state, debouncing, confirmation policy.
- `ai/`: future tool router consuming structured intents, not inventing intent from raw EEG.
- `storage/`: future consent-aware storage; raw EEG disabled by default.

## Non-goals
- Arbitrary thought reading.
- Medical diagnosis.
- Brain stimulation.
- Autonomous high-impact actions from EEG alone.
