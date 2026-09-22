# NeurOS Architecture

## Phase 0

```text
Synthetic SSVEP
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

## Phase 1

```text
                    EEGSource
                       |
          +------------+-------------+
          |                          |
          v                          v
SyntheticSSVEPSource          BrainFlowSource
                                      |
                           +----------+----------+
                           |                     |
                     board_id = -1         physical EEG
                     synthetic board      future device
                           |                     |
                           +----------+----------+
                                      |
                                      v
                                  EEGFrame
                         (channels x samples)
                                      |
                                      v
                              Signal Quality
                         flatline / mains noise
                                      |
                         +------------+------------+
                         |                         |
                      reject                    accept
                                                   |
                                                   v
                                            preprocessing
                                                [next]
                                                   |
                                                   v
                                            SSVEP Decoder
                                                   |
                                                   v
                                            Intent + Safety
```

## Core boundaries

- `sources/base.py`: hardware-neutral acquisition contract and validated `EEGFrame`.
- `sources/synthetic.py`: deterministic SSVEP source for tests and demos.
- `sources/brainflow.py`: optional BrainFlow adapter; BrainFlow is imported lazily.
- `preprocessing/quality.py`: channel-level flatline and mains-noise quality assessment.
- `decoders/`: probabilistic signal classification with explicit quality gates.
- `intent/`: confidence gating, debouncing, calibration state, and confirmation policy.
- `ai/`: future tool router consuming structured intents rather than raw EEG.
- `storage/`: future consent-aware storage; raw EEG disabled by default.

## Decoder safety properties

The FFT baseline rejects:

1. windows whose frequency resolution is too coarse for configured SSVEP bands;
2. signals with insufficient energy in supported target/harmonic bands;
3. ambiguous classifications below the configured confidence threshold.

The acquisition-quality layer separately flags:

1. flatline channels;
2. mains-noise-dominated channels at configurable 50/60 Hz;
3. per-channel RMS, standard deviation, and peak-to-peak metrics for later calibration.

## Non-goals

- Arbitrary thought reading.
- Medical diagnosis.
- Brain stimulation.
- Surgical or implant instructions.
- Autonomous high-impact actions from EEG alone.
