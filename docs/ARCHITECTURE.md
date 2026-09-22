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


## Phase 2 marker-aligned trials

```text
SSVEP browser
    |
    | POST /api/trial/start { intent }
    v
Local stimulus server
    |
    | BrainFlow insert_marker(101..104)
    v
EEG stream + marker channel
    |
    | visual trial
    v
SSVEP browser
    |
    | POST /api/trial/stop
    v
Local stimulus server
    |
    | BrainFlow insert_marker(201..204)
    | drain EEG + marker channel together
    v
TrialRecorder
    |
    +--> .neuros/sessions/<trial>.npz
    |      EEG + sample-aligned marker array
    |
    `--> .neuros/sessions/<trial>.json
           intent, timing, marker indices, browser FPS
```

The browser API never returns raw EEG. Session files are local-only and excluded from Git.
Start/stop markers are intentionally distinct so later epoch extraction can verify both
trial boundaries rather than inferring duration from browser timing alone.


## Trial analysis

```text
<id>.json + <id>.npz
        |
        v
verify start/stop markers
        |
        v
extract EEG samples strictly between markers
        |
        v
signal quality
        |
        +--> rejected -> UNKNOWN
        |
        v
band-pass + mains notch
        |
        v
channel selection / average
        |
        v
SSVEPDecoder(sample_rate_hz = recording rate)
        |
        v
TrialAnalysis
(expected, predicted, confidence, correct)
        |
        v
SessionMetrics
accuracy + classified accuracy + confusion matrix + ITR estimate
```

The decoder uses the stored recording sample rate instead of assuming 250 Hz. This is
required before supporting multiple physical EEG boards with different acquisition rates.
