# AI Cortex

The AI Cortex is the boundary between probabilistic BCI output and software actions.

## Input contract

The Cortex accepts `DecodedIntent` only:

```text
intent
confidence
source
timestamp
```

It does not accept raw EEG.

## Routing model

```text
DecodedIntent
     |
     v
registered ToolDefinition
     |
     v
SafetyPolicy
  /       \
reject   accept
          |
          +-- LOW risk --> execute
          |
          `-- HIGH risk --> PENDING_CONFIRMATION
                                |
                       explicit non-neural input
                       keyboard / touch / voice
                                |
                                v
                              execute
```

An EEG event can propose a high-impact action but can never confirm it. The confirmation
API explicitly rejects `eeg` as a confirmation channel.

## Initial tools

The first tool set is intentionally low impact:

- LEFT -> `ui.navigate_left`
- RIGHT -> `ui.navigate_right`
- SELECT -> `ui.select`
- BACK -> `ui.back`

These return structured actions only; they do not yet control the operating system.

## Audit

Every route, confirmation, cancellation, execution, and rejection is written to the local
append-only JSONL audit log:

```text
.neuros/audit/events.jsonl
```

The audit schema includes intent, confidence, selected tool, status, reason, risk, and
action ID. Raw EEG is never included.

## End-to-end trial command

```bash
neuro-os cortex-trial .neuros/sessions/<trial-id>.json
```

This performs:

```text
marker-aligned EEG
   -> trial analysis
   -> DecodedIntent
   -> SafetyPolicy
   -> CortexRouter
   -> low-risk UI action or rejection
   -> local audit log
```

External AI providers and permissioned personal knowledge retrieval are deliberately
separate future adapters. They must receive structured, permission-scoped requests rather
than raw neural data.


## OpenAI Responses provider

OpenAI support is optional:

```bash
pip install -e ".[ai]"
export OPENAI_API_KEY="..."
```

The provider uses the Responses API and defaults to `gpt-5.6-luna`. The model is
configurable from the CLI.

Run:

```bash
neuro-os ai-cortex .neuros/sessions/<trial-id>.json --prompt "Summarize my current task."
```

The API request is made only after:

1. marker-aligned trial analysis succeeds;
2. the decoder emits a supported intent;
3. confidence passes `SafetyPolicy`;
4. the Cortex routes an executable low-risk action.

The provider sends only:

- explicit user text;
- decoded intent;
- confidence;
- routed action name;
- small structured metadata such as trial ID and selected channel names.

The provider rejects raw neural context keys such as `eeg`, `raw_eeg`, `markers`,
`raw_signal`, and waveform payloads. NumPy arrays and nested bulk sequences are not
accepted at this boundary.

The Responses request explicitly sets `store=False`. API keys are read through the
OpenAI SDK/environment and are never written into NeurOS source or session files.
