import dataclasses
import json
import math
import typing

from neuro_os.domain import Intent


class OpenAIUnavailableError(RuntimeError):
    """Raised when the optional OpenAI SDK is unavailable."""


Scalar = str | int | float | bool | None
ContextValue = Scalar | tuple[Scalar, ...] | list[Scalar]


@dataclasses.dataclass(frozen=True, slots=True)
class CognitiveRequest:
    user_text: str
    intent: Intent
    confidence: float
    action_name: str
    context: dict[str, ContextValue]

    def __post_init__(self) -> None:
        if not self.user_text.strip():
            raise ValueError("user_text must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if not self.action_name.strip():
            raise ValueError("action_name must be non-empty")
        _validate_context(self.context)


@dataclasses.dataclass(frozen=True, slots=True)
class CognitiveResponse:
    model: str
    response_id: str | None
    text: str


class ResponsesClient(typing.Protocol):
    responses: typing.Any


class OpenAIResponsesProvider:
    """Text-only AI provider receiving structured Cortex context, never raw EEG."""

    def __init__(
        self,
        *,
        model: str = "gpt-5.6-luna",
        client: ResponsesClient | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must be non-empty")
        self.model = model
        self._client = client or self._create_client()

    def respond(self, request: CognitiveRequest) -> CognitiveResponse:
        payload = {
            "intent": request.intent.value,
            "confidence": round(request.confidence, 6),
            "action_name": request.action_name,
            "context": request.context,
        }
        response = self._client.responses.create(
            model=self.model,
            instructions=(
                "You are the text-only cognitive assistant inside NeurOS. "
                "Treat the structured BCI context as probabilistic metadata, not as "
                "a verbatim thought. Never claim to read the user's mind. Answer the "
                "explicit user text. Do not infer hidden mental or medical states."
            ),
            input=(
                request.user_text.strip()
                + "\n\nStructured NeurOS context:\n"
                + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            ),
            store=False,
        )
        output_text = str(getattr(response, "output_text", "")).strip()
        if not output_text:
            raise RuntimeError("OpenAI response did not contain output_text")
        response_id_value = getattr(response, "id", None)
        response_id = str(response_id_value) if response_id_value is not None else None
        return CognitiveResponse(
            model=self.model,
            response_id=response_id,
            text=output_text,
        )

    @staticmethod
    def _create_client() -> ResponsesClient:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise OpenAIUnavailableError(
                'OpenAI SDK is optional. Install it with: pip install -e ".[ai]"'
            ) from exc
        return OpenAI()


def _validate_context(context: dict[str, ContextValue]) -> None:
    forbidden_keys = {
        "eeg",
        "eeg_samples",
        "raw_eeg",
        "raw_signal",
        "brain_signal",
        "markers",
        "waveform",
    }
    for key, value in context.items():
        if not isinstance(key, str) or not key.strip():
            raise TypeError("context keys must be non-empty strings")
        if key.strip().lower() in forbidden_keys:
            raise ValueError(f"raw neural field is forbidden in AI context: {key}")
        _validate_context_value(value, key=key)


def _validate_context_value(value: ContextValue, *, key: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"context value must be finite: {key}")
        return
    if isinstance(value, (tuple, list)):
        if len(value) > 32:
            raise ValueError(f"context sequence is too large: {key}")
        for item in value:
            if isinstance(item, (tuple, list, dict)):
                raise TypeError(f"nested context sequences are not allowed: {key}")
            _validate_context_value(item, key=key)
        return
    raise TypeError(f"unsupported context value type for {key}: {type(value).__name__}")
