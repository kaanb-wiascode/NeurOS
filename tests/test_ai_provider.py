import numpy as np
import pytest

from neuro_os.ai.provider import CognitiveRequest, OpenAIResponsesProvider
from neuro_os.domain import Intent


class FakeResponse:
    id = "resp_test"
    output_text = "Structured answer"


class FakeResponses:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse()


class FakeClient:
    def __init__(self) -> None:
        self.responses = FakeResponses()


def test_provider_uses_responses_api_without_storage() -> None:
    client = FakeClient()
    provider = OpenAIResponsesProvider(
        model="gpt-5.6-luna",
        client=client,
    )
    request = CognitiveRequest(
        user_text="Summarize the current selection.",
        intent=Intent.SELECT,
        confidence=0.94,
        action_name="ui.select",
        context={
            "trial_id": "trial-1",
            "quality_acceptable": True,
            "selected_channels": ["Oz", "O1", "O2"],
        },
    )

    response = provider.respond(request)

    assert response.text == "Structured answer"
    assert response.response_id == "resp_test"
    call = client.responses.calls[0]
    assert call["model"] == "gpt-5.6-luna"
    assert call["store"] is False
    assert "SELECT" in str(call["input"])
    assert "trial-1" in str(call["input"])


@pytest.mark.parametrize(
    "forbidden_key",
    ["eeg", "eeg_samples", "raw_eeg", "raw_signal", "brain_signal", "markers", "waveform"],
)
def test_request_rejects_raw_neural_context_fields(forbidden_key: str) -> None:
    with pytest.raises(ValueError, match="raw neural field"):
        CognitiveRequest(
            user_text="Help me.",
            intent=Intent.SELECT,
            confidence=0.90,
            action_name="ui.select",
            context={forbidden_key: "sensitive"},
        )


def test_request_rejects_array_payloads() -> None:
    with pytest.raises(TypeError, match="unsupported context value"):
        CognitiveRequest(
            user_text="Help me.",
            intent=Intent.SELECT,
            confidence=0.90,
            action_name="ui.select",
            context={"features": np.asarray([1.0, 2.0])},
        )
