from __future__ import annotations

import pytest

from mysdk import AIActionError, AIActions, AIConfig


class FakeResponsesAPI:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("FakeResponse", (), {"output_text": self.output_text})()


class FakeOpenAI:
    def __init__(self, responses_api: FakeResponsesAPI) -> None:
        self.responses = responses_api


@pytest.fixture
def openai_factory(monkeypatch):
    created_clients = []

    def install(output_text: str):
        responses_api = FakeResponsesAPI(output_text=output_text)

        def fake_openai(*, api_key=None):
            client = FakeOpenAI(responses_api=responses_api)
            created_clients.append({"api_key": api_key, "client": client})
            return client

        monkeypatch.setattr("mysdk.client.OpenAI", fake_openai)
        return responses_api, created_clients

    return install


def test_generate_text_uses_configured_model(openai_factory):
    responses_api, created_clients = openai_factory("Hello from the SDK.")
    sdk = AIActions(config=AIConfig(api_key="test-key", model="gpt-5"))

    result = sdk.generate_text("Say hello.")

    assert result == "Hello from the SDK."
    assert created_clients[0]["api_key"] == "test-key"
    assert responses_api.calls[0]["model"] == "gpt-5"
    assert responses_api.calls[0]["input"] == "Say hello."


def test_extract_keywords_returns_list(openai_factory):
    openai_factory("python, sdk, ai, openai")
    sdk = AIActions(config=AIConfig(api_key="test-key"))

    result = sdk.extract_keywords("Python SDKs simplify AI integrations.", max_keywords=3)

    assert result == ["python", "sdk", "ai"]


def test_analyze_sentiment_normalizes_output(openai_factory):
    openai_factory("Positive")
    sdk = AIActions(config=AIConfig(api_key="test-key"))

    result = sdk.analyze_sentiment("I loved the product experience.")

    assert result == "positive"


def test_analyze_sentiment_rejects_invalid_label(openai_factory):
    openai_factory("mixed feelings")
    sdk = AIActions(config=AIConfig(api_key="test-key"))

    with pytest.raises(AIActionError):
        sdk.analyze_sentiment("The experience had good and bad parts.")


def test_empty_output_raises_sdk_error(openai_factory):
    openai_factory("   ")
    sdk = AIActions(config=AIConfig(api_key="test-key"))

    with pytest.raises(AIActionError):
        sdk.summarize("Some text")
