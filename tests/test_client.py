from __future__ import annotations

import httpx
import pytest
from openai import AuthenticationError, RateLimitError

from mysdk import (
    AIActionError,
    AIAuthenticationError,
    AIActions,
    AIConfig,
    AIRateLimitError,
    AIResponseFormatError,
)


class FakeResponsesAPI:
    def __init__(self, output_text: str | None = None, error: Exception | None = None) -> None:
        self.output_text = output_text
        self.error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return type("FakeResponse", (), {"output_text": self.output_text})()


class FakeOpenAI:
    def __init__(self, responses_api: FakeResponsesAPI) -> None:
        self.responses = responses_api


@pytest.fixture
def openai_factory(monkeypatch):
    created_clients = []

    def install(output_text: str | None = None, error: Exception | None = None):
        responses_api = FakeResponsesAPI(output_text=output_text, error=error)

        def fake_openai(*, api_key=None, timeout=None, max_retries=None):
            client = FakeOpenAI(responses_api=responses_api)
            created_clients.append(
                {
                    "api_key": api_key,
                    "timeout": timeout,
                    "max_retries": max_retries,
                    "client": client,
                }
            )
            return client

        monkeypatch.setattr("mysdk.client.OpenAI", fake_openai)
        return responses_api, created_clients

    return install


def make_httpx_response(status_code: int) -> httpx.Response:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    return httpx.Response(status_code, request=request)


def test_generate_text_uses_configured_model(openai_factory):
    responses_api, created_clients = openai_factory("Hello from the SDK.")
    sdk = AIActions(
        config=AIConfig(api_key="test-key", model="gpt-5", timeout=45.0, max_retries=4)
    )

    result = sdk.generate_text("Say hello.")

    assert result == "Hello from the SDK."
    assert created_clients[0]["api_key"] == "test-key"
    assert created_clients[0]["timeout"] == 45.0
    assert created_clients[0]["max_retries"] == 4
    assert responses_api.calls[0]["model"] == "gpt-5"
    assert responses_api.calls[0]["input"] == "Say hello."


def test_extract_keywords_returns_list(openai_factory):
    openai_factory('{"keywords": ["python", "sdk", "ai", "openai"], "summary": "Useful SDK terms."}')
    sdk = AIActions(config=AIConfig(api_key="test-key"))

    result = sdk.extract_keywords("Python SDKs simplify AI integrations.", max_keywords=3)

    assert result == ["python", "sdk", "ai"]


def test_extract_keywords_detailed_returns_typed_result(openai_factory):
    openai_factory('{"keywords": ["python", "sdk"], "summary": "Important Python SDK terms."}')
    sdk = AIActions(config=AIConfig(api_key="test-key"))

    result = sdk.extract_keywords_detailed("Python SDKs simplify AI integrations.")

    assert result.keywords == ["python", "sdk"]
    assert result.summary == "Important Python SDK terms."


def test_analyze_sentiment_normalizes_output(openai_factory):
    openai_factory('{"label": "Positive", "confidence": 0.98, "explanation": "Strong positive language."}')
    sdk = AIActions(config=AIConfig(api_key="test-key"))

    result = sdk.analyze_sentiment("I loved the product experience.")

    assert result == "positive"


def test_analyze_sentiment_detailed_returns_typed_result(openai_factory):
    openai_factory(
        '{"label": "neutral", "confidence": 0.62, "explanation": "The wording is mostly factual."}'
    )
    sdk = AIActions(config=AIConfig(api_key="test-key"))

    result = sdk.analyze_sentiment_detailed("The update contains mixed feedback.")

    assert result.label == "neutral"
    assert result.confidence == 0.62
    assert result.explanation == "The wording is mostly factual."


def test_empty_output_raises_sdk_error(openai_factory):
    openai_factory("   ")
    sdk = AIActions(config=AIConfig(api_key="test-key"))

    with pytest.raises(AIActionError):
        sdk.summarize("Some text")


def test_invalid_sentiment_payload_raises_response_format_error(openai_factory):
    openai_factory('{"label": "mixed", "confidence": 0.50, "explanation": "Not allowed."}')
    sdk = AIActions(config=AIConfig(api_key="test-key"))

    with pytest.raises(AIResponseFormatError):
        sdk.analyze_sentiment("The experience had good and bad parts.")


def test_invalid_json_raises_response_format_error(openai_factory):
    openai_factory("not-json")
    sdk = AIActions(config=AIConfig(api_key="test-key"))

    with pytest.raises(AIResponseFormatError):
        sdk.extract_keywords_detailed("Python SDKs simplify AI integrations.")


def test_authentication_error_is_mapped(openai_factory):
    response = make_httpx_response(401)
    openai_factory(error=AuthenticationError("invalid_api_key", response=response, body=None))
    sdk = AIActions(config=AIConfig(api_key="bad-key"))

    with pytest.raises(AIAuthenticationError):
        sdk.generate_text("Hello")


def test_rate_limit_error_is_mapped(openai_factory):
    response = make_httpx_response(429)
    openai_factory(error=RateLimitError("quota exceeded", response=response, body=None))
    sdk = AIActions(config=AIConfig(api_key="test-key"))

    with pytest.raises(AIRateLimitError):
        sdk.generate_text("Hello")
