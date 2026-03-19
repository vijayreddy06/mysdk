from __future__ import annotations

import httpx
import pytest

from openai import AuthenticationError, RateLimitError

from mysdk import (
    AIActionError,
    AIAuthenticationError,
    AIActions,
    AIConfig,
    AIFileError,
    AIRateLimitError,
    AIResponseFormatError,
)


class FakeResponsesAPI:
    def __init__(self, outputs: list[str] | None = None, error: Exception | None = None) -> None:
        self.outputs = list(outputs or [])
        self.error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        output_text = self.outputs.pop(0) if self.outputs else ""
        return type("FakeResponse", (), {"output_text": output_text})()


class FakeTranscriptionsAPI:
    def __init__(self, text: str = "", error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return type("FakeTranscript", (), {"text": self.text})()


class FakeAudioAPI:
    def __init__(self, transcription_api: FakeTranscriptionsAPI) -> None:
        self.transcriptions = transcription_api


class FakeOpenAIClient:
    def __init__(self, responses_api: FakeResponsesAPI, transcriptions_api: FakeTranscriptionsAPI) -> None:
        self.responses = responses_api
        self.audio = FakeAudioAPI(transcriptions_api)


@pytest.fixture
def openai_factory(monkeypatch):
    created_clients = []

    def install(
        *,
        outputs: list[str] | None = None,
        response_error: Exception | None = None,
        transcript_text: str = "",
        transcription_error: Exception | None = None,
    ):
        responses_api = FakeResponsesAPI(outputs=outputs, error=response_error)
        transcriptions_api = FakeTranscriptionsAPI(text=transcript_text, error=transcription_error)

        def fake_openai(*, api_key=None, timeout=None, max_retries=None):
            client = FakeOpenAIClient(responses_api, transcriptions_api)
            created_clients.append(
                {
                    "api_key": api_key,
                    "timeout": timeout,
                    "max_retries": max_retries,
                    "client": client,
                }
            )
            return client

        monkeypatch.setattr("mysdk._runtime.OpenAI", fake_openai)
        return responses_api, transcriptions_api, created_clients

    return install


def make_httpx_response(status_code: int) -> httpx.Response:
    request = httpx.Request("POST", "https://api.openai.com/v1/responses")
    return httpx.Response(status_code, request=request)


def test_generate_text_uses_document_module_and_config(openai_factory):
    responses_api, _, created_clients = openai_factory(outputs=["Hello from the SDK."])
    sdk = AIActions(
        config=AIConfig(api_key="test-key", text_model="gpt-5", timeout=45.0, max_retries=4)
    )

    result = sdk.generate_text("Say hello.")

    assert result == "Hello from the SDK."
    assert created_clients[0]["api_key"] == "test-key"
    assert created_clients[0]["timeout"] == 45.0
    assert created_clients[0]["max_retries"] == 4
    assert responses_api.calls[0]["model"] == "gpt-5"
    assert responses_api.calls[0]["input"] == "Say hello."


def test_document_summarize_short(openai_factory):
    openai_factory(outputs=["Short summary output"])
    sdk = AIActions(config=AIConfig(api_key="test-key"))

    result = sdk.documents.summarize_short("Long document text")

    assert result == "Short summary output"


def test_extract_keywords_detailed_returns_typed_result(openai_factory):
    openai_factory(outputs=['{"keywords": ["python", "sdk"], "summary": "Important SDK terms."}'])
    sdk = AIActions(config=AIConfig(api_key="test-key"))

    result = sdk.documents.extract_keywords_detailed("Python SDKs simplify AI integrations.")

    assert result.keywords == ["python", "sdk"]
    assert result.summary == "Important SDK terms."


def test_parse_json_accepts_fenced_json_for_image_analysis(openai_factory, tmp_path):
    responses_api, _, _ = openai_factory(
        outputs=[
            '```json\n{"description": "A bar chart.", "detected_text": "Q1 Revenue", "tags": ["chart"]}\n```'
        ]
    )
    sdk = AIActions(config=AIConfig(api_key="test-key", vision_model="gpt-4.1-mini"))
    image_path = tmp_path / "chart.png"
    image_path.write_bytes(b"fakepng")

    result = sdk.images.analyze(str(image_path))

    assert result.description == "A bar chart."
    assert result.detected_text == "Q1 Revenue"
    assert result.tags == ["chart"]
    assert responses_api.calls[0]["model"] == "gpt-4.1-mini"


def test_analyze_sentiment_detailed_returns_typed_result(openai_factory):
    openai_factory(outputs=['{"label": "neutral", "confidence": 0.62, "explanation": "Mostly factual."}'])
    sdk = AIActions(config=AIConfig(api_key="test-key"))

    result = sdk.documents.analyze_sentiment_detailed("The update contains mixed feedback.")

    assert result.label == "neutral"
    assert result.confidence == 0.62
    assert result.explanation == "Mostly factual."


def test_image_analyze_builds_image_input_payload(openai_factory, tmp_path):
    responses_api, _, _ = openai_factory(
        outputs=['{"description": "A bar chart.", "detected_text": "Q1 Revenue", "tags": ["chart", "finance"]}']
    )
    sdk = AIActions(config=AIConfig(api_key="test-key", vision_model="gpt-4.1-mini"))
    image_path = tmp_path / "chart.png"
    image_path.write_bytes(b"fakepng")

    result = sdk.images.analyze(str(image_path))

    assert result.description == "A bar chart."
    assert result.detected_text == "Q1 Revenue"
    assert result.tags == ["chart", "finance"]
    payload = responses_api.calls[0]["input"]
    assert payload[0]["content"][1]["type"] == "input_image"
    assert payload[0]["content"][1]["image_url"].startswith("data:image/png;base64,")
    assert responses_api.calls[0]["model"] == "gpt-4.1-mini"


def test_image_analyze_requires_existing_file(openai_factory):
    openai_factory(outputs=['{"description": "x", "detected_text": "", "tags": []}'])
    sdk = AIActions(config=AIConfig(api_key="test-key"))

    with pytest.raises(AIFileError):
        sdk.images.describe("missing-image.png")


def test_audio_transcribe_and_summarize(openai_factory, tmp_path):
    _, transcriptions_api, _ = openai_factory(
        outputs=["Meeting summary"],
        transcript_text="Meeting transcript text",
    )
    sdk = AIActions(config=AIConfig(api_key="test-key", transcription_model="gpt-4o-mini-transcribe"))
    audio_path = tmp_path / "meeting.wav"
    audio_path.write_bytes(b"RIFF....WAVE")

    result = sdk.audio.summarize(str(audio_path))

    assert result.transcript == "Meeting transcript text"
    assert result.summary == "Meeting summary"
    assert transcriptions_api.calls[0]["model"] == "gpt-4o-mini-transcribe"
    assert transcriptions_api.calls[0]["file"].name.endswith("meeting.wav")


def test_empty_output_raises_sdk_error(openai_factory):
    openai_factory(outputs=["   "])
    sdk = AIActions(config=AIConfig(api_key="test-key"))

    with pytest.raises(AIActionError):
        sdk.summarize("Some text")


def test_invalid_json_raises_response_format_error(openai_factory):
    openai_factory(outputs=["not-json"])
    sdk = AIActions(config=AIConfig(api_key="test-key"))

    with pytest.raises(AIResponseFormatError):
        sdk.documents.extract_keywords_detailed("Python SDKs simplify AI integrations.")


def test_authentication_error_is_mapped(openai_factory):
    response = make_httpx_response(401)
    openai_factory(response_error=AuthenticationError("invalid_api_key", response=response, body=None))
    sdk = AIActions(config=AIConfig(api_key="bad-key"))

    with pytest.raises(AIAuthenticationError):
        sdk.generate_text("Hello")


def test_rate_limit_error_is_mapped(openai_factory):
    response = make_httpx_response(429)
    openai_factory(response_error=RateLimitError("quota exceeded", response=response, body=None))
    sdk = AIActions(config=AIConfig(api_key="test-key"))

    with pytest.raises(AIRateLimitError):
        sdk.generate_text("Hello")
