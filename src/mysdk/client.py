from __future__ import annotations

import json
import os
from dataclasses import dataclass

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

from .errors import (
    AIActionError,
    AIAuthenticationError,
    AIConnectionError,
    AIRateLimitError,
    AIResponseFormatError,
    AITimeoutError,
)


@dataclass(slots=True)
class AIConfig:
    api_key: str | None = None
    model: str = "gpt-5"
    timeout: float = 30.0
    max_retries: int = 2


@dataclass(slots=True)
class SentimentResult:
    label: str
    confidence: float
    explanation: str


@dataclass(slots=True)
class KeywordExtractionResult:
    keywords: list[str]
    summary: str


class AIActions:
    """Intermediate SDK actions built on top of OpenAI's Responses API."""

    def __init__(self, config: AIConfig | None = None) -> None:
        self.config = config or AIConfig()
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise AIAuthenticationError(
                "OpenAI API key is missing. Set OPENAI_API_KEY or pass api_key in AIConfig."
            )
        self._client = OpenAI(
            api_key=api_key,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
        )

    def generate_text(self, prompt: str, system_prompt: str | None = None) -> str:
        instructions = system_prompt or "You are a helpful assistant for application developers."
        return self._run_action(prompt=prompt, instructions=instructions)

    def summarize(self, text: str, style: str = "brief") -> str:
        prompt = f"Summarize the following text in a {style} way:\n\n{text}"
        instructions = "Create a clear summary that preserves the key meaning."
        return self._run_action(prompt=prompt, instructions=instructions)

    def extract_keywords(self, text: str, max_keywords: int = 5) -> list[str]:
        return self.extract_keywords_detailed(text=text, max_keywords=max_keywords).keywords

    def extract_keywords_detailed(self, text: str, max_keywords: int = 5) -> KeywordExtractionResult:
        prompt = (
            f"Extract up to {max_keywords} important keywords from this text.\n"
            "Return JSON with exactly these keys: keywords, summary.\n"
            "keywords must be an array of short strings. summary must be one short sentence.\n\n"
            f"{text}"
        )
        raw_output = self._run_action(
            prompt=prompt,
            instructions="Return valid JSON only with the requested schema.",
        )
        payload = self._parse_json(raw_output)
        keywords = payload.get("keywords")
        summary = payload.get("summary")
        if not isinstance(keywords, list) or not isinstance(summary, str):
            raise AIResponseFormatError("Keyword extraction response is missing required fields.")

        cleaned_keywords = []
        for item in keywords:
            if isinstance(item, str):
                cleaned = item.strip()
                if cleaned and cleaned not in cleaned_keywords:
                    cleaned_keywords.append(cleaned)
            if len(cleaned_keywords) >= max_keywords:
                break

        if not cleaned_keywords:
            raise AIResponseFormatError("Keyword extraction returned no usable keywords.")

        return KeywordExtractionResult(keywords=cleaned_keywords, summary=summary.strip())

    def analyze_sentiment(self, text: str) -> str:
        return self.analyze_sentiment_detailed(text).label

    def analyze_sentiment_detailed(self, text: str) -> SentimentResult:
        prompt = (
            "Classify the sentiment of the following text.\n"
            "Return JSON with exactly these keys: label, confidence, explanation.\n"
            "label must be one of positive, negative, neutral.\n"
            "confidence must be a number from 0 to 1.\n"
            "explanation must be one short sentence.\n\n"
            f"{text}"
        )
        raw_output = self._run_action(
            prompt=prompt,
            instructions="Return valid JSON only with the requested schema.",
        )
        payload = self._parse_json(raw_output)
        label = str(payload.get("label", "")).strip().lower()
        explanation = str(payload.get("explanation", "")).strip()
        confidence = payload.get("confidence")

        if label not in {"positive", "negative", "neutral"}:
            raise AIResponseFormatError(f"Unexpected sentiment label returned: {label!r}")

        try:
            confidence_value = float(confidence)
        except (TypeError, ValueError) as exc:
            raise AIResponseFormatError("Sentiment response confidence must be numeric.") from exc

        if not 0.0 <= confidence_value <= 1.0:
            raise AIResponseFormatError("Sentiment confidence must be between 0 and 1.")
        if not explanation:
            raise AIResponseFormatError("Sentiment explanation must not be empty.")

        return SentimentResult(
            label=label,
            confidence=confidence_value,
            explanation=explanation,
        )

    def answer_question(self, context: str, question: str) -> str:
        prompt = (
            "Answer the question using only the provided context. "
            "If the answer is not in the context, say you do not know.\n\n"
            f"Context:\n{context}\n\nQuestion:\n{question}"
        )
        return self._run_action(
            prompt=prompt,
            instructions="Answer accurately and do not invent facts outside the context.",
        )

    def _run_action(self, prompt: str, instructions: str) -> str:
        try:
            response = self._client.responses.create(
                model=self.config.model,
                instructions=instructions,
                input=prompt,
            )
            output = getattr(response, "output_text", "").strip()
            if not output:
                raise AIActionError("The AI service returned an empty response.")
            return output
        except AIActionError:
            raise
        except AuthenticationError as exc:
            raise AIAuthenticationError("OpenAI authentication failed. Check your API key.") from exc
        except RateLimitError as exc:
            raise AIRateLimitError(
                "OpenAI rate limit or quota exceeded. Check usage limits and billing."
            ) from exc
        except APITimeoutError as exc:
            raise AITimeoutError("OpenAI request timed out.") from exc
        except APIConnectionError as exc:
            raise AIConnectionError("Could not connect to the OpenAI API.") from exc
        except Exception as exc:
            raise AIActionError(f"OpenAI request failed: {exc}") from exc

    @staticmethod
    def _parse_json(raw_output: str) -> dict:
        try:
            payload = json.loads(raw_output)
        except json.JSONDecodeError as exc:
            raise AIResponseFormatError("Model response was not valid JSON.") from exc

        if not isinstance(payload, dict):
            raise AIResponseFormatError("Model response JSON must be an object.")
        return payload
