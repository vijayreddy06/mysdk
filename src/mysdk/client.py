from __future__ import annotations

import os
from dataclasses import dataclass

from openai import OpenAI

from .errors import AIActionError


@dataclass(slots=True)
class AIConfig:
    api_key: str | None = None
    model: str = "gpt-5"


class AIActions:
    """Beginner-friendly AI actions built on top of OpenAI's Responses API."""

    def __init__(self, config: AIConfig | None = None) -> None:
        self.config = config or AIConfig()
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        self._client = OpenAI(api_key=api_key)

    def generate_text(self, prompt: str, system_prompt: str | None = None) -> str:
        instructions = system_prompt or "You are a helpful assistant for application developers."
        return self._run_action(prompt=prompt, instructions=instructions)

    def summarize(self, text: str, style: str = "brief") -> str:
        prompt = f"Summarize the following text in a {style} way:\n\n{text}"
        instructions = "Create a clear summary that preserves the key meaning."
        return self._run_action(prompt=prompt, instructions=instructions)

    def extract_keywords(self, text: str, max_keywords: int = 5) -> list[str]:
        prompt = (
            f"Extract up to {max_keywords} important keywords from this text.\n"
            "Return only a comma-separated list with no explanation.\n\n"
            f"{text}"
        )
        raw_output = self._run_action(
            prompt=prompt,
            instructions="Identify the most important keywords for developers.",
        )
        return self._parse_csv_list(raw_output, max_items=max_keywords)

    def analyze_sentiment(self, text: str) -> str:
        prompt = (
            "Classify the sentiment of the following text as exactly one word: "
            "positive, negative, or neutral.\n\n"
            f"{text}"
        )
        raw_output = self._run_action(
            prompt=prompt,
            instructions="Return only one lowercase sentiment label.",
        )
        normalized = raw_output.strip().lower()
        if normalized not in {"positive", "negative", "neutral"}:
            raise AIActionError(f"Unexpected sentiment label returned: {raw_output!r}")
        return normalized

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
        except Exception as exc:
            raise AIActionError(f"OpenAI request failed: {exc}") from exc

    @staticmethod
    def _parse_csv_list(raw_output: str, max_items: int) -> list[str]:
        items = []
        for part in raw_output.replace("\n", ",").split(","):
            cleaned = part.strip()
            if cleaned and cleaned not in items:
                items.append(cleaned)
            if len(items) >= max_items:
                break
        if not items:
            raise AIActionError("No keywords were returned.")
        return items
