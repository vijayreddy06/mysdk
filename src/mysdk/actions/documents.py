from __future__ import annotations

from ..models import KeywordExtractionResult, SentimentResult
from ..errors import AIResponseFormatError
from .._runtime import SDKRuntime


class DocumentActions:
    def __init__(self, runtime: SDKRuntime) -> None:
        self._runtime = runtime

    def generate_text(self, prompt: str, system_prompt: str | None = None) -> str:
        instructions = system_prompt or "You are a helpful assistant for application developers."
        return self._runtime.run_text_action(
            model=self._runtime.config.text_model,
            instructions=instructions,
            input_payload=prompt,
        )

    def summarize(self, text: str, style: str = "brief") -> str:
        prompt = f"Summarize the following text in a {style} way:\n\n{text}"
        instructions = "Create a clear summary that preserves the key meaning."
        return self._runtime.run_text_action(
            model=self._runtime.config.text_model,
            instructions=instructions,
            input_payload=prompt,
        )

    def summarize_short(self, text: str) -> str:
        return self.summarize(text=text, style="short")

    def extract_keywords(self, text: str, max_keywords: int = 5) -> list[str]:
        return self.extract_keywords_detailed(text=text, max_keywords=max_keywords).keywords

    def extract_keywords_detailed(self, text: str, max_keywords: int = 5) -> KeywordExtractionResult:
        prompt = (
            f"Extract up to {max_keywords} important keywords from this text.\n"
            "Return JSON with exactly these keys: keywords, summary.\n"
            "keywords must be an array of short strings. summary must be one short sentence.\n\n"
            f"{text}"
        )
        raw_output = self._runtime.run_text_action(
            model=self._runtime.config.text_model,
            instructions="Return valid JSON only with the requested schema.",
            input_payload=prompt,
        )
        payload = self._runtime.parse_json(raw_output)
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
        raw_output = self._runtime.run_text_action(
            model=self._runtime.config.text_model,
            instructions="Return valid JSON only with the requested schema.",
            input_payload=prompt,
        )
        payload = self._runtime.parse_json(raw_output)
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
        return self._runtime.run_text_action(
            model=self._runtime.config.text_model,
            instructions="Answer accurately and do not invent facts outside the context.",
            input_payload=prompt,
        )
