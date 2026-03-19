from __future__ import annotations

from ._runtime import SDKRuntime
from .actions import AudioActions, DocumentActions, ImageActions
from .config import AIConfig
from .models import AudioTranscriptionResult, ImageAnalysisResult, KeywordExtractionResult, SentimentResult


class AIActions:
    """Advanced SDK facade exposing document, image, and audio action groups."""

    def __init__(self, config: AIConfig | None = None) -> None:
        self._runtime = SDKRuntime(config=config)
        self.documents = DocumentActions(self._runtime)
        self.images = ImageActions(self._runtime)
        self.audio = AudioActions(self._runtime, self.documents)

    # Backward-compatible document shortcuts.
    def generate_text(self, prompt: str, system_prompt: str | None = None) -> str:
        return self.documents.generate_text(prompt=prompt, system_prompt=system_prompt)

    def summarize(self, text: str, style: str = "brief") -> str:
        return self.documents.summarize(text=text, style=style)

    def summarize_short(self, text: str) -> str:
        return self.documents.summarize_short(text=text)

    def extract_keywords(self, text: str, max_keywords: int = 5) -> list[str]:
        return self.documents.extract_keywords(text=text, max_keywords=max_keywords)

    def extract_keywords_detailed(self, text: str, max_keywords: int = 5) -> KeywordExtractionResult:
        return self.documents.extract_keywords_detailed(text=text, max_keywords=max_keywords)

    def analyze_sentiment(self, text: str) -> str:
        return self.documents.analyze_sentiment(text=text)

    def analyze_sentiment_detailed(self, text: str) -> SentimentResult:
        return self.documents.analyze_sentiment_detailed(text=text)

    def answer_question(self, context: str, question: str) -> str:
        return self.documents.answer_question(context=context, question=question)


__all__ = [
    "AIActions",
    "AIConfig",
    "AudioTranscriptionResult",
    "ImageAnalysisResult",
    "KeywordExtractionResult",
    "SentimentResult",
]
