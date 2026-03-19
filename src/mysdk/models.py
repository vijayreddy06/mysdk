from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SentimentResult:
    label: str
    confidence: float
    explanation: str


@dataclass(slots=True)
class KeywordExtractionResult:
    keywords: list[str]
    summary: str


@dataclass(slots=True)
class ImageAnalysisResult:
    description: str
    detected_text: str
    tags: list[str]


@dataclass(slots=True)
class AudioTranscriptionResult:
    transcript: str
    summary: str | None = None
