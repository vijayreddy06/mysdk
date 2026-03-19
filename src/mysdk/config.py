from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AIConfig:
    api_key: str | None = None
    text_model: str = "gpt-5"
    vision_model: str = "gpt-4.1-mini"
    transcription_model: str = "gpt-4o-mini-transcribe"
    timeout: float = 30.0
    max_retries: int = 2
