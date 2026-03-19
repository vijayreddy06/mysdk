from __future__ import annotations

from .._runtime import SDKRuntime
from ..models import AudioTranscriptionResult
from .documents import DocumentActions


class AudioActions:
    def __init__(self, runtime: SDKRuntime, documents: DocumentActions) -> None:
        self._runtime = runtime
        self._documents = documents

    def transcribe(self, file_path: str) -> str:
        return self._runtime.transcribe_audio(file_path)

    def transcribe_detailed(self, file_path: str) -> AudioTranscriptionResult:
        transcript = self.transcribe(file_path)
        return AudioTranscriptionResult(transcript=transcript)

    def summarize(self, file_path: str, style: str = "brief") -> AudioTranscriptionResult:
        transcript = self.transcribe(file_path)
        summary = self._documents.summarize(transcript, style=style)
        return AudioTranscriptionResult(transcript=transcript, summary=summary)
