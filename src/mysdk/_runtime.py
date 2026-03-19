from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    OpenAI,
    RateLimitError,
)

from .config import AIConfig
from .errors import (
    AIActionError,
    AIAuthenticationError,
    AIConnectionError,
    AIFileError,
    AIRateLimitError,
    AIResponseFormatError,
    AITimeoutError,
)


class SDKRuntime:
    def __init__(self, config: AIConfig | None = None) -> None:
        self.config = config or AIConfig()
        api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise AIAuthenticationError(
                "OpenAI API key is missing. Set OPENAI_API_KEY or pass api_key in AIConfig."
            )
        self.client = OpenAI(
            api_key=api_key,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
        )

    def run_text_action(self, *, model: str, instructions: str, input_payload) -> str:
        try:
            response = self.client.responses.create(
                model=model,
                instructions=instructions,
                input=input_payload,
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

    def transcribe_audio(self, file_path: str | Path) -> str:
        path = self._ensure_file(file_path)
        try:
            with path.open("rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    model=self.config.transcription_model,
                    file=audio_file,
                )
            transcript = getattr(response, "text", "").strip()
            if not transcript:
                raise AIActionError("The transcription service returned an empty transcript.")
            return transcript
        except AIActionError:
            raise
        except AuthenticationError as exc:
            raise AIAuthenticationError("OpenAI authentication failed. Check your API key.") from exc
        except RateLimitError as exc:
            raise AIRateLimitError(
                "OpenAI rate limit or quota exceeded. Check usage limits and billing."
            ) from exc
        except APITimeoutError as exc:
            raise AITimeoutError("OpenAI transcription request timed out.") from exc
        except APIConnectionError as exc:
            raise AIConnectionError("Could not connect to the OpenAI API.") from exc
        except Exception as exc:
            raise AIActionError(f"OpenAI transcription failed: {exc}") from exc

    @staticmethod
    def parse_json(raw_output: str) -> dict:
        candidates = [raw_output.strip()]
        stripped = raw_output.strip()

        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if len(lines) >= 3 and lines[-1].strip() == "```":
                candidates.append("\n".join(lines[1:-1]).strip())

        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(stripped[start:end + 1])

        for candidate in candidates:
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                raise AIResponseFormatError("Model response JSON must be an object.")
            return payload

        raise AIResponseFormatError("Model response was not valid JSON.")

    @staticmethod
    def ensure_file(file_path: str | Path) -> Path:
        return SDKRuntime._ensure_file(file_path)

    @staticmethod
    def build_data_url(file_path: str | Path) -> str:
        path = SDKRuntime._ensure_file(file_path)
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _ensure_file(file_path: str | Path) -> Path:
        path = Path(file_path)
        if not path.is_file():
            raise AIFileError(f"File not found: {path}")
        return path
