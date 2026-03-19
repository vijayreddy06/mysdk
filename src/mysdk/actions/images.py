from __future__ import annotations

from .._runtime import SDKRuntime
from ..errors import AIResponseFormatError
from ..models import ImageAnalysisResult


class ImageActions:
    def __init__(self, runtime: SDKRuntime) -> None:
        self._runtime = runtime

    def describe(self, image_path: str, prompt: str | None = None) -> str:
        return self.analyze(image_path=image_path, prompt=prompt).description

    def analyze(self, image_path: str, prompt: str | None = None) -> ImageAnalysisResult:
        data_url = self._runtime.build_data_url(image_path)
        user_prompt = prompt or (
            "Describe this image for a developer. Mention the main subject, any visible text, "
            "and a few concise tags."
        )
        input_payload = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": user_prompt},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ]
        raw_output = self._runtime.run_text_action(
            model=self._runtime.config.vision_model,
            instructions="Return valid JSON only with keys description, detected_text, tags.",
            input_payload=input_payload,
        )
        payload = self._runtime.parse_json(raw_output)
        description = str(payload.get("description", "")).strip()
        detected_text = str(payload.get("detected_text", "")).strip()
        tags = payload.get("tags")
        if not description or not isinstance(tags, list):
            raise AIResponseFormatError("Image analysis response is missing required fields.")
        cleaned_tags = [tag.strip() for tag in tags if isinstance(tag, str) and tag.strip()]
        return ImageAnalysisResult(
            description=description,
            detected_text=detected_text,
            tags=cleaned_tags,
        )

    def answer_question(self, image_path: str, question: str) -> str:
        data_url = self._runtime.build_data_url(image_path)
        input_payload = [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": question},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ]
        return self._runtime.run_text_action(
            model=self._runtime.config.vision_model,
            instructions="Answer the question about the image accurately and concisely.",
            input_payload=input_payload,
        )
