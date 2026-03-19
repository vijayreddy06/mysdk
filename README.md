# MySDK

`mysdk` is a modular Python SDK that wraps OpenAI behind practical AI actions for documents, images, and audio.

## Install

```bash
pip install .
```

For development:

```bash
pip install -e .[dev]
```

## Environment

```powershell
$env:OPENAI_API_KEY="your_api_key"
```

## Advanced Quick Start

```python
from mysdk import AIActions

sdk = AIActions()

print(sdk.documents.summarize_short("Long document text"))
print(sdk.documents.extract_keywords_detailed("Python SDKs simplify AI integration."))
print(sdk.documents.answer_question(
    context="Paris is the capital of France.",
    question="What is the capital of France?",
))
print(sdk.images.describe("sample.png"))
print(sdk.audio.transcribe("meeting.wav"))
```

## Modules

- `sdk.documents`: text generation, summarization, keywords, sentiment, question answering
- `sdk.images`: image description, image question answering
- `sdk.audio`: audio transcription and transcript summarization

## Notes

- Internally, the SDK uses the official OpenAI Python library.
- The advanced stage preserves top-level compatibility methods like `generate_text()` and `summarize()`.
- Unit tests mock OpenAI, so `pytest` does not require a live API key.

## Verify Installation

```bash
python -c "from mysdk import AIActions; print(AIActions)"
```

## Run Tests

```bash
pytest
```
