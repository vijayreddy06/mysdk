# MySDK

`mysdk` is a Python SDK that gives application developers simple AI actions without exposing OpenAI request details in their application code.

This is the beginner stage of the SDK. It focuses on a clean foundation:

- `generate_text(...)`
- `summarize(...)`
- `extract_keywords(...)`
- `analyze_sentiment(...)`
- `answer_question(...)`

## Why this exists

Instead of writing OpenAI client code everywhere, developers can install this SDK and call a stable, easy-to-read Python API.

## Install

Install from a local checkout:

```bash
pip install .
```

For development:

```bash
pip install -e .[dev]
```

## Environment

Set your API key before using the SDK:

```powershell
$env:OPENAI_API_KEY="your_api_key"
```

## Quick start

```python
from mysdk import AIActions

sdk = AIActions()

print(sdk.generate_text("Write a short welcome message for a new SaaS user."))
print(sdk.summarize("Long text goes here"))
print(sdk.extract_keywords("Python SDKs make AI integration easier for developers."))
print(sdk.analyze_sentiment("The onboarding experience was fast and clear."))
print(sdk.answer_question(
    context="Paris is the capital city of France.",
    question="What is the capital of France?",
))
```

## Notes

- Internally, the SDK uses the official OpenAI Python library and the Responses API.
- The default model is `gpt-5`.
- This stage is intentionally simple. The next stages can add structured outputs, streaming, async support, retries, and richer action modules.

## Verify installation

After installation, this should work from any folder:

```bash
python -c "from mysdk import AIActions; print(AIActions)"
```

To run a real request:

```bash
python -c "from mysdk import AIActions; print(AIActions().summarize('OpenAI provides APIs for building AI applications.'))"
```

## Run tests

```bash
pytest
```
