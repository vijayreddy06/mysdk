# MySDK

`mysdk-vijay` is a modular Python SDK that wraps OpenAI behind practical AI actions for documents, images, and audio.

The goal of this SDK is to let developers call a clean Python API instead of writing raw OpenAI request code throughout their applications.

## What The SDK Can Do

- Document actions: generate text, summarize, extract keywords, analyze sentiment, answer questions from context
- Image actions: describe images, analyze images, answer questions about an image
- Audio actions: transcribe audio and summarize transcripts

## Package Structure

The main SDK entry point is `AIActions`.

```python
from mysdk import AIActions

sdk = AIActions()
```

After creating the SDK, the advanced modules are available as:

- `sdk.documents`
- `sdk.images`
- `sdk.audio`

The package also keeps backward-compatible shortcuts like:

- `sdk.generate_text(...)`
- `sdk.summarize(...)`
- `sdk.extract_keywords(...)`
- `sdk.analyze_sentiment(...)`
- `sdk.answer_question(...)`

## Install

Install the package normally:

```bash
pip install mysdk-vijay
```

If you are working from the source repo locally:

```bash
pip install .
```

For development with tests:

```bash
pip install -e .[dev]
```

## Environment Setup

Set your OpenAI API key before running live SDK calls.

PowerShell:

```powershell
$env:OPENAI_API_KEY="your_api_key"
```

Command Prompt:

```cmd
set OPENAI_API_KEY=your_api_key
```

## Quick Start

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

## Main Modules

### Documents

Use `sdk.documents` for text and document-style operations.

Examples:

```python
sdk.documents.summarize("Long text")
sdk.documents.summarize_short("Long text")
sdk.documents.extract_keywords_detailed("Some content")
sdk.documents.analyze_sentiment_detailed("The SDK is practical and clear.")
sdk.documents.answer_question(
    context="The company is based in Austin.",
    question="Where is the company based?",
)
```

### Images

Use `sdk.images` for image understanding.

Examples:

```python
sdk.images.describe("photo.jpg")
sdk.images.analyze("chart.png")
sdk.images.answer_question("screenshot.png", "What does this image show?")
```

### Audio

Use `sdk.audio` for audio transcription and transcript summarization.

Examples:

```python
sdk.audio.transcribe("meeting.wav")
sdk.audio.transcribe_detailed("meeting.wav")
sdk.audio.summarize("meeting.wav")
```

## Test The Installed Package

After installing the package on any machine, you can test document, image, and audio support with this script.

Replace the image and audio paths with real files on that machine.

```python
from mysdk import AIActions, AIActionError

DOCUMENT_TEXT = """
OpenAI provides APIs that help developers build AI-powered applications.
Python SDKs make integration cleaner, faster, and easier to maintain.
"""

IMAGE_PATH = r"C:\path\to\your\image.jpg"
AUDIO_PATH = r"C:\path\to\your\audio.mp4"

try:
    sdk = AIActions()

    print("DOCUMENT RESULT:")
    print(sdk.documents.summarize_short(DOCUMENT_TEXT))

    print("\nIMAGE RESULT:")
    print(sdk.images.analyze(IMAGE_PATH))

    print("\nAUDIO RESULT:")
    print(sdk.audio.summarize(AUDIO_PATH))

except AIActionError as exc:
    print(f"SDK error: {exc}")
```

Run it with:

PowerShell:

```powershell
$env:OPENAI_API_KEY="your_api_key"
python test_sdk.py
```

Command Prompt:

```cmd
set OPENAI_API_KEY=your_api_key
python test_sdk.py
```

## Verify The Installed Version

To confirm that the installed package includes the advanced API:

```bash
python -c "from mysdk import AIActions; sdk=AIActions(); print(hasattr(sdk, 'documents'), hasattr(sdk, 'images'), hasattr(sdk, 'audio'))"
```

Expected output:

```python
True True True
```

## Run Unit Tests

The test suite uses mocked OpenAI behavior, so it does not require a live API key.

```bash
pytest
```

## Notes

- Internally, the SDK uses the official OpenAI Python library.
- The package is modular, but the top-level compatibility methods are still available for simpler usage.
- Image and audio actions require real local file paths.
- Current document methods operate on text content, not document file paths like PDF or DOCX.
