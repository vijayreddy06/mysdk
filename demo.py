from pathlib import Path
from mysdk import AIActionError, AIAuthenticationError, AIActions

IMAGE_PATH = Path(r"C:\Users\vrsr1\Downloads\linkedin headshot.jpg")
AUDIO_PATH = Path(r"C:\Users\vrsr1\Downloads\testaudio.mp4")

try:
    sdk = AIActions()
except AIAuthenticationError as exc:
    print(f"Configuration error: {exc}")
    raise SystemExit(1)

try:
    print("Short summary:", sdk.documents.summarize_short(
        "OpenAI enables developers to build AI-powered applications with a cleaner Python SDK."
    ))
    print("Keywords:", sdk.documents.extract_keywords_detailed(
        "OpenAI enables developers to build AI-powered applications with a cleaner Python SDK."
    ))

    if IMAGE_PATH.exists():
        print("Image:", sdk.images.analyze(str(IMAGE_PATH)))
    else:
        print(f"Image file not found: {IMAGE_PATH}")

    if AUDIO_PATH.exists():
        print("Audio:", sdk.audio.summarize(str(AUDIO_PATH)))
    else:
        print(f"Audio file not found: {AUDIO_PATH}")
except AIActionError as exc:
    print(f"Request failed: {exc}")
    raise SystemExit(1)
