from mysdk import AIActionError, AIAuthenticationError, AIActions

try:
    sdk = AIActions()
except AIAuthenticationError as exc:
    print(f"Configuration error: {exc}")
    raise SystemExit(1)

try:
    keywords = sdk.extract_keywords_detailed(
        "OpenAI enables developers to build AI-powered applications with a cleaner Python SDK."
    )
    sentiment = sdk.analyze_sentiment_detailed(
        "The beginner SDK is clear, practical, and easy to test."
    )
    answer = sdk.answer_question(
        context="The company was founded in 2020 and is based in Austin.",
        question="Where is the company based?"
    )
except AIActionError as exc:
    print(f"Request failed: {exc}")
    raise SystemExit(1)

print("Keywords:", keywords)
print("Sentiment:", sentiment)
print("Answer:", answer)
