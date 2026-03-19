class AIActionError(RuntimeError):
    """Raised when an SDK action cannot complete successfully."""


class AIAuthenticationError(AIActionError):
    """Raised when the SDK cannot authenticate with OpenAI."""


class AIRateLimitError(AIActionError):
    """Raised when the OpenAI API rate limit or quota is exceeded."""


class AITimeoutError(AIActionError):
    """Raised when an OpenAI request times out."""


class AIConnectionError(AIActionError):
    """Raised when the SDK cannot reach the OpenAI API."""


class AIResponseFormatError(AIActionError):
    """Raised when the model response cannot be parsed into the expected format."""


class AIFileError(AIActionError):
    """Raised when an SDK file input is missing or invalid."""
