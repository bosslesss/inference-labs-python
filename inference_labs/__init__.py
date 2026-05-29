"""Inference Labs Python SDK.

Vendor-neutral multi-model AI routing. Replace dozens of provider SDKs with one:

    >>> from inference_labs import InferenceLabs
    >>> client = InferenceLabs(api_key="il_live_...")
    >>> out = client.generate(prompt="Summarize this ticket: ...", strategy="cost-first")
    >>> print(out.text)
"""
from .client import (
    InferenceLabs,
    AsyncInferenceLabs,
    GenerationResult,
    InferenceLabsError,
    AuthenticationError,
    RateLimitError,
    InsufficientCreditsError,
    APIError,
)

__version__ = "0.1.1"

__all__ = [
    "InferenceLabs",
    "AsyncInferenceLabs",
    "GenerationResult",
    "InferenceLabsError",
    "AuthenticationError",
    "RateLimitError",
    "InsufficientCreditsError",
    "APIError",
    "__version__",
]
