# Changelog

## 0.1.0 — 2026-05-27

Initial release.

- `InferenceLabs` and `AsyncInferenceLabs` sync + async clients.
- `client.generate(...)` and `client.stream(...)` for `/api/v1/generate` and `/api/v1/generate/stream`.
- Typed errors: `AuthenticationError`, `RateLimitError`, `InsufficientCreditsError`, `APIError`.
- `GenerationResult` dataclass with model / provider / cost / latency / cached / trace_id fields.
- Optional LangChain integration via `inference_labs.langchain.ChatInferenceLabs` (extra `pip install "inference-labs[langchain]"`).
- Examples in `examples/quickstart.py`.
