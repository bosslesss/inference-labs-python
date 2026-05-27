# inference-labs

[![CI](https://github.com/bosslesss/inference-labs-python/actions/workflows/ci.yml/badge.svg)](https://github.com/bosslesss/inference-labs-python/actions/workflows/ci.yml)
[![GitHub release](https://img.shields.io/github/v/release/bosslesss/inference-labs-python)](https://github.com/bosslesss/inference-labs-python/releases)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://pypi.org/project/inference-labs/)

Official Python client for [Inference Labs](https://inference-labs.com) — a vendor-neutral router for the major cloud LLMs (OpenAI / Azure / Anthropic / Google / AWS Bedrock / RunwayML). One endpoint, one billing surface, automatic failover, semantic caching, and policy-driven model selection (`cost-first`, `quality-first`, `latency-first`, `balanced`, `judge`).

```bash
pip install inference-labs
```

Or install the current release directly from GitHub (no PyPI account needed by us — works today):

```bash
pip install https://github.com/bosslesss/inference-labs-python/releases/download/v0.1.0/inference_labs-0.1.0-py3-none-any.whl
```

Optional LangChain integration:

```bash
pip install "inference-labs[langchain]"
```

## Quickstart

```python
from inference_labs import InferenceLabs

client = InferenceLabs(api_key="il_live_...")  # or INFERENCE_LABS_API_KEY env var

out = client.generate(
    prompt="Summarize this ticket: the laser printer is offline...",
    strategy="cost-first",
    max_cost_usd=0.01,
)
print(out.text)
print(f"routed via {out.provider}/{out.model} -- ${out.cost_usd:.5f}")
```

Streaming:

```python
for chunk in client.stream(prompt="Write a haiku about caching."):
    print(chunk, end="", flush=True)
```

Async (same surface, awaitable):

```python
import asyncio
from inference_labs import AsyncInferenceLabs

async def main():
    async with AsyncInferenceLabs() as client:
        out = await client.generate(prompt="Hello.")
        print(out.text)

asyncio.run(main())
```

## LangChain

```python
from inference_labs.langchain import ChatInferenceLabs
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatInferenceLabs(
    api_key="il_live_...",
    strategy="balanced",
    max_cost_usd=0.01,
)

resp = llm.invoke([
    SystemMessage(content="You are a terse SRE."),
    HumanMessage(content="What does workers=1 threads=8 mean for SQLite?"),
])
print(resp.content)
print(resp.additional_kwargs)   # -> model, provider, cost_usd, latency_ms, cached, trace_id
```

## Routing options

All parameters below are optional and can be set per call.

| Parameter | Type | What it does |
|---|---|---|
| `strategy` | `"balanced"` / `"cost-first"` / `"quality-first"` / `"latency-first"` / `"judge"` | Picks the policy the router uses to choose between models in your allowlist. |
| `max_cost_usd` | `float` | Hard cap on per-request cost in USD. |
| `max_latency_ms` | `int` | Latency budget in milliseconds. |
| `allow_models` | `list[str]` | Restrict the call to a subset of model IDs. |
| `deny_models` | `list[str]` | Exclude specific model IDs from selection. |
| `workspace_id` | `str` | Override the API key's default workspace. |
| `collect_trace` | `bool` | Persist a redacted trace for evals (default `True`). |
| `redact_pii` | `bool` | Run the PII / secrets redactor before storage (default `True`). |

The router returns a small typed object:

```python
@dataclass
class GenerationResult:
    text: str
    model: str
    provider: str
    cost_usd: float
    latency_ms: int
    cached: bool
    trace_id: str | None
    raw: dict   # whole response payload if you need fields we don't surface
```

## Errors

```python
from inference_labs import (
    InferenceLabsError, AuthenticationError, RateLimitError,
    InsufficientCreditsError, APIError,
)
```

All exceptions inherit from `InferenceLabsError` so you can catch one.

## Configuration

```python
client = InferenceLabs(
    api_key="il_live_...",                  # or INFERENCE_LABS_API_KEY
    base_url="https://app.inference-labs.com",  # override for staging / self-hosted
    timeout=60.0,
)
```

For multi-tenant frameworks pass your own `httpx.Client` / `httpx.AsyncClient` via the `client=` kwarg so the SDK reuses your connection pool.

## License

Apache-2.0. See [LICENSE](./LICENSE).

## Links

- Marketing: <https://inference-labs.com>
- App / dashboard: <https://app.inference-labs.com>
- Issues: <https://github.com/bosslesss/InferenceLabs/issues>
- Blog: <https://blog.inference-labs.com>
