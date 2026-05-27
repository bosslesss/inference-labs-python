"""Inference Labs HTTP client.

Wraps POST /api/v1/generate and /api/v1/generate/stream on app.inference-labs.com.
Sync + async (httpx) implementations share the same request shape; the response
is normalised into a small ``GenerationResult`` dataclass so callers don't have
to dig through nested dicts.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator, Mapping, Optional, Sequence

import httpx


DEFAULT_BASE_URL = "https://app.inference-labs.com"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class InferenceLabsError(Exception):
    """Base class for all SDK errors."""


class AuthenticationError(InferenceLabsError):
    """Bad/missing API key."""


class RateLimitError(InferenceLabsError):
    """Per-key or per-workspace rate-limit exceeded."""


class InsufficientCreditsError(InferenceLabsError):
    """Workspace credit balance is too low for the request."""


class APIError(InferenceLabsError):
    """Any other 4xx/5xx returned by the platform."""

    def __init__(self, status: int, code: str, message: str, details: Any = None):
        super().__init__(f"{status} {code}: {message}")
        self.status = status
        self.code = code
        self.message = message
        self.details = details


def _raise_for_error(status: int, body: Mapping[str, Any]):
    err = body.get("error") if isinstance(body, Mapping) else None
    if err and isinstance(err, Mapping):
        code = err.get("code", "unknown")
        msg = err.get("message", "")
        details = err.get("details")
        if status == 401:
            raise AuthenticationError(msg or "Invalid API key")
        if status == 402 or code in ("insufficient_credits", "credit_required"):
            raise InsufficientCreditsError(msg or "Workspace out of credits")
        if status == 429:
            raise RateLimitError(msg or "Rate limit exceeded")
        raise APIError(status, code, msg, details)
    raise APIError(status, "unknown", str(body)[:200])


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass
class GenerationResult:
    """Normalised output of a successful /api/v1/generate call."""

    text: str = ""
    model: str = ""
    provider: str = ""
    cost_usd: float = 0.0
    latency_ms: int = 0
    cached: bool = False
    trace_id: Optional[str] = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "GenerationResult":
        d = payload.get("data", payload) if isinstance(payload, Mapping) else {}
        return cls(
            text=d.get("text") or d.get("completion") or d.get("output", ""),
            model=d.get("model", ""),
            provider=d.get("provider", ""),
            cost_usd=float(d.get("cost_usd", 0) or 0),
            latency_ms=int(d.get("latency_ms", 0) or 0),
            cached=bool(d.get("cached", False)),
            trace_id=d.get("trace_id"),
            raw=d,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_request_body(
    *,
    prompt: str,
    modality: str = "text",
    strategy: Optional[str] = None,
    max_cost_usd: Optional[float] = None,
    max_latency_ms: Optional[int] = None,
    allow_models: Optional[Sequence[str]] = None,
    deny_models: Optional[Sequence[str]] = None,
    workspace_id: Optional[str] = None,
    collect_trace: bool = True,
    redact_pii: bool = True,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict:
    constraints: dict = {}
    if strategy is not None:
        constraints["strategy"] = strategy
    if max_cost_usd is not None:
        constraints["max_cost_usd"] = max_cost_usd
    if max_latency_ms is not None:
        constraints["max_latency_ms"] = max_latency_ms

    body: dict = {"modality": modality, "prompt": prompt}
    if constraints:
        body["constraints"] = constraints
    if allow_models:
        body["allow_models"] = list(allow_models)
    if deny_models:
        body["deny_models"] = list(deny_models)
    if workspace_id:
        body["workspace_id"] = workspace_id
    body["trace"] = {"collect": collect_trace, "redact_pii": redact_pii}
    if extra:
        for k, v in extra.items():
            body.setdefault(k, v)
    return body


def _headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "inference-labs-python/0.1.0",
        "Accept": "application/json",
    }


# ---------------------------------------------------------------------------
# Sync client
# ---------------------------------------------------------------------------
class InferenceLabs:
    """Synchronous Inference Labs client.

    Parameters
    ----------
    api_key : str, optional
        Your ``il_live_*`` API key. Falls back to ``INFERENCE_LABS_API_KEY`` env var.
    base_url : str, optional
        Override the platform base URL (default https://app.inference-labs.com).
    timeout : float, optional
        Per-request timeout in seconds (default 60).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
        client: Optional[httpx.Client] = None,
    ):
        key = api_key or os.environ.get("INFERENCE_LABS_API_KEY")
        if not key:
            raise AuthenticationError(
                "Provide api_key or set INFERENCE_LABS_API_KEY env var.")
        self._api_key = key
        self._base_url = base_url.rstrip("/")
        self._http = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None

    def close(self):
        if self._owns_client:
            self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # --- main entry point ---------------------------------------------------
    def generate(
        self,
        prompt: str,
        *,
        modality: str = "text",
        strategy: Optional[str] = None,
        max_cost_usd: Optional[float] = None,
        max_latency_ms: Optional[int] = None,
        allow_models: Optional[Sequence[str]] = None,
        deny_models: Optional[Sequence[str]] = None,
        workspace_id: Optional[str] = None,
        collect_trace: bool = True,
        redact_pii: bool = True,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> GenerationResult:
        body = _build_request_body(
            prompt=prompt, modality=modality, strategy=strategy,
            max_cost_usd=max_cost_usd, max_latency_ms=max_latency_ms,
            allow_models=allow_models, deny_models=deny_models,
            workspace_id=workspace_id, collect_trace=collect_trace,
            redact_pii=redact_pii, extra=extra,
        )
        r = self._http.post(
            f"{self._base_url}/api/v1/generate",
            headers=_headers(self._api_key),
            content=json.dumps(body).encode(),
        )
        return self._parse(r)

    def stream(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Yield text chunks from a streamed generation. SSE under the hood."""
        body = _build_request_body(prompt=prompt, **kwargs)
        with self._http.stream(
            "POST",
            f"{self._base_url}/api/v1/generate/stream",
            headers=_headers(self._api_key),
            content=json.dumps(body).encode(),
        ) as r:
            if r.status_code >= 400:
                payload: Any = {}
                try:
                    payload = r.json()
                except Exception:
                    payload = {"error": {"code": "stream_error", "message": r.text}}
                _raise_for_error(r.status_code, payload)
            for line in r.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    yield data
                    continue
                chunk = obj.get("delta") or obj.get("text") or ""
                if chunk:
                    yield chunk

    # --- internal -----------------------------------------------------------
    def _parse(self, r: httpx.Response) -> GenerationResult:
        try:
            payload = r.json()
        except json.JSONDecodeError:
            raise APIError(r.status_code, "non_json_response", r.text[:200])
        if r.status_code >= 400:
            _raise_for_error(r.status_code, payload)
        return GenerationResult.from_payload(payload)


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------
class AsyncInferenceLabs:
    """Async Inference Labs client. Mirrors :class:`InferenceLabs`."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
        client: Optional[httpx.AsyncClient] = None,
    ):
        key = api_key or os.environ.get("INFERENCE_LABS_API_KEY")
        if not key:
            raise AuthenticationError(
                "Provide api_key or set INFERENCE_LABS_API_KEY env var.")
        self._api_key = key
        self._base_url = base_url.rstrip("/")
        self._http = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None

    async def aclose(self):
        if self._owns_client:
            await self._http.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.aclose()

    async def generate(
        self,
        prompt: str,
        *,
        modality: str = "text",
        strategy: Optional[str] = None,
        max_cost_usd: Optional[float] = None,
        max_latency_ms: Optional[int] = None,
        allow_models: Optional[Sequence[str]] = None,
        deny_models: Optional[Sequence[str]] = None,
        workspace_id: Optional[str] = None,
        collect_trace: bool = True,
        redact_pii: bool = True,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> GenerationResult:
        body = _build_request_body(
            prompt=prompt, modality=modality, strategy=strategy,
            max_cost_usd=max_cost_usd, max_latency_ms=max_latency_ms,
            allow_models=allow_models, deny_models=deny_models,
            workspace_id=workspace_id, collect_trace=collect_trace,
            redact_pii=redact_pii, extra=extra,
        )
        r = await self._http.post(
            f"{self._base_url}/api/v1/generate",
            headers=_headers(self._api_key),
            content=json.dumps(body).encode(),
        )
        return self._parse(r)

    async def stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        body = _build_request_body(prompt=prompt, **kwargs)
        async with self._http.stream(
            "POST",
            f"{self._base_url}/api/v1/generate/stream",
            headers=_headers(self._api_key),
            content=json.dumps(body).encode(),
        ) as r:
            if r.status_code >= 400:
                try:
                    payload = await r.aread()
                    payload = json.loads(payload)
                except Exception:
                    payload = {"error": {"code": "stream_error", "message": "see status"}}
                _raise_for_error(r.status_code, payload)
            async for line in r.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    yield data
                    continue
                chunk = obj.get("delta") or obj.get("text") or ""
                if chunk:
                    yield chunk

    def _parse(self, r: httpx.Response) -> GenerationResult:
        try:
            payload = r.json()
        except json.JSONDecodeError:
            raise APIError(r.status_code, "non_json_response", r.text[:200])
        if r.status_code >= 400:
            _raise_for_error(r.status_code, payload)
        return GenerationResult.from_payload(payload)
