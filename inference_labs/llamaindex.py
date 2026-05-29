"""LlamaIndex integration -- InferenceLabsLLM.

Optional. Requires ``llama-index-core``. Install via:

    pip install "inference-labs[llamaindex]"

Usage:

    from inference_labs.llamaindex import InferenceLabsLLM
    from llama_index.core.llms import ChatMessage

    llm = InferenceLabsLLM(api_key="il_live_...", strategy="cost-first")
    print(llm.complete("Summarize: ...").text)

    # Use in any LlamaIndex pipeline:
    from llama_index.core import Settings
    Settings.llm = llm
"""
from __future__ import annotations

from typing import Any, AsyncGenerator, Generator, List, Optional, Sequence

try:
    from llama_index.core.llms import (
        ChatMessage,
        ChatResponse,
        ChatResponseAsyncGen,
        ChatResponseGen,
        CompletionResponse,
        CompletionResponseAsyncGen,
        CompletionResponseGen,
        CustomLLM,
        LLMMetadata,
        MessageRole,
    )
    from llama_index.core.llms.callbacks import (
        llm_chat_callback,
        llm_completion_callback,
    )
    from pydantic import Field
except ImportError as exc:  # pragma: no cover -- imported only when extra present
    raise ImportError(
        "InferenceLabsLLM requires llama-index-core. "
        "Install with: pip install 'inference-labs[llamaindex]'"
    ) from exc

from .client import AsyncInferenceLabs, InferenceLabs


def _messages_to_prompt(messages: Sequence[ChatMessage]) -> str:
    """Flatten LlamaIndex ChatMessages into a single prompt string.

    The Inference Labs router takes a free-form prompt; we serialise with
    Role: content prefixes that the routing engine forwards verbatim to the
    chosen provider.
    """
    parts: List[str] = []
    for m in messages:
        role = m.role.value if isinstance(m.role, MessageRole) else str(m.role)
        parts.append(f"{role.capitalize()}: {m.content}")
    return "\n\n".join(parts)


def _additional_kwargs(out) -> dict:
    return {
        "model": out.model,
        "provider": out.provider,
        "cost_usd": out.cost_usd,
        "latency_ms": out.latency_ms,
        "cached": out.cached,
        "trace_id": out.trace_id,
    }


class InferenceLabsLLM(CustomLLM):
    """LlamaIndex CustomLLM wrapper for the Inference Labs router.

    Every call routes through Inference Labs's policy engine. The chosen
    model id appears in the response's ``additional_kwargs['model']``.
    """

    api_key: Optional[str] = Field(default=None, description="Inference Labs API key (il_live_*). Falls back to INFERENCE_LABS_API_KEY env var.")
    base_url: str = Field(default="https://app.inference-labs.com")
    strategy: Optional[str] = Field(default=None, description="balanced | cost-first | quality-first | latency-first | judge")
    max_cost_usd: Optional[float] = Field(default=None)
    max_latency_ms: Optional[int] = Field(default=None)
    allow_models: Optional[List[str]] = Field(default=None)
    deny_models: Optional[List[str]] = Field(default=None)
    workspace_id: Optional[str] = Field(default=None)
    timeout: float = Field(default=60.0)

    # Reasonable defaults for the metadata LlamaIndex expects. Router picks
    # the actual model per request so context_window is the practical floor
    # across the catalog -- 128K covers every current entry.
    context_window: int = Field(default=128_000)
    num_output: int = Field(default=4096)

    @classmethod
    def class_name(cls) -> str:
        return "InferenceLabsLLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name="inference-labs:router",
            is_chat_model=True,
        )

    def _kwargs(self) -> dict:
        return dict(
            strategy=self.strategy,
            max_cost_usd=self.max_cost_usd,
            max_latency_ms=self.max_latency_ms,
            allow_models=self.allow_models,
            deny_models=self.deny_models,
            workspace_id=self.workspace_id,
        )

    def _sync(self) -> InferenceLabs:
        return InferenceLabs(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    def _async(self) -> AsyncInferenceLabs:
        return AsyncInferenceLabs(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    # --- sync ---------------------------------------------------------------
    @llm_completion_callback()
    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        with self._sync() as client:
            out = client.generate(prompt=prompt, **self._kwargs())
        return CompletionResponse(text=out.text, additional_kwargs=_additional_kwargs(out))

    @llm_completion_callback()
    def stream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseGen:
        def _gen() -> Generator[CompletionResponse, None, None]:
            accumulated = ""
            with self._sync() as client:
                for chunk in client.stream(prompt=prompt, **self._kwargs()):
                    accumulated += chunk
                    yield CompletionResponse(text=accumulated, delta=chunk)
        return _gen()

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = _messages_to_prompt(messages)
        with self._sync() as client:
            out = client.generate(prompt=prompt, **self._kwargs())
        msg = ChatMessage(role=MessageRole.ASSISTANT, content=out.text)
        return ChatResponse(message=msg, additional_kwargs=_additional_kwargs(out))

    @llm_chat_callback()
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        prompt = _messages_to_prompt(messages)
        def _gen() -> Generator[ChatResponse, None, None]:
            accumulated = ""
            with self._sync() as client:
                for chunk in client.stream(prompt=prompt, **self._kwargs()):
                    accumulated += chunk
                    msg = ChatMessage(role=MessageRole.ASSISTANT, content=accumulated)
                    yield ChatResponse(message=msg, delta=chunk)
        return _gen()

    # --- async --------------------------------------------------------------
    @llm_completion_callback()
    async def acomplete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponse:
        async with self._async() as client:
            out = await client.generate(prompt=prompt, **self._kwargs())
        return CompletionResponse(text=out.text, additional_kwargs=_additional_kwargs(out))

    @llm_completion_callback()
    async def astream_complete(self, prompt: str, formatted: bool = False, **kwargs: Any) -> CompletionResponseAsyncGen:
        async def _agen() -> AsyncGenerator[CompletionResponse, None]:
            accumulated = ""
            async with self._async() as client:
                async for chunk in client.stream(prompt=prompt, **self._kwargs()):
                    accumulated += chunk
                    yield CompletionResponse(text=accumulated, delta=chunk)
        return _agen()

    @llm_chat_callback()
    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = _messages_to_prompt(messages)
        async with self._async() as client:
            out = await client.generate(prompt=prompt, **self._kwargs())
        msg = ChatMessage(role=MessageRole.ASSISTANT, content=out.text)
        return ChatResponse(message=msg, additional_kwargs=_additional_kwargs(out))

    @llm_chat_callback()
    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseAsyncGen:
        prompt = _messages_to_prompt(messages)
        async def _agen() -> AsyncGenerator[ChatResponse, None]:
            accumulated = ""
            async with self._async() as client:
                async for chunk in client.stream(prompt=prompt, **self._kwargs()):
                    accumulated += chunk
                    msg = ChatMessage(role=MessageRole.ASSISTANT, content=accumulated)
                    yield ChatResponse(message=msg, delta=chunk)
        return _agen()
