"""LangChain integration -- ChatInferenceLabs.

Optional. Requires ``langchain-core``. Install via:

    pip install "inference-labs[langchain]"

Usage:

    from inference_labs.langchain import ChatInferenceLabs
    from langchain_core.messages import HumanMessage

    llm = ChatInferenceLabs(api_key="il_live_...", strategy="balanced")
    print(llm.invoke([HumanMessage(content="Summarize: ...")]).content)
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Iterator, List, Optional, Sequence

try:
    from langchain_core.callbacks.manager import (
        CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun)
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import (
        AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage)
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
    from pydantic import Field
except ImportError as exc:  # pragma: no cover -- imported only when extra present
    raise ImportError(
        "ChatInferenceLabs requires langchain-core. "
        "Install with: pip install 'inference-labs[langchain]'"
    ) from exc

from .client import AsyncInferenceLabs, InferenceLabs


def _flatten_messages(messages: Sequence[BaseMessage]) -> str:
    """Serialise a LangChain message list into a single prompt string.

    The Inference Labs API takes a free-form prompt, not a typed message list,
    so we flatten with simple Role: content prefixes that the routing engine
    forwards verbatim to the chosen provider.
    """
    parts: List[str] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            parts.append(f"System: {m.content}")
        elif isinstance(m, HumanMessage):
            parts.append(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            parts.append(f"Assistant: {m.content}")
        else:
            parts.append(f"{m.type.capitalize()}: {m.content}")
    return "\n\n".join(parts)


class ChatInferenceLabs(BaseChatModel):
    """LangChain BaseChatModel wrapper for the Inference Labs router.

    Parameters
    ----------
    api_key : str
        Inference Labs API key (``il_live_*``). Falls back to ``INFERENCE_LABS_API_KEY``.
    base_url : str
        Defaults to https://app.inference-labs.com.
    strategy : str, optional
        Routing strategy: ``balanced``, ``cost-first``, ``quality-first``, ``latency-first``, ``judge``.
    max_cost_usd : float, optional
        Per-request cost cap in USD.
    max_latency_ms : int, optional
        Per-request latency budget in milliseconds.
    allow_models / deny_models : list[str], optional
        Restrict the router to / exclude specific model IDs for this call.
    workspace_id : str, optional
        Override the API key's default workspace.
    """

    api_key: Optional[str] = None
    base_url: str = "https://app.inference-labs.com"
    strategy: Optional[str] = None
    max_cost_usd: Optional[float] = None
    max_latency_ms: Optional[int] = None
    allow_models: Optional[List[str]] = None
    deny_models: Optional[List[str]] = None
    workspace_id: Optional[str] = None
    timeout: float = 60.0

    _sync_client: Optional[InferenceLabs] = Field(default=None, exclude=True)
    _async_client: Optional[AsyncInferenceLabs] = Field(default=None, exclude=True)

    @property
    def _llm_type(self) -> str:
        return "inference-labs"

    def _kwargs(self) -> dict:
        return dict(
            strategy=self.strategy,
            max_cost_usd=self.max_cost_usd,
            max_latency_ms=self.max_latency_ms,
            allow_models=self.allow_models,
            deny_models=self.deny_models,
            workspace_id=self.workspace_id,
        )

    def _get_sync(self) -> InferenceLabs:
        if self._sync_client is None:
            self._sync_client = InferenceLabs(
                api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        return self._sync_client

    def _get_async(self) -> AsyncInferenceLabs:
        if self._async_client is None:
            self._async_client = AsyncInferenceLabs(
                api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        return self._async_client

    # --- sync ---------------------------------------------------------------
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = _flatten_messages(messages)
        result = self._get_sync().generate(prompt, **self._kwargs())
        msg = AIMessage(content=result.text, additional_kwargs={
            "model": result.model,
            "provider": result.provider,
            "cost_usd": result.cost_usd,
            "latency_ms": result.latency_ms,
            "cached": result.cached,
            "trace_id": result.trace_id,
        })
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        prompt = _flatten_messages(messages)
        for chunk in self._get_sync().stream(prompt, **self._kwargs()):
            if run_manager:
                run_manager.on_llm_new_token(chunk)
            yield ChatGenerationChunk(message=AIMessageChunk(content=chunk))

    # --- async --------------------------------------------------------------
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = _flatten_messages(messages)
        result = await self._get_async().generate(prompt, **self._kwargs())
        msg = AIMessage(content=result.text, additional_kwargs={
            "model": result.model,
            "provider": result.provider,
            "cost_usd": result.cost_usd,
            "latency_ms": result.latency_ms,
            "cached": result.cached,
            "trace_id": result.trace_id,
        })
        return ChatResult(generations=[ChatGeneration(message=msg)])

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        prompt = _flatten_messages(messages)
        async for chunk in self._get_async().stream(prompt, **self._kwargs()):
            if run_manager:
                await run_manager.on_llm_new_token(chunk)
            yield ChatGenerationChunk(message=AIMessageChunk(content=chunk))
