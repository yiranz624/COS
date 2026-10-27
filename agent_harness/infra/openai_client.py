"""OpenAI-compatible LLMClient — wraps :class:`openai.AsyncOpenAI`.

Works against the OpenAI API and any OpenAI-Chat-Completions-compatible
endpoint (vLLM, SGLang, OpenRouter, etc.). For Anthropic,
use ``infra.anthropic_client.AnthropicClient`` (separate file, separate SDK).
"""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from openai import AsyncOpenAI
from agent_harness.infra.openrouter_policy import request_policy, reasoning_text, field, applies

from agent_harness.core.llm import LLMClient, LLMResponse, StreamDelta
from agent_harness.core.messages import Message, ToolCall

logger = logging.getLogger(__name__)


class OpenAIClient(LLMClient):
    """Thin async wrapper for OpenAI-compatible chat-completion endpoints."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = None,
        max_completion_tokens: int | None = None,
        max_tokens_param: str = "max_completion_tokens",
        parallel_tool_calls: bool | None = True,
        timeout: float | None = None,
        default_headers: dict[str, str] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        if max_tokens_param not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError(
                "max_tokens_param must be 'max_tokens' or "
                "'max_completion_tokens'"
            )
        effective = request_policy(base_url, {"model": model, "max_tokens": max_completion_tokens})
        model = effective["model"]
        if applies(base_url, model):
            max_completion_tokens = effective["max_tokens"]
            max_tokens_param = "max_tokens"
        self.model = model
        self.default_temperature = temperature
        self.default_max_tokens = max_completion_tokens
        self.max_tokens_param = max_tokens_param
        self.parallel_tool_calls = parallel_tool_calls
        self.default_timeout = timeout
        self.extra_body = extra_body or {}
        # Per-call timeout is enforced by the agent loop's
        # ``asyncio.wait_for`` wrapper, not by the SDK.
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url or None,
            timeout=timeout,
            default_headers=default_headers,
            max_retries=0,           # retries are owned by the runtime loop
        )

    # ── Non-streaming ────────────────────────────────────────────────────

    async def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
        }
        if tools:
            kwargs["tools"] = tools
            if self.parallel_tool_calls is not None:
                kwargs["parallel_tool_calls"] = self.parallel_tool_calls
        eff_temp = temperature if temperature is not None else self.default_temperature
        if eff_temp is not None:
            kwargs["temperature"] = eff_temp
        eff_mt = max_tokens if max_tokens is not None else self.default_max_tokens
        if eff_mt is not None:
            kwargs[self.max_tokens_param] = eff_mt
        if extra_headers:
            kwargs["extra_headers"] = extra_headers
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        # Per-call ``timeout`` triggers ``x-stainless-read-timeout``
        # header; only set when the caller explicitly passed one.
        if timeout is not None:
            kwargs["timeout"] = timeout

        kwargs = request_policy(self._client.base_url, kwargs, sdk=True)
        raw_response = await self._client.chat.completions.with_raw_response.create(
            **kwargs,
        )
        raw = raw_response.parse()
        response = _to_llm_response(raw)
        response.response_metadata["request"] = {
            "model": self.model,
            "max_tokens_param": "max_tokens" if "max_tokens" in kwargs else self.max_tokens_param,
            "max_tokens": kwargs.get("max_tokens", eff_mt),
            "reasoning": (kwargs.get("extra_body") or {}).get("reasoning"),
            "provider": (kwargs.get("extra_body") or {}).get("provider"),
            "temperature": eff_temp,
        }
        return response

    # ── Streaming ────────────────────────────────────────────────────────

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[StreamDelta]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = tools
            if self.parallel_tool_calls is not None:
                kwargs["parallel_tool_calls"] = self.parallel_tool_calls
        eff_temp = temperature if temperature is not None else self.default_temperature
        if eff_temp is not None:
            kwargs["temperature"] = eff_temp
        eff_mt = max_tokens if max_tokens is not None else self.default_max_tokens
        if eff_mt is not None:
            kwargs[self.max_tokens_param] = eff_mt
        if extra_headers:
            kwargs["extra_headers"] = extra_headers
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body
        if timeout is not None:
            kwargs["timeout"] = timeout
        elif self.default_timeout is not None:
            kwargs["timeout"] = self.default_timeout

        kwargs = request_policy(self._client.base_url, kwargs, sdk=True)
        async for chunk in await self._client.chat.completions.create(**kwargs):
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            yield StreamDelta(
                content=field(delta, "content", None) or "",
                reasoning_content=reasoning_text(delta),
                tool_call_deltas=_tool_call_deltas(field(delta, "tool_calls", None)),
            )


# ── Adapters ─────────────────────────────────────────────────────────────


def _to_llm_response(raw: Any) -> LLMResponse:
    """Convert an OpenAI ``ChatCompletion`` into an :class:`LLMResponse`."""
    choice = field(raw, "choices")[0]
    msg = field(choice, "message")
    tool_calls: list[ToolCall] = []
    for tc in (field(msg, "tool_calls", None) or []):
        tool_calls.append({
            "type": "function",
            "id": field(tc, "id"),
            "function": {
                "name": field(field(tc, "function"), "name"),
                "arguments": field(field(tc, "function"), "arguments") or "{}",
            },
        })
    usage = field(raw, "usage", None)
    usage_dict: dict[str, int] = {}
    if usage:
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            v = field(usage, k, None)
            if v is not None:
                usage_dict[k] = int(v)
        # Prompt-caching surfaced under ``usage.prompt_tokens_details.cached_tokens``
        # on modern OpenAI / OpenRouter responses.
        ptd = field(usage, "prompt_tokens_details", None)
        cached = field(ptd, "cached_tokens", None) if ptd is not None else None
        if cached is not None:
            usage_dict["cached_tokens"] = int(cached)
    details = field(usage, "completion_tokens_details")
    reasoning_tokens = field(details, "reasoning_tokens")
    if reasoning_tokens is not None:
        usage_dict["reasoning_tokens"] = int(reasoning_tokens)
    return LLMResponse(
        content=field(msg, "content", "") or "",
        tool_calls=tool_calls,
        reasoning_content=reasoning_text(msg),
        finish_reason=field(choice, "finish_reason", "") or "",
        model=field(raw, "model", "") or "",
        usage=usage_dict,
        response_metadata={
            "id": field(raw, "id", ""),
            # Response body only: never persist request headers or credentials.
            "raw_response": raw.model_dump(mode="json") if hasattr(raw, "model_dump") else raw if isinstance(raw, dict) else None,
        },
    )


def _tool_call_deltas(raw: Any) -> list[dict[str, Any]]:
    if not raw:
        return []
    out: list[dict[str, Any]] = []
    for tc in raw:
        out.append({
            "index": getattr(tc, "index", 0),
            "id": getattr(tc, "id", None),
            "name": getattr(getattr(tc, "function", None), "name", None),
            "arguments": getattr(getattr(tc, "function", None), "arguments", None),
        })
    return out


__all__ = ["OpenAIClient"]
