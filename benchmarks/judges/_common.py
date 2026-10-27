"""Shared judge infrastructure — LLM client, retry helpers, type aliases.

The judge endpoint is read from ``JUDGE_API_KEY`` / ``JUDGE_BASE_URL`` env
(falling back to ``OPENAI_*``). Judge models and reasoning effort are selected
by the per-family modules. The BrowseComp-Plus semantic judge model can be
overridden with ``SEMANTIC_JUDGE_MODEL``.
"""

from __future__ import annotations

import asyncio
import logging
import json
import os
from collections.abc import Awaitable, Callable
from typing import Literal

from openai import AsyncOpenAI
from agent_harness.infra.openrouter_policy import request_policy, applies

logger = logging.getLogger(__name__)

Verdict = Literal["CORRECT", "INCORRECT", "NOT_ATTEMPTED"]
JudgeFn = Callable[[str, str, str], Awaitable[Verdict]]

_DEFAULT_JUDGE_MODEL = "gpt-4.1-2025-04-14"


def _make_client() -> AsyncOpenAI:
    api_key = os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("JUDGE_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    if base_url:
        base_url = base_url.rstrip("/").removesuffix("/chat/completions")
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


def _build_judge_kwargs(
    prompt: str, *, max_completion_tokens: int,
    model_override: str | None = None,
    default_model: str | None = None,
    default_reasoning_effort: str | None = None,
) -> dict:
    """Common kwargs for ``chat.completions.create`` on judge calls.

    Model resolution: ``model_override`` (call-site pin) > ``default_model``
    (per-benchmark default) > ``_DEFAULT_JUDGE_MODEL`` (global fallback).
    Reasoning effort: ``default_reasoning_effort`` if supplied, else unset.
    """
    model = model_override or default_model or _DEFAULT_JUDGE_MODEL
    kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens": max_completion_tokens,
    }
    if default_reasoning_effort:
        kwargs["reasoning_effort"] = default_reasoning_effort
    return kwargs


def _model_supports_reasoning_effort(model: str) -> bool:
    """Whether the model accepts the ``reasoning_effort`` kwarg.

    OpenAI o-series (o1 / o3 / o4) and gpt-5 family accept it; classic
    chat models (gpt-4.1 / gpt-4o / gpt-4) reject it with a 400.
    """
    m = (model or "").lower()
    if m.startswith("openai/"):
        m = m[len("openai/"):]
    if m.startswith(("o1", "o3", "o4")):
        return True
    return m.startswith("gpt-5")


def _effort_chain(model: str, initial: str) -> list[str]:
    """Cascade of ``reasoning_effort`` levels to try (``initial`` → medium → low).

    For non-reasoning models returns ``[""]`` so the kwarg is never sent.
    """
    if not _model_supports_reasoning_effort(model):
        return [""]
    chain: list[str] = []
    for e in [(initial or "").lower(), "medium", "low"]:
        if e and e not in chain:
            chain.append(e)
    return chain or [""]


async def _judge_call_with_effort_fallback(kwargs: dict, *, label: str) -> str:
    """Run ``chat.completions.create`` with ``reasoning_effort`` cascade.

    Cascades the initial effort (from ``kwargs["reasoning_effort"]`` if set)
    through medium → low, retrying once at each level. Returns the completion
    string, or ``""`` if every attempt yielded empty content or errored.
    """
    initial = (kwargs.get("reasoning_effort") or "").lower()
    chain = _effort_chain(kwargs.get("model", ""), initial)
    content = ""
    for attempt_effort in chain:
        if attempt_effort:
            kwargs["reasoning_effort"] = attempt_effort
        else:
            kwargs.pop("reasoning_effort", None)
        for attempt_idx in range(2):
            try:
                client = _make_client()
                policy_kwargs = kwargs
                if applies(client.base_url, kwargs.get("model")):
                    policy_kwargs = {**kwargs, "extra_body": {**json.loads(os.getenv("OPENAI_EXTRA_BODY") or "{}"), **(kwargs.get("extra_body") or {})}}
                wire_kwargs = request_policy(client.base_url, policy_kwargs, sdk=True)
                resp = await client.chat.completions.create(**wire_kwargs)
            except Exception as exc:  # noqa: BLE001
                # API error (rate limit, 5xx, network). Sleep + continue so
                # the surrounding effort×attempt loop actually retries; the
                # earlier ``return ""`` here defeated the retry budget by
                # giving up after a single transient failure.
                logger.warning("%s judge call error (%s, effort=%s, attempt=%d): %s",
                               label, kwargs.get("model"), attempt_effort, attempt_idx, exc)
                await asyncio.sleep(1.0)
                continue
            content = (resp.choices[0].message.content or "").strip()
            if content:
                if attempt_effort != initial or attempt_idx > 0:
                    logger.info(
                        "%s judge: empty at effort=%s, recovered at effort=%s (retry=%d)",
                        label, initial, attempt_effort, attempt_idx,
                    )
                return content
            await asyncio.sleep(1.0)
    return content
