"""OpenRouter DeepSeek wire policy; other routes are unchanged."""
from copy import deepcopy
from urllib.parse import urlparse
from typing import Any

FLASH_MODEL = "deepseek/deepseek-v4.1-flash"
MAX_TOKENS = 65536


def applies(base_url: Any, model: Any) -> bool:
    return (
        urlparse(str(base_url or "")).hostname == "openrouter.ai"
        and str(model).lower().startswith(("deepseek/", "deepseek-"))
    )


def request_policy(base_url: Any, parameters: dict[str, Any], *, sdk: bool = False) -> dict[str, Any]:
    # SDK extra_body wins on the wire: inspect its effective model as well.
    extra = (parameters.get("extra_body") or {}) if sdk else {}
    model = extra.get("model", parameters.get("model", ""))
    if not applies(base_url, model):
        return parameters
    result = deepcopy(parameters)
    body = (result.pop("extra_body", {}) or {}) if sdk else result
    model = str(model)
    if model in {"deepseek-v4-flash", "deepseek/deepseek-v4-flash", FLASH_MODEL}:
        model = FLASH_MODEL
    for target in (result, body):
        for key in ("max_completion_tokens", "max_tokens", "reasoning_effort", "reasoning", "thinking", "include_reasoning"):
            target.pop(key, None)
    # The Flash route does not advertise these parameters. With strict
    # provider matching, sending either leaves no eligible endpoint. Strip
    # both SDK kwargs and extra_body, which otherwise wins on the wire.
    if model == FLASH_MODEL:
        for target in (result, body):
            target.pop("parallel_tool_calls", None)
            target.pop("repetition_penalty", None)
    body.pop("model", None)
    result["model"] = model
    result["max_tokens"] = MAX_TOKENS
    body["reasoning"] = {"effort": "high"}
    body["provider"] = {
        **((result.pop("provider", {}) or {}) if sdk else {}),
        **(body.get("provider") or {}),
        "require_parameters": True,
    }
    if sdk:
        result["extra_body"] = body
    return result


def field(value, name, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def reasoning_text(message):
    return field(message, "reasoning") or field(message, "reasoning_content") or ""
