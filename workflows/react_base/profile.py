"""react_base workflow profile - loads YAML config and builds an LLM.

Model descriptors and agent profiles live in ``workflows/react_base/profiles/``.
Provider connection settings live in ``workflows/react_base/providers.yaml``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from agent_harness.core.llm import LLMClient

logger = logging.getLogger(__name__)

_PROFILES_DIR = Path(__file__).resolve().parent / "profiles"
_PROVIDERS_CONFIG_PATH = Path(__file__).resolve().parent / "providers.yaml"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_MODEL_CONFIG = "gpt_5_mini.yaml"
_MODEL_DESCRIPTOR_KEYS = {
    "model",
    "provider",
    "summary_model",
    "summary_provider",
}


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    import yaml

    if not path.exists():
        raise FileNotFoundError(f"react_base config not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"react_base config must be a mapping: {path}")
    return raw


def _model_descriptor_path(selection: str) -> Path:
    path = Path(selection)
    if path.is_absolute():
        return path
    if path.parent == Path("."):
        return _PROFILES_DIR / path
    return _PROJECT_ROOT / path


def _validate_model_descriptor(
    descriptor: dict[str, Any],
    path: Path,
) -> dict[str, Any]:
    if not descriptor.get("model") or not descriptor.get("provider"):
        raise ValueError(
            f"react_base model descriptor must contain 'model' and "
            f"'provider': {path}"
        )
    unexpected = set(descriptor) - _MODEL_DESCRIPTOR_KEYS
    if unexpected:
        keys = ", ".join(sorted(unexpected))
        raise ValueError(
            f"react_base model descriptor has unsupported keys ({keys}): {path}"
        )
    has_summary_model = bool(descriptor.get("summary_model"))
    has_summary_provider = bool(descriptor.get("summary_provider"))
    if has_summary_model != has_summary_provider:
        raise ValueError(
            "react_base model descriptor must define 'summary_model' and "
            f"'summary_provider' together: {path}"
        )
    return descriptor


def load_profile(name: str) -> dict[str, Any]:
    """Compose a model descriptor, provider settings, and agent profile."""
    from dotenv import load_dotenv
    from agent_harness.infra.config import _resolve_env_vars

    load_dotenv(_PROJECT_ROOT / ".env", override=False)

    path = _PROFILES_DIR / f"{name}.yaml"
    profile_raw = _load_yaml_mapping(path)
    if set(profile_raw) <= _MODEL_DESCRIPTOR_KEYS:
        model_path = path
        model_cfg = _validate_model_descriptor(profile_raw, model_path)
        profile_raw = _load_yaml_mapping(_PROFILES_DIR / "default.yaml")
    else:
        selected_config = os.getenv(
            "REACT_BASE_MODEL_CONFIG",
            _DEFAULT_MODEL_CONFIG,
        )
        model_path = _model_descriptor_path(selected_config)
        model_cfg = _validate_model_descriptor(
            _load_yaml_mapping(model_path),
            model_path,
        )

    provider_raw = _resolve_env_vars(
        _load_yaml_mapping(_PROVIDERS_CONFIG_PATH)
    )
    defaults = provider_raw.get("defaults", {})
    providers = provider_raw.get("providers")
    if not isinstance(defaults, dict):
        raise ValueError(
            f"react_base provider defaults must be a mapping: "
            f"{_PROVIDERS_CONFIG_PATH}"
        )
    if not isinstance(providers, dict):
        raise ValueError(
            f"react_base providers must be a mapping: {_PROVIDERS_CONFIG_PATH}"
        )

    def resolve_model(model: str, provider: str) -> dict[str, Any]:
        provider_cfg = providers.get(provider)
        if not isinstance(provider_cfg, dict):
            available = ", ".join(sorted(str(key) for key in providers))
            raise ValueError(
                f"unknown react_base provider {provider!r}; available: "
                f"{available}"
            )
        return {
            **defaults,
            **provider_cfg,
            "model": model,
            "provider": provider,
        }

    # Preserve llm overrides in older behavior profiles while model descriptor
    # files stay restricted to model/provider selection.
    profile_llm = profile_raw.pop("llm", {})
    if not isinstance(profile_llm, dict):
        raise ValueError(f"react_base profile 'llm' must be a mapping: {path}")
    summary_model = model_cfg.get("summary_model", model_cfg["model"])
    summary_provider = model_cfg.get("summary_provider", model_cfg["provider"])
    merged = {
        **profile_raw,
        "llm": {
            **resolve_model(model_cfg["model"], model_cfg["provider"]),
            **profile_llm,
        },
        "summary_llm": resolve_model(summary_model, summary_provider),
    }
    merged = _resolve_env_vars(merged)
    from agent_harness.infra.openrouter_policy import applies, request_policy
    import json
    for section in ("llm", "summary_llm"):
        cfg = merged[section]
        if applies(cfg.get("base_url"), cfg.get("model")):
            extra = {**json.loads(os.getenv("OPENAI_EXTRA_BODY") or "{}"), **(cfg.get("extra_body") or {})}
            wire = request_policy(cfg["base_url"], {"model": cfg["model"], "extra_body": extra}, sdk=True)
            cfg.update(model=wire["model"], max_tokens=wire["max_tokens"], max_tokens_param="max_tokens", extra_body=wire["extra_body"])
    return merged


def _clean_tool_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Strip per-property metadata absent from FastMCP wire shape:
    drop ``title`` and ``description`` at every level, rename
    ``anyOf`` → ``oneOf``.
    """
    import copy

    schema = copy.deepcopy(schema)
    properties = schema.get("properties") if isinstance(schema, dict) else None

    def _walk(node: Any, *, in_properties: bool) -> None:
        if not isinstance(node, dict):
            return
        node.pop("title", None)
        if in_properties:
            node.pop("description", None)
        if "anyOf" in node:
            node["oneOf"] = node.pop("anyOf")
        for k, v in node.items():
            if isinstance(v, dict):
                _walk(v, in_properties=(k == "properties") or in_properties)
            elif isinstance(v, list):
                for item in v:
                    _walk(item, in_properties=in_properties)

    _walk(schema, in_properties=False)
    if isinstance(properties, dict):
        for v in properties.values():
            if isinstance(v, dict):
                v.pop("description", None)
    return schema


def _collapse_optional_unions(schema: dict[str, Any]) -> dict[str, Any]:
    """Collapse ``oneOf: [T, null]`` → ``{type: <T>, default: null, ...}``.

    FastMCP emits ``Optional[T]`` as a single-type schema with
    ``default: null`` (not a union). Non-null unions (e.g.
    ``Union[str, list[str]]``) are left as-is.
    """
    import copy
    schema = copy.deepcopy(schema)

    def _walk(node: Any) -> Any:
        if isinstance(node, dict):
            for k, v in list(node.items()):
                node[k] = _walk(v)
            branches = node.get("oneOf")
            if isinstance(branches, list) and len(branches) == 2:
                null_b: dict | None = None
                other_b: dict | None = None
                for b in branches:
                    if (
                        isinstance(b, dict) and b.get("type") == "null"
                        and len(b) == 1
                    ):
                        null_b = b
                    else:
                        other_b = b if isinstance(b, dict) else None
                if null_b is not None and other_b is not None:
                    siblings = {
                        k: v for k, v in node.items() if k != "oneOf"
                    }
                    return {**other_b, **siblings}
            return node
        if isinstance(node, list):
            return [_walk(item) for item in node]
        return node

    return _walk(schema)


def _sort_keys_alphabetical(obj: Any) -> Any:
    """Sort dict keys alphabetically at every depth. List order preserved."""
    if isinstance(obj, dict):
        return {k: _sort_keys_alphabetical(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, list):
        return [_sort_keys_alphabetical(item) for item in obj]
    return obj


def _raw_docstring_with_indent(func: Any) -> str:
    """Return ``func``'s docstring preserving original source indentation.

    Uses ``ast.get_docstring(node, clean=False)`` to reach past any
    decorator that may have already dedented ``__doc__``.
    """
    import ast
    import inspect
    import textwrap
    try:
        src = inspect.getsource(func)
    except (OSError, TypeError):
        return (func.__doc__ or "").rstrip()
    try:
        node = ast.parse(textwrap.dedent(src)).body[0]
    except (SyntaxError, IndexError):
        return (func.__doc__ or "").rstrip()
    doc = ast.get_docstring(node, clean=False)
    return (doc or "").rstrip()


def _build_fastmcp_style_tool_spec(tool: Any) -> dict[str, Any]:
    """Build an OpenAI tool spec in FastMCP shape.

    - Description = the function's RAW docstring (preserving indent).
    - ``Optional[T]`` params collapsed to ``{type:<T>, default:null}``.
    - All dict keys sorted alphabetically for stable byte layout.
    """
    if isinstance(tool, dict):
        spec = dict(tool)
        underlying = None
    else:
        schema_fn = getattr(tool, "to_openai_schema", None)
        if not callable(schema_fn):
            return tool
        spec = schema_fn()
        underlying = getattr(tool, "func", None)

    fn = spec.get("function") or {}
    if not isinstance(fn, dict):
        fn = {}

    if underlying is not None:
        raw_doc = _raw_docstring_with_indent(underlying)
        if raw_doc:
            fn["description"] = raw_doc

    params = fn.get("parameters")
    if isinstance(params, dict):
        params = _clean_tool_schema(params)
        params = _collapse_optional_unions(params)
        fn["parameters"] = params

    return _sort_keys_alphabetical({"function": fn, "type": "function"})


def clean_tool_schemas(tools: list[Any]) -> list[dict[str, Any]]:
    """Rebuild tool schemas in FastMCP form for wire transmission."""
    return [_build_fastmcp_style_tool_spec(t) for t in tools]


def apply_fastmcp_schemas(tools: list[Any]) -> None:
    """Pin each Tool's wire schema to the FastMCP-shaped spec, in place."""
    for t in tools:
        if hasattr(t, "metadata"):
            t.metadata["openai_schema"] = _build_fastmcp_style_tool_spec(t)


class _ReasoningContentInliner(LLMClient):
    """Wraps an :class:`LLMClient` and inlines ``reasoning_content`` into
    ``content`` as a ``{reasoning}</think>\\n\\n{text}`` block.

    The SGLang chat template auto-prepends ``<think>\\n`` to assistant
    generation, so the response arrives as ``{reasoning}</think>\\n\\n{text}``
    *without* a leading opening tag. We mirror that shape so history
    matches the training distribution.
    """

    def __init__(self, inner: LLMClient) -> None:
        self.inner = inner
        self.model = inner.model

    @staticmethod
    def _inline(response: Any) -> Any:
        rc = getattr(response, "reasoning_content", "") or ""
        if not rc:
            return response
        if "</think>" in rc:
            rc = rc.replace("</think>", "</ think>")
        existing = response.content if isinstance(response.content, str) else ""
        response.content = (
            f"{rc}</think>\n\n{existing}" if existing else f"{rc}</think>\n\n"
        )
        return response

    async def chat(self, messages, **kwargs):
        return self._inline(await self.inner.chat(messages, **kwargs))

    async def stream(self, messages, **kwargs):
        async for delta in self.inner.stream(messages, **kwargs):
            yield delta


def create_llm(profile: dict[str, Any]) -> LLMClient:
    """Build an OpenAI-compatible :class:`LLMClient` from a profile dict.

    ``thinking_format: tag`` wraps the client with :class:`_ReasoningContentInliner`
    so SGLang ``reasoning_content`` ends up in the visible content as a
    ``</think>``-bounded block.
    """
    from agent_harness.infra.openai_client import OpenAIClient

    cfg = profile["llm"]

    temperature = cfg.get("temperature", 1.0)
    top_p = cfg.get("top_p", 0.95)
    max_tokens = cfg.get("max_tokens", 32_768)
    repetition_penalty = cfg.get("repetition_penalty", 1.05)

    extra_body: dict[str, Any] = dict(cfg.get("extra_body") or {})
    if "top_p" not in extra_body:
        extra_body["top_p"] = top_p
    if repetition_penalty != 1.0 and "repetition_penalty" not in extra_body:
        extra_body["repetition_penalty"] = repetition_penalty

    client = OpenAIClient(
        model=cfg["model"],
        api_key=cfg.get("api_key", "dummy"),
        base_url=cfg.get("base_url"),
        temperature=temperature,
        max_completion_tokens=max_tokens,
        max_tokens_param=cfg.get(
            "max_tokens_param",
            "max_completion_tokens",
        ),
        parallel_tool_calls=cfg.get("parallel_tool_calls", True),
        default_headers={
            "HTTP-Referer": "agent_harness",
            "X-Title": "AgentHarness-ProfileReAct",
        },
        extra_body=extra_body or None,
    )

    if _resolve_thinking_format(profile) == "tag":
        return _ReasoningContentInliner(client)
    return client


def _resolve_thinking_format(profile: dict[str, Any]) -> str:
    """Model config > legacy agent override > model-id inference."""
    from agent_harness.core.runtime.loop.model_profile import (
        infer_thinking_format,
    )
    explicit = (profile.get("llm") or {}).get("thinking_format")
    if not explicit:
        explicit = (profile.get("agent") or {}).get("thinking_format")
    if explicit:
        return explicit
    model_id = (profile.get("llm") or {}).get("model")
    return infer_thinking_format(model_id, default="tag")


def build_model_profile(profile: dict[str, Any]) -> Any:
    """Build a kernel ``ModelProfile`` from a profile dict."""
    from agent_harness.core.runtime.loop.model_profile import ModelProfile
    return ModelProfile(
        model_id=profile["llm"]["model"],
        provider=profile["llm"]["provider"],
        thinking_format=_resolve_thinking_format(profile),
    )
