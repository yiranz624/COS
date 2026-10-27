"""Workflow-agnostic ``LoopObserver`` that records the agent loop to disk.

Two output formats, both written by default:

* ``json``  — OpenAI message-format envelope. Rewritten atomically after
              every event (``.tmp`` + ``os.replace``) so a mid-loop crash
              still leaves a valid snapshot.
* ``jsonl`` — Append-only event log (one JSON object per line, flushed
              after each write so partial files remain valid).

Override the default at construction or via the ``TRAJECTORY_FORMATS`` env
var (comma-separated subset of ``json,jsonl``). Adding a new format is one
new ``_write_<format>`` helper plus an entry in ``_FORMATS``.

JSONL event types: ``start`` / ``llm`` / ``result`` / ``end``. Create one
instance per loop — both writers stash state on ``self``.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import IO, Any

from agent_harness.core.loop_types import (
    AgentLoopResult,
    BaseObserver,
    Intervention,
    LoopConfig,
    ToolResult,
    TurnContext,
)

_FORMATS: tuple[str, ...] = ("json", "jsonl")
_DEFAULT_FORMATS: tuple[str, ...] = _FORMATS


def _resolve_formats(arg: Iterable[str] | None) -> set[str]:
    """Resolve enabled formats: explicit arg → env var → default."""
    if arg is not None:
        return {f for f in arg if f in _FORMATS}
    env = os.getenv("TRAJECTORY_FORMATS", "")
    if env:
        return {f.strip() for f in env.split(",") if f.strip() in _FORMATS}
    return set(_DEFAULT_FORMATS)


class TrajectoryFileObserver(BaseObserver):
    """Saves an agent's main / sub-agent traces in one or more formats."""

    # Trace persistence is awaited so records are ordered and flushed before
    # the runtime advances to the next lifecycle step.
    critical: bool = True

    def __init__(
        self,
        output_dir: Path,
        *,
        filename: str | None = None,
        formats: Iterable[str] | None = None,
        tools: list[Any] | None = None,
        model_name: str | None = None,
        system_prompt: str | None = None,
        user_message: str | None = None,
    ) -> None:
        """Args:
            output_dir: Directory where trajectory file(s) are written.
            filename: Output stem (no extension). Defaults to the loop's
                ``task_id``; sub-agents pass their session name.
            formats: Subset of ``("json", "jsonl")``. ``None`` → env var
                ``TRAJECTORY_FORMATS`` → both.
            tools: Tools bound to the LLM. Serialized to OpenAI tool
                schema and emitted on the JSON envelope; names are also
                recorded in the JSONL ``start`` event.
            model_name: Recorded on both formats for post-run rendering.
            system_prompt: Recorded on the JSONL ``start`` event so
                renderers can reconstruct the full message history without
                re-reading the JSON envelope.
            user_message: Same purpose as ``system_prompt``.
        """
        self._dir = output_dir
        self._filename = filename
        self._formats = _resolve_formats(formats)
        self._tools_schema: list[dict] = self._serialize_tools(tools or [])
        self._tool_names: list[str] = [
            (t.get("function", {}) or {}).get("name", "")
            for t in self._tools_schema
            if isinstance(t, dict)
        ]
        self._tool_names = [n for n in self._tool_names if n]
        self._model_name = model_name or ""
        self._system_prompt = system_prompt or ""
        self._user_message = user_message or ""
        self._task_id: str = ""
        self._role_id: str = ""
        self._max_turns: int = 0
        self._turns_used: int = 0
        self._tool_calls_count: int = 0
        self._stopped_by: str = ""

        # JSON envelope state
        self._messages: list[dict] = []
        self._emitted_prior: bool = False
        self._tool_call_ids: dict[int, list[str]] = {}
        self._tool_results_seen: dict[int, int] = {}

        # JSONL append state
        self._jsonl_handle: IO[str] | None = None
        self._sequence: int = 0

    # ── Path helpers ────────────────────────────────────────────────────

    @staticmethod
    def _safe(stem: str) -> str:
        return stem.replace("/", "__").replace(":", "_")

    @staticmethod
    def _serialize_tools(tools: list[Any]) -> list[dict]:
        """Convert :class:`Tool` instances / dicts to OpenAI tool schema.

        Best effort: tools without ``to_openai_schema`` fall through to
        a name + description stub.
        """
        out: list[dict] = []
        for t in tools:
            if isinstance(t, dict):
                if "function" in t and "type" in t:
                    out.append(t)
                elif "name" in t:
                    out.append({"type": "function", "function": t})
                continue
            schema_fn = getattr(t, "to_openai_schema", None)
            if callable(schema_fn):
                try:
                    out.append(schema_fn())
                    continue
                except Exception:  # noqa: BLE001
                    pass
            out.append({
                "type": "function",
                "function": {
                    "name": getattr(t, "name", None) or t.__class__.__name__,
                    "description": getattr(t, "description", "") or "",
                    "parameters": getattr(t, "parameters", {}) or {},
                },
            })
        return out

    def _path(self, ext: str) -> Path:
        self._dir.mkdir(parents=True, exist_ok=True)
        stem = self._filename or self._task_id or "trace"
        return self._dir / f"{self._safe(stem)}.{ext}"

    # ── JSONL writer ────────────────────────────────────────────────────

    def _write_jsonl(self, record: dict) -> None:
        if "jsonl" not in self._formats:
            return
        if self._jsonl_handle is None:
            self._jsonl_handle = open(  # noqa: SIM115
                self._path("jsonl"), "a", encoding="utf-8",
            )
        self._sequence += 1
        record.setdefault("sequence", self._sequence)
        record.setdefault("ts", time.time())
        self._jsonl_handle.write(
            json.dumps(record, ensure_ascii=False) + "\n",
        )
        self._jsonl_handle.flush()

    def _close_jsonl(self) -> None:
        if self._jsonl_handle is not None:
            self._jsonl_handle.close()
            self._jsonl_handle = None

    # ── JSON envelope writer ────────────────────────────────────────────

    def _flush_json(self) -> None:
        if "json" not in self._formats:
            return
        path = self._path("json")
        payload: dict = {
            "task_id": self._task_id,
            "role_id": self._role_id,
            "max_turns": self._max_turns,
            "turns_used": self._turns_used,
            "tool_calls_count": self._tool_calls_count,
            "stopped_by": self._stopped_by,
            "tools": self._tools_schema,
            "messages": self._messages,
        }
        if self._model_name:
            payload["model_name"] = self._model_name
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, path)

    @staticmethod
    def _synth_id(turn: int, idx: int) -> str:
        return f"call_{turn}_{idx}"

    @staticmethod
    def _stringify(value: Any) -> str:
        if isinstance(value, str):
            return value
        return str(value) if value is not None else ""

    @staticmethod
    def _format_args(args: Any) -> str:
        if isinstance(args, (dict, list)):
            return json.dumps(args, ensure_ascii=False)
        return TrajectoryFileObserver._stringify(args)

    def _message_to_dict(self, m: Any) -> dict | None:
        """Pass through dict messages (already OpenAI shape)."""
        if isinstance(m, dict) and m.get("role"):
            return dict(m)
        return None

    # ── Lifecycle hooks ─────────────────────────────────────────────────

    async def on_loop_start(self, config: LoopConfig) -> None:
        self._task_id = config.task_id
        self._role_id = config.role_id
        self._max_turns = config.max_turns

        start_record: dict = {
            "t": "start",
            "事件": "Agent循环开始",
            "说明": "初始化Agent、模型、提示词和运行限制",
            "task_id": config.task_id,
            "role_id": config.role_id,
            "max_turns": config.max_turns,
        }
        if self._model_name:
            start_record["model_name"] = self._model_name
        if self._system_prompt:
            start_record["system_prompt"] = self._system_prompt
        if self._user_message:
            start_record["user_message"] = self._user_message
        if self._tool_names:
            start_record["tool_names"] = self._tool_names
            start_record["可用工具定义"] = self._tools_schema
        self._write_jsonl(start_record)
        self._flush_json()

    async def on_llm_response(
        self, ctx: TurnContext,
    ) -> Intervention | None:
        # JSONL: append one event
        record: dict = {
            "t": "llm",
            "事件": "模型调用完成",
            "说明": "记录本次模型实际接收的信息和模型返回的信息",
            "turn": ctx.turn,
            "轮次": ctx.turn,
            "模型输入消息": ctx.llm_input_messages,
            "content": ctx.ai_text,
            "模型输出文本": ctx.ai_text,
            "模型调用耗时毫秒": ctx.llm_duration_ms,
        }
        if ctx.llm_tools:
            record["本次可用工具定义"] = ctx.llm_tools
        if ctx.tool_calls:
            record["tool_calls"] = [
                {
                    "id": tc.get("id", ""),
                    "name": tc.get("name"),
                    "args": tc.get("args", {}),
                }
                for tc in ctx.tool_calls
            ]
            record["模型请求的工具调用"] = record["tool_calls"]
        if ctx.rejected_tool_calls:
            rejected = [
                {
                    "id": tc.get("id", ""),
                    "name": tc.get("name"),
                    "args": tc.get("args", {}),
                }
                for tc in ctx.rejected_tool_calls
            ]
            record["最终轮被拒绝的工具调用"] = rejected
            record["工具调用拒绝原因"] = "最终轮仅允许返回结果，未执行工具"
        if ctx.thinking:
            record["thinking"] = ctx.thinking
            record["模型思考内容"] = ctx.thinking
        if ctx.usage:
            record["usage"] = ctx.usage
            record["Token用量"] = ctx.usage
        self._write_jsonl(record)

        # JSON envelope: emit prior context once, then this turn.
        if "json" in self._formats:
            if not self._emitted_prior:
                # ctx.messages includes the assistant we just got; skip it
                # and emit ourselves below with reasoning_content split out.
                for m in list(ctx.messages or [])[:-1]:
                    d = self._message_to_dict(m)
                    if d:
                        self._messages.append(d)
                self._emitted_prior = True

            msg: dict = {
                "role": "assistant",
                "content": self._stringify(ctx.ai_text),
            }
            if ctx.thinking:
                msg["reasoning_content"] = self._stringify(ctx.thinking)
            if ctx.tool_calls:
                ids: list[str] = []
                tcs: list[dict] = []
                for idx, tc in enumerate(ctx.tool_calls):
                    cid = tc.get("id") or self._synth_id(ctx.turn, idx)
                    ids.append(cid)
                    tcs.append({
                        "id": cid,
                        "type": "function",
                        "function": {
                            "name": tc.get("name", ""),
                            "arguments": self._format_args(
                                tc.get("args", {}),
                            ),
                        },
                    })
                msg["tool_calls"] = tcs
                self._tool_call_ids[ctx.turn] = ids
                self._tool_results_seen[ctx.turn] = 0
                self._tool_calls_count += len(tcs)

            self._messages.append(msg)
            self._turns_used = ctx.turn
            self._flush_json()
        return None

    async def on_llm_attempt(self, event: dict[str, Any]) -> None:
        status = str(event.get("状态", ""))
        descriptions = {
            "开始": "准备向模型服务发送一次实际API请求",
            "成功": "本次模型API请求成功返回",
            "失败": "本次模型API请求失败，运行时可能按策略重试",
        }
        self._write_jsonl({
            "t": "llm_attempt",
            "事件": f"模型调用尝试{status}",
            "说明": descriptions.get(status, "模型调用尝试状态变化"),
            **event,
        })

    async def on_tool_call(
        self, ctx: TurnContext, tool_call: dict,
    ) -> None:
        """Write a call event only when the model actually requested a tool."""
        self._write_jsonl({
            "t": "tool_call",
            "事件": "工具调用开始",
            "说明": "模型请求执行工具，以下为实际工具名称和输入参数",
            "turn": ctx.turn,
            "轮次": ctx.turn,
            "tool_call_id": tool_call.get("id", ""),
            "工具名称": tool_call.get("name", ""),
            "工具输入参数": tool_call.get("args", {}) or {},
        })
        return None

    async def on_tool_result(
        self, ctx: TurnContext, result: ToolResult,
    ) -> Intervention | None:
        self._write_jsonl({
            "t": "result",
            "事件": "工具调用完成",
            "说明": "记录工具实际返回的信息、耗时和错误状态",
            "turn": ctx.turn,
            "name": result.name,
            "result": result.result,
            "error": result.is_error,
            "ms": result.duration_ms,
            "tool_call_id": result.tool_call_id,
            "工具名称": result.name,
            "工具输入参数": result.args,
            "工具返回信息": result.result,
            "是否错误": result.is_error,
            "工具耗时毫秒": result.duration_ms,
        })

        if "json" in self._formats:
            cid = getattr(result, "tool_call_id", "") or ""
            if not cid:
                ids = self._tool_call_ids.get(ctx.turn, [])
                seen = self._tool_results_seen.get(ctx.turn, 0)
                cid = (
                    ids[seen] if seen < len(ids)
                    else self._synth_id(ctx.turn, seen)
                )
                self._tool_results_seen[ctx.turn] = seen + 1
            body = self._stringify(result.result)
            self._messages.append({
                "role": "tool",
                "tool_call_id": cid,
                "content": f"[error] {body}" if result.is_error else body,
            })
            self._flush_json()
        return None

    async def on_loop_end(self, result: AgentLoopResult) -> None:
        self._turns_used = max(self._turns_used, result.turns_used)
        self._tool_calls_count = result.tool_calls_count
        self._stopped_by = (
            getattr(result, "stopped_by", "") or self._stopped_by
        )

        self._write_jsonl({
            "t": "end",
            "事件": "Agent循环结束",
            "说明": "记录最终停止原因和累计轮次、工具调用次数",
            "turns": result.turns_used,
            "tool_calls": result.tool_calls_count,
            "stopped_by": result.stopped_by,
            "累计轮次": result.turns_used,
            "累计工具调用次数": result.tool_calls_count,
            "停止原因": result.stopped_by,
            "最终输出": result.final_content,
        })
        self._close_jsonl()
        self._flush_json()
