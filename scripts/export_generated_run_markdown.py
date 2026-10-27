#!/usr/bin/env python3
"""Export one generated-code benchmark run as a complete readable Markdown record."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def fenced(value: Any, language: str = "text") -> str:
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    value = value.replace("```", "''' ")
    return f"```{language}\n{value or '（空）'}\n```"


def one_line(value: Any) -> str:
    return " ".join(str(value or "").replace("|", "&#124;").split())


def trial_sort(path: Path) -> tuple[int, str]:
    return (int(path.name), path.name) if path.name.isdigit() else (10**9, path.name)


def trace_sort(path: Path) -> tuple[int, int, str]:
    match = re.search(r"_job_(\d+)_", path.name)
    job = int(match.group(1)) if match else 10**9
    variant = 0
    if "forced_final" in path.stem:
        variant = 1
    elif "output_repair" in path.stem:
        variant = 2
    return job, variant, path.name


def job_timings(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event.get("job_id"):
            grouped[str(event["job_id"])].append(event)
    output: dict[str, dict[str, Any]] = {}
    for job_id, rows in grouped.items():
        start = next((row for row in rows if row.get("事件类型") == "job_started"), None)
        end = next((
            row for row in reversed(rows)
            if row.get("事件类型") in {
                "job_succeeded", "job_failed", "job_timed_out", "job_cancelled",
            }
        ), None)
        created = next((row for row in rows if row.get("事件类型") == "job_created"), {})
        output[job_id] = {
            "key": created.get("key", ""),
            "status": (end or {}).get("事件类型", "未知"),
            "duration": (
                float(end["timestamp"]) - float(start["timestamp"])
                if start and end else None
            ),
            "failure": (end or {}).get("详细信息", {}),
        }
    return output


def find_job_id(path: Path, events: list[dict[str, Any]]) -> str:
    match = re.search(r"_job_(\d+)_", path.name)
    if not match:
        return ""
    number = f":job:{match.group(1)}"
    return next((str(row["job_id"]) for row in events if str(row.get("job_id", "")).endswith(number)), "")


def render_messages(messages: Any) -> list[str]:
    lines: list[str] = []
    if not isinstance(messages, list):
        return [fenced(messages)]
    for index, message in enumerate(messages, 1):
        if not isinstance(message, dict):
            lines.extend([f"##### 消息 {index}", "", fenced(message), ""])
            continue
        role = message.get("role", "unknown")
        lines.extend([f"##### 消息 {index} · `{role}`", ""])
        if message.get("name"):
            lines.append(f"- 名称：`{message['name']}`")
        if message.get("tool_call_id"):
            lines.append(f"- Tool Call ID：`{message['tool_call_id']}`")
        lines.extend([fenced(message.get("content", "")), ""])
        if message.get("tool_calls"):
            lines.extend(["消息中的工具请求：", "", fenced(message["tool_calls"], "json"), ""])
    return lines


def render_agent_trace(
    path: Path,
    events: list[dict[str, Any]],
    timings: dict[str, dict[str, Any]],
) -> list[str]:
    rows = load_jsonl(path)
    start = next((row for row in rows if row.get("t") == "start"), {})
    job_id = find_job_id(path, events)
    timing = timings.get(job_id, {})
    role = start.get("role_id", "unknown")
    variant = "主 Agent Loop"
    if "forced_final" in path.stem:
        variant = "强制最终回答"
    elif "output_repair" in path.stem:
        variant = "结构化输出修复"
    duration = timing.get("duration")
    lines = [
        f"### {job_id or path.stem} · `{role}` · {variant}",
        "",
        f"- Trace 文件：`{path.name}`",
        f"- 编排键：`{timing.get('key', '')}`",
        f"- 状态：{timing.get('status', '附加调用')}",
        f"- Job 总耗时：{duration:.3f} 秒" if duration is not None else "- Job 总耗时：计入所属 Job",
        f"- 最大 Turn：{start.get('max_turns', '未知')}",
        "",
        "<details>",
        "<summary>展开初始委派输入</summary>",
        "",
        fenced(start.get("user_message", "")),
        "",
        "</details>",
        "",
    ]
    tool_calls: dict[str, dict[str, Any]] = {}
    llm_index = 0
    tool_index = 0
    for row in rows:
        kind = row.get("t")
        if kind == "llm":
            llm_index += 1
            usage = row.get("usage") or row.get("Token用量") or {}
            elapsed = float(row.get("模型调用耗时毫秒", 0) or 0) / 1000
            lines.extend([
                f"#### 模型调用 {llm_index} · Turn {row.get('turn', '?')}",
                "",
                f"- 耗时：{elapsed:.3f} 秒",
                f"- 模型：`{usage.get('model', '未知')}`",
                f"- 输入 Token：{usage.get('prompt_tokens', 0):,}",
                f"- 输出 Token：{usage.get('completion_tokens', 0):,}",
                f"- 缓存 Token：{usage.get('cached_tokens', 0):,}",
                "",
                "<details>",
                "<summary>展开本次模型实际输入的全部消息</summary>",
                "",
            ])
            lines.extend(render_messages(row.get("模型输入消息", [])))
            lines.extend(["</details>", "", "模型输出文本：", "", fenced(
                row.get("模型输出文本", row.get("content", "")),
            ), ""])
            calls = row.get("模型请求的工具调用") or row.get("tool_calls") or []
            if calls:
                lines.extend(["本次模型请求的工具调用：", "", fenced(calls, "json"), ""])
        elif kind == "tool_call":
            tool_calls[str(row.get("tool_call_id", ""))] = row
        elif kind == "result":
            tool_index += 1
            call = tool_calls.get(str(row.get("tool_call_id", "")), {})
            name = row.get("工具名称") or row.get("name") or call.get("工具名称") or "unknown"
            args = call.get("工具输入参数") or row.get("工具输入参数") or {}
            elapsed = float(row.get("工具耗时毫秒", row.get("ms", 0)) or 0) / 1000
            result = row.get("工具返回信息", row.get("result", ""))
            lines.extend([
                f"#### 工具调用 {tool_index} · `{name}`",
                "",
                f"- Tool Call ID：`{row.get('tool_call_id', '')}`",
                f"- 耗时：{elapsed:.3f} 秒",
                f"- 是否错误：{'是' if row.get('是否错误', row.get('error', False)) else '否'}",
                "- 输入参数：",
                "",
                fenced(args, "json"),
                "",
                "<details>",
                "<summary>展开完整工具返回</summary>",
                "",
                fenced(result),
                "",
                "</details>",
                "",
            ])
    end = next((row for row in reversed(rows) if row.get("t") == "end"), None)
    if end:
        lines.extend([
            "#### Agent Loop 最终状态",
            "",
            f"- 停止原因：{end.get('停止原因', end.get('stopped_by', '未知'))}",
            f"- 累计 Turn：{end.get('累计轮次', end.get('turns', '未知'))}",
            f"- 累计工具调用：{end.get('累计工具调用次数', end.get('tool_calls', '未知'))}",
            "- 最终输出：",
            "",
            fenced(end.get("最终输出", "")),
            "",
        ])
    return lines


def render_planner(trial: Path, result: dict[str, Any]) -> list[str]:
    generated = trial / "generated"
    generation = result.get("generation", {})
    trace_path = trial / "trace.jsonl"
    trace_events = load_jsonl(trace_path) if trace_path.exists() else []
    orchestrator_starts = {
        int(event.get("payload", {}).get("attempt", 0)): event.get("payload", {})
        for event in trace_events
        if event.get("event_type") in {"orchestrator_generation_started", "code_repair_started"}
    }
    natural_plan_start = next((
        event.get("payload", {}) for event in trace_events
        if event.get("event_type") == "natural_language_plan_started"
    ), {})
    lines = [
        "## Planner 规划与 Orchestrator 代码生成验证",
        "",
        f"- 是否回退：{'是' if generation.get('fallback_reason') else '否'}",
        f"- 回退原因：{generation.get('fallback_reason') or '无'}",
        f"- Planner 与 Orchestrator 总用量：{json.dumps(generation.get('planner_usage', {}), ensure_ascii=False)}",
        "",
    ]
    request = generated / "generation_request.json"
    if request.exists():
        lines.extend([
            "<details>", "<summary>展开 Planner 完整生成请求</summary>", "",
            fenced(load_json(request), "json"), "", "</details>", "",
        ])
    plan_dir = generated / "plan"
    plan_usage = (
        load_json(plan_dir / "usage.json")
        if (plan_dir / "usage.json").exists() else {}
    )
    plan_input = natural_plan_start.get("model_input", {})
    if plan_dir.exists():
        lines.extend([
            "### 自然语言总计划", "",
            f"- 模型：`{natural_plan_start.get('model', '未知')}`",
            f"- 耗时：{plan_usage.get('duration_seconds', 0):.3f} 秒",
            f"- Token：{plan_usage.get('tokens', 0):,}",
            f"- LLM 调用：{plan_usage.get('llm_calls', 0)}", "",
        ])
        if plan_input.get("system_prompt"):
            lines.extend([
                "<details>", "<summary>展开自然语言 Planner System Prompt</summary>", "",
                fenced(plan_input["system_prompt"], "text"), "", "</details>", "",
            ])
        for filename, title in (
            ("planner_prompt.md", "自然语言 Planner 完整输入"),
            ("raw_response.md", "自然语言 Planner 原始输出"),
            ("plan.md", "最终自然语言总计划"),
        ):
            source = plan_dir / filename
            if source.exists():
                lines.extend([
                    "<details>", f"<summary>展开{title}</summary>", "",
                    fenced(source.read_text(encoding="utf-8"), "text"),
                    "", "</details>", "",
                ])
    attempts = generated / "attempts"
    for attempt in sorted(attempts.glob("attempt_*")):
        attempt_number = int(attempt.name.rsplit("_", 1)[-1])
        start_payload = orchestrator_starts.get(attempt_number, {})
        model_input = start_payload.get("model_input", {})
        usage = load_json(attempt / "usage.json") if (attempt / "usage.json").exists() else {}
        validation = load_json(attempt / "validation.json")
        lines.extend([
            f"### {attempt.name}", "",
            f"- 模型：`{start_payload.get('model', '未知')}`",
            f"- Orchestrator 耗时：{usage.get('duration_seconds', 0):.3f} 秒",
            f"- Token：{usage.get('tokens', 0):,}",
            f"- LLM 调用：{usage.get('llm_calls', 0)}",
            f"- 验证通过：{'是' if validation.get('accepted') else '否'}",
            f"- 验证问题：{json.dumps(validation.get('issues', []), ensure_ascii=False)}",
            f"- 非阻塞健壮性警告：{json.dumps(validation.get('warnings', []), ensure_ascii=False)}",
            "",
        ])
        if model_input.get("system_prompt"):
            lines.extend([
                "<details>", "<summary>展开 Orchestrator System Prompt</summary>", "",
                fenced(model_input["system_prompt"], "text"), "", "</details>", "",
            ])
        for filename, title, language in (
            ("orchestrator_prompt.md", "Orchestrator 完整输入", "text"),
            ("raw_response.md", "Orchestrator 原始输出", "python"),
        ):
            source = attempt / filename
            if source.exists():
                lines.extend([
                    "<details>", f"<summary>展开{title}</summary>", "",
                    fenced(source.read_text(encoding="utf-8"), language), "", "</details>", "",
                ])
    source = generated / "solve.py"
    if source.exists():
        lines.extend(["### 最终生成的 `solve(ctx)`", "", fenced(
            source.read_text(encoding="utf-8"), "python",
        ), ""])
    return lines


def build(run_dir: Path) -> str:
    trial_dirs = sorted((run_dir / "trials").iterdir(), key=trial_sort)
    records = [(trial, load_json(trial / "result.json")) for trial in trial_dirs if (trial / "result.json").exists()]
    totals = defaultdict(float)
    for _trial, result in records:
        usage = result.get("usage", {})
        for field in ("tokens", "llm_calls", "tool_calls", "jobs"):
            totals[field] += float(usage.get(field, 0) or 0)
        totals["job_duration_seconds"] += float(usage.get("duration_seconds", 0) or 0)
        totals["wall_duration_seconds"] += float(result.get("duration_seconds", 0) or 0)
    lines = [
        "# Generated Code Orchestration 10 题完整运行记录",
        "",
        f"> 来源目录：`{run_dir}`。本文从持久化 Trace 自动生成，保留 Planner 输入输出、代码验证、最终 solve 源码、每次模型实际输入输出、工具参数与完整返回、耗时、Token 和最终评分。原始 JSON/JSONL 文件未改动。",
        "",
        "## 总览",
        "",
        f"- 题数：{len(records)}",
        f"- 累计 Token：{int(totals['tokens']):,}",
        f"- 累计 LLM 调用：{int(totals['llm_calls'])}",
        f"- 累计工具调用：{int(totals['tool_calls'])}",
        f"- 累计 Job：{int(totals['jobs'])}",
        f"- 各题墙钟耗时之和：{totals['wall_duration_seconds']:.2f} 秒（{totals['wall_duration_seconds']/60:.2f} 分钟）",
        f"- Agent/Planner 工作时长累加：{totals['job_duration_seconds']:.2f} 秒（并行分支分别计时，不能视为墙钟时间）",
        "",
        "| ID | 模型答案 | 标准答案 | 严格匹配 | 回退 | 耗时（秒） | Token | LLM | 工具 | Job |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _trial, result in records:
        generation = result.get("generation", {})
        usage = result.get("usage", {})
        lines.append(
            f"| {result.get('question_id')} | {one_line(result.get('predicted_answer'))} | "
            f"{one_line(result.get('ground_truth'))} | "
            f"{'是' if result.get('strict_exact_match') else '否'} | "
            f"{'是' if generation.get('fallback_reason') else '否'} | "
            f"{result.get('duration_seconds', 0):.2f} | {usage.get('tokens', 0):,} | "
            f"{usage.get('llm_calls', 0)} | {usage.get('tool_calls', 0)} | {usage.get('jobs', 0)} |"
        )
    for trial, result in records:
        usage = result.get("usage", {})
        instruction_path = run_dir / "tasks" / str(result.get("question_id")) / "instruction.md"
        question = (
            instruction_path.read_text(encoding="utf-8")
            if instruction_path.exists() else result.get("question", "")
        )
        lines.extend([
            "", "---", "", f"# 题目 {result.get('question_id')}", "",
            "## 原题", "", fenced(question), "",
            "## 最终结果", "",
            f"- 模型答案：{result.get('predicted_answer')}",
            f"- 标准答案：{result.get('ground_truth')}",
            f"- 严格匹配：{'正确' if result.get('strict_exact_match') else '错误'}",
            f"- 无序球队匹配：{'正确' if result.get('unordered_team_match') else '错误'}",
            f"- 总耗时：{result.get('duration_seconds', 0):.3f} 秒",
            f"- Token：{usage.get('tokens', 0):,}",
            f"- LLM：{usage.get('llm_calls', 0)}；工具：{usage.get('tool_calls', 0)}；Job：{usage.get('jobs', 0)}",
            f"- 预算违规：{json.dumps(result.get('budget_violations', []), ensure_ascii=False)}",
            "",
        ])
        lines.extend(render_planner(trial, result))
        events_path = trial / "orchestration_live.jsonl"
        events = load_jsonl(events_path) if events_path.exists() else []
        timings = job_timings(events)
        lines.extend(["## Agent 与工具实际执行记录", ""])
        for trace in sorted((trial / "agent_traces").glob("*.jsonl"), key=trace_sort):
            lines.extend(render_agent_trace(trace, events, timings))
            lines.extend(["---", ""])
        lines.extend([
            "## 本题原始记录索引", "",
            f"- 结果：`{trial / 'result.json'}`",
            f"- 编排事件：`{events_path}`",
            f"- 标准 Trace：`{trial / 'trace.jsonl'}`",
            f"- Agent Trace 目录：`{trial / 'agent_traces'}`",
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build(args.run_dir.resolve()), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
