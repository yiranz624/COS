"""Render generated-run agent and reviewer records as readable Markdown."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in _read(path).splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _fence(text: str, language: str = "text") -> str:
    safe = (text or "（空）").replace("```", "''' ")
    return f"```{language}\n{safe}\n```"


def _verdict_summary(raw: str) -> tuple[str, str]:
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return "无法从输出中解析", raw.strip() or "未提供"
    if not isinstance(value, dict) or "accepted" not in value:
        return "未声明", "未提供"
    reasons: list[str] = []
    if value.get("critique"):
        reasons.append(str(value["critique"]))
    for key in ("issues", "failed_criteria"):
        items = value.get(key)
        if isinstance(items, list):
            reasons.extend(str(item) for item in items if item)
    return ("放行" if value.get("accepted") else "拒绝"), ("；".join(reasons) or "无")


def _review_sections(trial_dir: Path) -> list[str]:
    sections: list[str] = []
    groups = (
        ("Plan verifier", trial_dir / "generated" / "plan" / "attempts"),
        ("代码 verifier", trial_dir / "generated" / "attempts"),
    )
    for label, attempts in groups:
        if not attempts.exists():
            continue
        for attempt in sorted(attempts.glob("attempt_*")):
            review = attempt / "review"
            if not review.exists():
                continue
            raw = _read(review / "raw_response.md")
            status, reason = _verdict_summary(raw)
            sections.extend([
                f"## {label} · {attempt.name}", "",
                f"- 结论：{status}", f"- 打回理由：{reason}", "",
                "### System prompt", "", _fence(_read(review / "system_prompt.md")), "",
                "### 输入", "", _fence(_read(review / "review_prompt.md")), "",
                "### 输出", "", _fence(raw, "json"), "",
            ])
    return sections


def _runtime_sections(trial_dir: Path) -> list[str]:
    sections: list[str] = []
    trace_dir = trial_dir / "agent_traces"
    if not trace_dir.exists():
        return sections
    role_counts = {"researcher": 0, "verifier": 0}
    for path in sorted(trace_dir.glob("*.jsonl")):
        rows = _jsonl(path)
        start = next((row for row in rows if row.get("t") == "start"), {})
        role = str(start.get("role_id", ""))
        if role not in role_counts:
            continue
        role_counts[role] += 1
        end = next((row for row in reversed(rows) if row.get("t") == "end"), {})
        output = str(end.get("最终输出", end.get("final_content", "")))
        if role == "verifier":
            status, reason = _verdict_summary(output)
        else:
            status, reason = "研究完成", "不适用（researcher 不负责放行或打回）"
        sections.extend([
            f"## 运行时 {role} #{role_counts[role]}", "",
            f"- Trace：`agent_traces/{path.name}`",
            f"- 结论：{status}", f"- 打回理由：{reason}", "",
            "### System prompt", "", _fence(str(start.get("system_prompt", ""))), "",
            "### 输入", "", _fence(str(start.get("user_message", ""))), "",
            "### 输出", "", _fence(output, "json" if role == "verifier" else "text"), "",
        ])
    return sections


def write_trial_agent_report(trial_dir: str | Path) -> Path:
    """Write one human-readable report beside result.json and trace.jsonl."""
    trial = Path(trial_dir)
    lines = [
        "# Researcher 与 Verifier 输入输出记录", "",
        "> 本文件由运行器自动生成。它汇总自然语言计划 verifier、代码 verifier，"
        "以及运行时 researcher/verifier 的输入、输出和打回理由。原始 JSONL trace 仍保留用于机器分析。",
        "",
    ]
    sections = _review_sections(trial) + _runtime_sections(trial)
    lines.extend(sections or ["本次运行没有找到 researcher 或 verifier 记录。", ""])
    output = trial / "researcher_verifier_report.md"
    output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return output
