"""Human-reviewed three-phase Decomposer–Solver/Verifier–Synthesizer flow."""

from __future__ import annotations

import json
from typing import Any

from agent_harness.orchestration.context import OrchestrationContext
from orchestrations.contracts import (
    SubproblemStateV1,
    SubtaskPlanV1,
    SubtaskSpec,
    fallback_plan,
    validate_plan,
)

MANIFEST = {
    "script_id": "reference_v1",
    "version": 1,
    "entrypoint": "solve",
    "input_schema": "string",
    "output_schema": "string",
    "compatible_runtime": ">=1.0",
    "roles": ("decomposer", "solver", "verifier", "synthesizer"),
}

_ROUND_LIMIT = {"easy": 1, "medium": 2, "hard": 3}
_ESTIMATED_ROUND_TOKENS = 12_000


async def solve(ctx: OrchestrationContext) -> str:
    task = str(ctx.task)
    identity = _identity(ctx)
    ctx.emit("script_started", **identity)

    restored = ctx.load_checkpoint("reference_v1_state") or {}
    if restored.get("final_answer"):
        ctx.emit("script_completed", **identity, restored=True)
        return str(restored["final_answer"])

    plan = await _load_or_create_plan(ctx, task, restored)
    results = {
        key: SubproblemStateV1.from_dict(value)
        for key, value in restored.get("results", {}).items()
    }
    partial: dict[str, SubproblemStateV1] = {}
    for subtask in plan.subtasks:
        if subtask.id in results:
            continue
        saved = ctx.load_checkpoint(f"subtask:{subtask.id}")
        if saved:
            recovered = SubproblemStateV1.from_dict(saved)
            if recovered.status == "accepted" or (
                len(recovered.rounds) >= _ROUND_LIMIT[subtask.difficulty]
            ):
                results[subtask.id] = recovered
            else:
                partial[subtask.id] = recovered

    while len(results) < len(plan.subtasks):
        frontier = [
            item for item in plan.subtasks
            if item.id not in results
            and all(dep in results for dep in item.dependencies)
        ]
        if not frontier:
            raise RuntimeError("validated DAG has no executable frontier")
        ctx.emit(
            "frontier_started", **identity,
            subtask_ids=[item.id for item in frontier],
        )
        states = await ctx.map(
            lambda item: _solve_one(
                ctx, task, item, results, partial.get(item.id)
            ), frontier
        )
        results.update({state.subtask_id: state for state in states})
        await _save_state(ctx, plan, results)
        ctx.emit(
            "frontier_completed", **identity,
            subtask_ids=[item.id for item in frontier],
        )

    ctx.emit("synthesis_started", **identity)
    synthesis_input = _render_synthesis(task, plan, results)
    try:
        answer = str(await ctx.call(
            "synthesizer", synthesis_input,
            key="reference_v1:synthesis", max_turns=1,
        )).strip()
    except Exception:
        answer = "信息未查全"
    answer = answer or "信息未查全"
    await _save_state(ctx, plan, results, final_answer=answer)
    ctx.emit("synthesis_completed", **identity)
    ctx.emit("script_completed", **identity, restored=False)
    return answer


async def _load_or_create_plan(
    ctx: OrchestrationContext, task: str, restored: dict[str, Any]
) -> SubtaskPlanV1:
    if restored.get("plan"):
        plan = SubtaskPlanV1.from_dict(restored["plan"])
        validate_plan(plan)
        return plan
    ctx.emit("plan_created", **_identity(ctx), status="requested")
    try:
        raw = await ctx.call(
            "decomposer", _render_decomposer(task),
            key="reference_v1:decomposition", max_turns=1,
        )
        plan = SubtaskPlanV1.from_dict(raw)
        validate_plan(plan)
        fallback = False
    except Exception as exc:
        plan = fallback_plan(task)
        fallback = True
        fallback_reason = f"{type(exc).__name__}: {exc}"
    ctx.emit(
        "plan_validated", **_identity(ctx), fallback=fallback,
        fallback_reason=fallback_reason if fallback else None,
        subtask_count=len(plan.subtasks),
    )
    await ctx.checkpoint("reference_v1_state", {
        "plan": plan.to_dict(), "results": {}, "final_answer": None,
    })
    return plan


async def _solve_one(
    ctx: OrchestrationContext,
    original_task: str,
    subtask: SubtaskSpec,
    completed: dict[str, SubproblemStateV1],
    initial: SubproblemStateV1 | None = None,
) -> SubproblemStateV1:
    state = initial or SubproblemStateV1(subtask_id=subtask.id)
    dependencies = {
        dep: completed[dep].to_dict() for dep in subtask.dependencies
    }
    for round_index in range(
        len(state.rounds), _ROUND_LIMIT[subtask.difficulty]
    ):
        if not _can_start_round(ctx):
            state.status = "incomplete"
            state.error = "预算不足，保留当前 best effort 并停止新一轮"
            break
        ctx.emit(
            "solver_round_started", **_identity(ctx),
            subtask_id=subtask.id, round_index=round_index,
        )
        try:
            solution = str(await ctx.call(
                "solver",
                _render_solver(
                    original_task, subtask, dependencies, state, round_index
                ),
                key=f"reference_v1:{subtask.id}:solver:{round_index}",
                max_turns=4,
                metadata={"allowed_tools": _allowed_tools(subtask)},
            )).strip()
        except Exception as exc:  # worker failure is data, not run failure
            state.status = "incomplete"
            state.error = f"Solver失败: {type(exc).__name__}: {exc}"
            state.critique = state.error
            state.failed_criteria = list(subtask.success_criteria)
            state.rounds.append({
                "round_index": round_index,
                "solution": state.solution,
                "verdict": {
                    "accepted": False,
                    "critique": state.error,
                    "satisfied_criteria": list(state.satisfied_criteria),
                    "failed_criteria": list(subtask.success_criteria),
                    "verified_evidence": list(state.verified_evidence),
                    "confidence": state.confidence,
                },
            })
            await ctx.checkpoint(f"subtask:{subtask.id}", state.to_dict())
            continue
        ctx.emit(
            "solver_round_completed", **_identity(ctx),
            subtask_id=subtask.id, round_index=round_index,
        )
        try:
            raw_verdict = await ctx.call(
                "verifier",
                _render_verifier(original_task, subtask, solution),
                key=f"reference_v1:{subtask.id}:verifier:{round_index}",
                max_turns=2,
                metadata={"allowed_tools": _allowed_tools(subtask)},
            )
            verdict = _normalize_verdict(raw_verdict)
        except Exception as exc:
            verdict = {
                "accepted": False,
                "critique": f"Verifier失败: {type(exc).__name__}: {exc}",
                "satisfied_criteria": [],
                "failed_criteria": list(subtask.success_criteria),
                "verified_evidence": [],
                "confidence": 0.0,
            }
        state.solution = solution
        state.critique = verdict["critique"]
        state.satisfied_criteria = verdict["satisfied_criteria"]
        state.failed_criteria = verdict["failed_criteria"]
        state.verified_evidence = verdict["verified_evidence"]
        state.confidence = verdict["confidence"]
        state.rounds.append({
            "round_index": round_index,
            "solution": solution,
            "verdict": verdict,
        })
        state.status = "accepted" if verdict["accepted"] else "incomplete"
        ctx.emit(
            "verifier_round_completed", **_identity(ctx),
            subtask_id=subtask.id, round_index=round_index,
            accepted=verdict["accepted"],
        )
        await ctx.checkpoint(
            f"subtask:{subtask.id}", state.to_dict()
        )
        if verdict["accepted"]:
            ctx.emit(
                "subtask_accepted", **_identity(ctx), subtask_id=subtask.id
            )
            return state
    ctx.emit(
        "subtask_best_effort", **_identity(ctx), subtask_id=subtask.id,
        status=state.status,
    )
    return state


def _can_start_round(ctx: OrchestrationContext) -> bool:
    budget = ctx.metadata.get("budget", {})
    maximum = budget.get("max_tokens")
    if maximum is None:
        return True
    used = ctx.remaining_budget().usage["tokens"]
    phase_three_reserve = int(maximum * 0.15)
    phase_two_cap = int(maximum * 0.75)
    return (
        used + _ESTIMATED_ROUND_TOKENS <= maximum - phase_three_reserve
        and used + _ESTIMATED_ROUND_TOKENS <= int(maximum * 0.10) + phase_two_cap
    )


async def _save_state(
    ctx: OrchestrationContext,
    plan: SubtaskPlanV1,
    results: dict[str, SubproblemStateV1],
    *,
    final_answer: str | None = None,
) -> None:
    await ctx.checkpoint("reference_v1_state", {
        "plan": plan.to_dict(),
        "results": {key: value.to_dict() for key, value in results.items()},
        "final_answer": final_answer,
    })


def _identity(ctx: OrchestrationContext) -> dict[str, Any]:
    return {
        "script_id": ctx.metadata.get("script_id", "reference_v1"),
        "version": ctx.metadata.get("script_version", 1),
        "source_hash": ctx.metadata.get("script_source_hash", ""),
    }


def _render_decomposer(task: str) -> str:
    return (
        "请把原始任务拆成最多3个互补子问题DAG。连续过滤步骤必须用dependencies表示，"
        "不可把每个子问题写成重复回答完整原题的路线。tools只能从web_search、"
        "web_fetch、run_python_code中选择。不要解题。输出必须严格匹配角色JSON schema。\n\n"
        f"原始任务：\n{task}"
    )


def _render_solver(
    task: str, subtask: SubtaskSpec, dependencies: dict[str, Any],
    state: SubproblemStateV1, round_index: int,
) -> str:
    return (
        f"原始任务：\n{task}\n\n当前子问题：\n"
        f"{json.dumps(subtask.__dict__, ensure_ascii=False)}\n\n"
        f"依赖子问题结果（含不确定性）：\n{json.dumps(dependencies, ensure_ascii=False)}\n\n"
        f"轮次：{round_index + 1}\n上一轮完整solution：\n{state.solution}\n\n"
        f"Verifier critique：\n{state.critique}\n\n"
        f"已确认事实与URL：\n{json.dumps(state.verified_evidence, ensure_ascii=False)}\n"
        f"尚未满足标准：\n{json.dumps(state.failed_criteria, ensure_ascii=False)}\n\n"
        "使用允许的联网或计算工具补足证据；若仍不足，明确说明信息未查全。"
    )


def _render_verifier(task: str, subtask: SubtaskSpec, solution: str) -> str:
    return (
        f"原始任务：\n{task}\n\n只验证当前子问题：\n{subtask.question}\n\n"
        f"成功标准：\n{json.dumps(subtask.success_criteria, ensure_ascii=False)}\n"
        f"所需证据：\n{json.dumps(subtask.required_evidence, ensure_ascii=False)}\n\n"
        f"待验证solution：\n{solution}\n\n"
        "逐项判断成功标准，不要以是否已回答整道原题作为唯一标准。"
    )


def _render_synthesis(
    task: str, plan: SubtaskPlanV1,
    results: dict[str, SubproblemStateV1],
) -> str:
    return (
        f"原始问题：\n{task}\n\n完整DAG：\n"
        f"{json.dumps(plan.to_dict(), ensure_ascii=False)}\n\n"
        f"各子问题最终状态和全部轮次：\n{json.dumps({k: v.to_dict() for k, v in results.items()}, ensure_ascii=False)}\n\n"
        "综合 accepted 事实和 incomplete 中已验证的局部事实；不要采用被明确证伪的事实。"
        "最终只输出 benchmark 要求的简洁答案；依据不足时只输出：信息未查全。"
    )


def _normalize_verdict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("verifier output must be an object")
    failed = [str(x) for x in value.get("failed_criteria", [])]
    accepted = bool(value.get("accepted", False)) and not failed
    return {
        "accepted": accepted,
        "critique": str(value.get("critique", "")),
        "satisfied_criteria": [
            str(x) for x in value.get("satisfied_criteria", [])
        ],
        "failed_criteria": failed,
        "verified_evidence": list(value.get("verified_evidence", [])),
        "confidence": max(0.0, min(1.0, float(value.get("confidence", 0.0)))),
    }


def _allowed_tools(subtask: SubtaskSpec) -> list[str]:
    """Apply the AI-authored per-subtask capability set to each Job."""
    allowed = {"web_search", "web_fetch", "run_python_code"}
    return [tool for tool in subtask.tools if tool in allowed]
