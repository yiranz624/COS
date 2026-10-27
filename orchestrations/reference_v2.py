"""Candidate-funnel workflow with coverage-first budget scheduling."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from typing import Any

from agent_harness.orchestration.context import OrchestrationContext
from orchestrations.contracts import (
    SolverResultV2,
    SubtaskPlanV1,
    SubtaskSpec,
    validate_plan,
)

MANIFEST = {
    "script_id": "reference_v2",
    "version": 2,
    "entrypoint": "solve",
    "input_schema": "string",
    "output_schema": "string",
    "compatible_runtime": ">=1.0",
    "roles": ("decomposer", "solver", "verifier", "synthesizer"),
}

_ALLOWED_TOOLS = {"web_search", "web_fetch", "run_python_code"}
_ROUND_LIMIT = {"easy": 1, "medium": 2, "hard": 3}
_FIRST_PASS_ESTIMATE = 8_000
_REVISION_ESTIMATE = 10_000
_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S | re.I)


async def solve(ctx: OrchestrationContext) -> str:
    task = str(ctx.task)
    identity = _identity(ctx)
    ctx.emit("script_started", **identity)
    restored = ctx.load_checkpoint("reference_v2_state") or {}
    if restored.get("final_answer"):
        ctx.emit("script_completed", **identity, restored=True)
        return str(restored["final_answer"])

    plan = await _plan(ctx, task, restored)
    states: dict[str, dict[str, Any]] = dict(restored.get("states", {}))

    # Coverage pass: every reachable DAG node gets one bounded attempt before
    # any upstream revision is allowed to spend the shared retry pool.
    while any(item.id not in states for item in plan.subtasks):
        frontier = [
            item for item in plan.subtasks
            if item.id not in states
            and all(dependency in states for dependency in item.dependencies)
        ]
        if not frontier:
            raise RuntimeError("validated DAG has no executable frontier")
        ctx.emit("frontier_started", **identity,
                 subtask_ids=[item.id for item in frontier], pass_type="coverage")
        completed = await ctx.map(
            lambda item: _run_round(
                ctx, task, item, states, previous=None, round_index=0,
                verifier_turns=1,
            ),
            frontier,
        )
        states.update({item["subtask_id"]: item for item in completed})
        await _save(ctx, plan, states)
        ctx.emit("frontier_completed", **identity,
                 subtask_ids=[item.id for item in frontier], pass_type="coverage")

    # Revision pass: only rejected/incomplete nodes, while preserving synthesis.
    for subtask in plan.subtasks:
        state = states[subtask.id]
        for round_index in range(1, _ROUND_LIMIT[subtask.difficulty]):
            if state.get("accepted") or not _can_spend(ctx, _REVISION_ESTIMATE):
                break
            state = await _run_round(
                ctx, task, subtask, states, previous=state,
                round_index=round_index, verifier_turns=2,
            )
            states[subtask.id] = state
            await _save(ctx, plan, states)

    for subtask in plan.subtasks:
        state = states[subtask.id]
        ctx.emit(
            "subtask_accepted" if state.get("accepted") else "subtask_best_effort",
            **identity, subtask_id=subtask.id,
            status="accepted" if state.get("accepted") else "incomplete",
        )

    ctx.emit("synthesis_started", **identity)
    try:
        answer = str(await ctx.call(
            "synthesizer", _synthesis_instruction(task, plan, states),
            key="reference_v2:synthesis", max_turns=1,
        )).strip()
    except Exception:
        answer = "信息未查全"
    answer = answer or "信息未查全"
    await _save(ctx, plan, states, final_answer=answer)
    ctx.emit("synthesis_completed", **identity)
    ctx.emit("script_completed", **identity, restored=False)
    return answer


async def _plan(
    ctx: OrchestrationContext, task: str, restored: dict[str, Any]
) -> SubtaskPlanV1:
    if restored.get("plan"):
        result = SubtaskPlanV1.from_dict(restored["plan"])
        validate_plan(result)
        return result
    ctx.emit("plan_created", **_identity(ctx), status="requested")
    reason = ""
    try:
        raw = await ctx.call(
            "decomposer", _decomposer_instruction(task),
            key="reference_v2:decomposition", max_turns=1,
        )
        result = SubtaskPlanV1.from_dict(raw)
        validate_plan(result)
        _validate_plan_semantics(result)
        fallback = False
    except (KeyError, TypeError, ValueError) as exc:
        reason = str(exc)
        result = _candidate_funnel(task)
        fallback = True
    ctx.emit("plan_validated", **_identity(ctx), fallback=fallback,
             fallback_reason=reason, subtask_count=len(result.subtasks))
    await _save(ctx, result, {})
    return result


def _validate_plan_semantics(plan: SubtaskPlanV1) -> None:
    unknown = {
        tool for item in plan.subtasks for tool in item.tools
        if tool not in _ALLOWED_TOOLS
    }
    if unknown:
        raise ValueError(f"plan references unavailable tools: {sorted(unknown)}")
    broad_markers = (
        "所有足球比赛", "全部足球比赛", "all football matches",
        "all soccer matches", "complete database", "完整数据库",
    )
    for item in plan.subtasks:
        lowered = item.question.casefold()
        if any(marker in lowered for marker in broad_markers):
            raise ValueError("plan contains an unbounded exhaustive-search subtask")
        if "web_search" in item.tools and len(item.success_criteria) == 0:
            raise ValueError("search subtask lacks bounded success criteria")


def _candidate_funnel(task: str) -> SubtaskPlanV1:
    return SubtaskPlanV1(
        original_task=task,
        subtasks=[
            SubtaskSpec(
                id="candidate_discovery",
                question=(
                    "围绕完整原题中至少三个最稀有且可检索的联合约束，发现至多5个"
                    "具体候选答案；不要枚举整个时间段、领域或数据库。"
                ),
                dependencies=[], difficulty="medium",
                success_criteria=["产生不超过5个有来源URL的具体候选"],
                required_evidence=["候选名称", "联合约束命中说明", "来源URL"],
                tools=["web_search", "web_fetch"],
            ),
            SubtaskSpec(
                id="constraint_verification",
                question=(
                    "只针对上游候选逐一验证原题的全部时间、数量、身份和事件约束，"
                    "淘汰明确不满足者并保留未确认项。"
                ),
                dependencies=["candidate_discovery"], difficulty="medium",
                success_criteria=["每个候选的每项原题约束均有状态和URL证据"],
                required_evidence=["逐约束状态", "支持或反驳URL"],
                tools=["web_search", "web_fetch"],
            ),
            SubtaskSpec(
                id="cross_check",
                question=(
                    "对剩余候选使用不同来源交叉复核决定性约束，并确定 benchmark 所需"
                    "的精确简短答案；不要重新进行无边界候选发现。"
                ),
                dependencies=["constraint_verification"], difficulty="easy",
                success_criteria=["得到唯一答案或明确列出仍缺失的决定性证据"],
                required_evidence=["独立来源URL", "最终答案格式"],
                tools=["web_search", "web_fetch"],
            ),
        ],
        synthesis_requirements=[
            "仅输出有完整证据支持的简短答案；否则输出信息未查全"
        ],
    )


async def _run_round(
    ctx: OrchestrationContext, task: str, subtask: SubtaskSpec,
    states: dict[str, dict[str, Any]], *, previous: dict[str, Any] | None,
    round_index: int, verifier_turns: int,
) -> dict[str, Any]:
    if not _can_spend(ctx, _FIRST_PASS_ESTIMATE if round_index == 0 else _REVISION_ESTIMATE):
        return _empty_state(subtask.id, round_index, "预算不足，未启动本轮")
    dependencies = {key: states[key] for key in subtask.dependencies}
    ctx.emit("solver_round_started", **_identity(ctx),
             subtask_id=subtask.id, round_index=round_index)
    try:
        raw = await ctx.call(
            "solver", _solver_instruction(task, subtask, dependencies, previous),
            key=f"reference_v2:{subtask.id}:solver:{round_index}", max_turns=2,
        )
        solution = _parse_solver_result(raw)
        solver_error = None
    except Exception as exc:
        solution = SolverResultV2(
            status="incomplete", candidates=[], verified_facts=[],
            missing_constraints=["Solver执行失败"], search_summary="",
        )
        solver_error = f"{type(exc).__name__}: {exc}"
    ctx.emit("solver_round_completed", **_identity(ctx),
             subtask_id=subtask.id, round_index=round_index,
             candidate_count=len(solution.candidates))
    try:
        verdict = await ctx.call(
            "verifier", _verifier_instruction(task, subtask, solution),
            key=f"reference_v2:{subtask.id}:verifier:{round_index}",
            max_turns=verifier_turns,
        )
        verdict = _normalize_verdict(verdict)
    except Exception as exc:
        verdict = {"accepted": False, "critique": f"Verifier失败: {exc}",
                   "satisfied_criteria": [],
                   "failed_criteria": list(subtask.success_criteria),
                   "verified_evidence": [], "confidence": 0.0}
    state = {
        "subtask_id": subtask.id,
        "accepted": verdict["accepted"],
        "solution": solution.to_dict(),
        "verdict": verdict,
        "round_index": round_index,
        "error": solver_error,
    }
    ctx.emit("verifier_round_completed", **_identity(ctx),
             subtask_id=subtask.id, round_index=round_index,
             accepted=verdict["accepted"])
    await ctx.checkpoint(f"subtask:{subtask.id}", state)
    return state


def _parse_solver_result(raw: Any) -> SolverResultV2:
    if isinstance(raw, dict):
        value = raw
    else:
        text = str(raw).strip()
        match = _JSON_FENCE.match(text)
        if match:
            text = match.group(1)
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return SolverResultV2(
                status="incomplete", candidates=[], verified_facts=[],
                missing_constraints=["Solver未返回结构化候选"],
                search_summary=text[:2000],
            )
    if not isinstance(value, dict):
        raise TypeError("Solver result must be a JSON object")
    return SolverResultV2.from_dict(value)


def _normalize_verdict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("Verifier result must be an object")
    evidence = []
    for item in value.get("verified_evidence", []):
        if isinstance(item, dict):
            evidence.append({
                "claim": str(item.get("claim", "")),
                "url": str(item.get("url", "")),
                "source": str(item.get("source", "")),
                "status": str(item.get("status", "unverified")),
            })
    return {
        "accepted": bool(value.get("accepted")),
        "critique": str(value.get("critique", "")),
        "satisfied_criteria": [str(x) for x in value.get("satisfied_criteria", [])],
        "failed_criteria": [str(x) for x in value.get("failed_criteria", [])],
        "verified_evidence": evidence,
        "confidence": max(0.0, min(1.0, float(value.get("confidence", 0)))),
    }


def _can_spend(ctx: OrchestrationContext, estimate: int) -> bool:
    maximum = ctx.metadata.get("budget", {}).get("max_tokens")
    if maximum is None:
        return True
    used = ctx.remaining_budget().usage["tokens"]
    return used + estimate <= int(maximum * 0.85)


async def _save(
    ctx: OrchestrationContext, plan: SubtaskPlanV1,
    states: dict[str, dict[str, Any]], *, final_answer: str | None = None,
) -> None:
    await ctx.checkpoint("reference_v2_state", {
        "plan": plan.to_dict(), "states": states, "final_answer": final_answer,
    })


def _empty_state(subtask_id: str, round_index: int, error: str) -> dict[str, Any]:
    return {"subtask_id": subtask_id, "accepted": False,
            "solution": SolverResultV2(
                status="incomplete", candidates=[], verified_facts=[],
                missing_constraints=[error], search_summary="",
            ).to_dict(), "verdict": {"accepted": False, "critique": error,
                "satisfied_criteria": [], "failed_criteria": [],
                "verified_evidence": [], "confidence": 0.0},
            "round_index": round_index, "error": error}


def _identity(ctx: OrchestrationContext) -> dict[str, Any]:
    return {"script_id": ctx.metadata.get("script_id", "reference_v2"),
            "version": ctx.metadata.get("script_version", 2),
            "source_hash": ctx.metadata.get("script_source_hash", "")}


def _decomposer_instruction(task: str) -> str:
    return (
        "生成最多3个可执行的候选漏斗DAG。禁止要求枚举某时期或领域的所有记录。"
        "首节点必须使用原题至少三个稀有约束联合发现不超过5个候选；后续节点只验证"
        "候选。只能声明web_search、web_fetch、run_python_code。输出角色schema JSON。\n\n"
        f"原始任务：\n{task}"
    )


def _solver_instruction(
    task: str, subtask: SubtaskSpec, dependencies: dict[str, Any],
    previous: dict[str, Any] | None,
) -> str:
    schema = {"status": "candidate|incomplete", "candidates": [{
        "answer": "具体候选", "constraint_status": {"原题约束": "verified|refuted|unknown"},
        "evidence": [{"claim": "事实", "url": "https://...", "source": "来源名",
                      "status": "verified|refuted|unverified"}]}],
        "verified_facts": [{"claim": "事实", "url": "https://...",
                            "source": "来源名", "status": "verified"}],
        "missing_constraints": ["仍缺约束"], "search_summary": "简短检索摘要"}
    return (
        f"原始任务：\n{task}\n\n当前子任务：\n{json.dumps(asdict(subtask), ensure_ascii=False)}"
        f"\n\n依赖结果：\n{json.dumps(dependencies, ensure_ascii=False)}"
        f"\n\n上一轮状态：\n{json.dumps(previous, ensure_ascii=False)}\n\n"
        "优先用多个稀有约束组成联合查询；最多保留5个具体候选。不要枚举完整数据库，"
        "不要重复无新增证据的搜索。每条证据必须保留URL。最终只输出以下结构的JSON：\n"
        f"{json.dumps(schema, ensure_ascii=False)}"
    )


def _verifier_instruction(
    task: str, subtask: SubtaskSpec, solution: SolverResultV2,
) -> str:
    return (
        f"原始任务：\n{task}\n\n子任务及成功标准：\n"
        f"{json.dumps(asdict(subtask), ensure_ascii=False)}\n\n"
        f"结构化候选：\n{json.dumps(solution.to_dict(), ensure_ascii=False)}\n\n"
        "逐项验证当前子任务标准。没有URL的事实不得列入verified_evidence；"
        "verified_evidence必须使用claim/url/source/status对象。"
    )


def _synthesis_instruction(
    task: str, plan: SubtaskPlanV1, states: dict[str, dict[str, Any]],
) -> str:
    return (
        f"原始任务：\n{task}\n\nDAG：\n{json.dumps(plan.to_dict(), ensure_ascii=False)}"
        f"\n\n所有结构化结果：\n{json.dumps(states, ensure_ascii=False)}\n\n"
        "只能采用有URL支持且未被refuted的事实。输出benchmark要求的最短精确答案；"
        "决定性约束仍未验证时只输出：信息未查全。"
    )
