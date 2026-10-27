"""Accuracy-first stateful Decomposer–Solver/Verifier–Synthesizer workflow."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from agent_harness.orchestration.context import OrchestrationContext
from orchestrations.contracts import SubtaskPlanV1, SubtaskSpec, fallback_plan, validate_plan

MANIFEST = {
    "script_id": "reference_v6",
    "version": 6,
    "entrypoint": "solve",
    "input_schema": "string",
    "output_schema": "string",
    "compatible_runtime": ">=1.1",
    "roles": ("decomposer", "plan_reviewer", "stateful_solver", "verifier",
              "synthesizer"),
}

_ROUND_CAP = {"easy": 2, "medium": 4, "hard": 6}
_KNOWN_TOOLS = {"web_search", "web_fetch", "run_python_code"}


async def solve(ctx: OrchestrationContext) -> str:
    task = str(ctx.task)
    restored = ctx.load_checkpoint("reference_v6_state") or {}
    ctx.emit("script_started", **_identity(ctx))
    if restored.get("final_answer"):
        ctx.emit("script_completed", **_identity(ctx), restored=True)
        return str(restored["final_answer"])

    plan = await _plan(ctx, task, restored)
    states = {key: _state(value) for key, value in restored.get("states", {}).items()}
    ordered = _topological(plan)

    # Fairness pass: every DAG node receives one Solver–Verifier opportunity
    # before any node receives a revision.
    for subtask in ordered:
        if states.get(subtask.id, {}).get("rounds"):
            continue
        dependencies = {dep: states[dep] for dep in subtask.dependencies}
        states[subtask.id] = await _run_round(
            ctx, task, subtask, dependencies, states.get(subtask.id),
        )
        await _save(ctx, plan, states)

    # Accuracy-first revisions. A node stops on acceptance, its difficulty cap,
    # or two consecutive rounds without material state change.
    changed = True
    while changed:
        changed = False
        for subtask in ordered:
            current = states[subtask.id]
            dependencies = {dep: states[dep] for dep in subtask.dependencies}
            dep_signature = _digest({key: _summary(value)
                                     for key, value in dependencies.items()})
            stale = bool(current.get("dependency_signature") != dep_signature
                         and subtask.dependencies)
            cap = _round_cap(ctx, subtask)
            if len(current["rounds"]) >= cap:
                continue
            if current["status"] == "accepted" and not stale:
                continue
            if current.get("stagnant_rounds", 0) >= 2 and not stale:
                continue
            states[subtask.id] = await _run_round(
                ctx, task, subtask, dependencies, current,
            )
            await _save(ctx, plan, states)
            changed = True

    ctx.emit("synthesis_started", **_identity(ctx))
    try:
        answer = str(await ctx.call(
            "synthesizer", _synthesis_prompt(task, plan, states),
            key="reference_v6:synthesis", max_turns=2,
            metadata={"allowed_tools": []},
        )).strip()
    except Exception:
        answer = "信息未查全"
    answer = answer or "信息未查全"
    await _save(ctx, plan, states, final_answer=answer)
    ctx.emit("synthesis_completed", **_identity(ctx))
    ctx.emit("script_completed", **_identity(ctx), restored=False)
    return answer


async def _plan(
    ctx: OrchestrationContext, task: str, restored: dict[str, Any],
) -> SubtaskPlanV1:
    if restored.get("plan"):
        value = SubtaskPlanV1.from_dict(restored["plan"])
        validate_plan(value)
        return value
    critique = ""
    failure = ""
    for attempt in range(2):
        try:
            raw = await ctx.call(
                "decomposer", _decomposer_prompt(task, critique),
                key=f"reference_v6:decomposition:{attempt}:plan", max_turns=1,
                metadata={"allowed_tools": []},
            )
            candidate = SubtaskPlanV1.from_dict(raw)
            validate_plan(candidate)
            review = await ctx.call(
                "plan_reviewer", _review_prompt(task, candidate),
                key=f"reference_v6:plan_review:{attempt}:plan", max_turns=1,
                metadata={"allowed_tools": []},
            )
            missing = list(review.get("missing_requirements", []))
            invented = list(review.get("invented_requirements", []))
            accepted = bool(review.get("accepted", False)) and not missing and not invented
            critique = str(review.get("critique", ""))
            ctx.emit("plan_reviewed", **_identity(ctx), attempt=attempt,
                     accepted=accepted, critique=critique,
                     missing_requirements=missing,
                     invented_requirements=invented)
            if accepted:
                await _save(ctx, candidate, {})
                return candidate
            failure = critique or "plan reviewer rejected the plan"
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            critique = failure
            ctx.emit("plan_attempt_failed", **_identity(ctx), attempt=attempt,
                     reason=failure)
    value = fallback_plan(task)
    ctx.emit("plan_fallback", **_identity(ctx), reason=failure)
    await _save(ctx, value, {})
    return value


async def _run_round(
    ctx: OrchestrationContext, task: str, subtask: SubtaskSpec,
    dependencies: dict[str, dict[str, Any]], current: dict[str, Any] | None,
) -> dict[str, Any]:
    state = _state(current or {"subtask_id": subtask.id})
    round_index = len(state["rounds"])
    dep_signature = _digest({key: _summary(value)
                             for key, value in dependencies.items()})
    ctx.emit("solver_round_started", **_identity(ctx), subtask_id=subtask.id,
             round_index=round_index)
    before = _research_signature(state["research"])
    try:
        update = await ctx.call(
            "stateful_solver",
            _solver_prompt(task, subtask, dependencies, state, round_index),
            key=f"reference_v6:{subtask.id}:solver:{round_index}", max_turns=10,
            metadata={"allowed_tools": _tools(subtask)},
        )
        if not isinstance(update, dict):
            raise TypeError("stateful_solver output must be an object")
        research = _merge_research(state["research"], update)
        solution = str(update.get("solution", "")).strip()
        solver_error = None
    except Exception as exc:
        research = state["research"]
        solution = state.get("solution", "")
        solver_error = f"Solver失败: {type(exc).__name__}: {exc}"
    if solver_error:
        verdict = _failed_verdict(subtask, solver_error)
    else:
        try:
            raw = await ctx.call(
                "verifier", _verifier_prompt(task, subtask, solution, research),
                key=f"reference_v6:{subtask.id}:verifier:{round_index}",
                max_turns=3, metadata={"allowed_tools": _tools(subtask)},
            )
            verdict = _verdict(raw, subtask)
        except Exception as exc:
            verdict = _failed_verdict(
                subtask, f"Verifier失败: {type(exc).__name__}: {exc}",
            )
    research = _merge_research(research, {
        "verified_evidence": verdict.get("verified_evidence", []),
    })
    gained = before != _research_signature(research)
    ctx.emit("solver_round_completed", **_identity(ctx), subtask_id=subtask.id,
             round_index=round_index, information_gain=gained,
             query_count=len(research["queries_attempted"]),
             candidate_count=len(research["candidates"]))
    accepted = bool(verdict["accepted"])
    state.update({
        "status": "accepted" if accepted else "incomplete",
        "solution": solution,
        "critique": verdict["critique"],
        "research": research,
        "verdict": verdict,
        "dependency_signature": dep_signature,
        "stagnant_rounds": 0 if gained else state.get("stagnant_rounds", 0) + 1,
        "error": solver_error,
    })
    state["rounds"].append({
        "round_index": round_index, "solution": solution,
        "research_update": update if not solver_error else {},
        "information_gain": gained, "verdict": verdict,
    })
    ctx.emit("verifier_round_completed", **_identity(ctx),
             subtask_id=subtask.id, round_index=round_index,
             accepted=accepted, information_gain=gained)
    await ctx.checkpoint(f"reference_v6:subtask:{subtask.id}", state)
    return state


def _state(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "subtask_id": str(value.get("subtask_id", "")),
        "status": str(value.get("status", "pending")),
        "solution": str(value.get("solution", "")),
        "critique": str(value.get("critique", "")),
        "research": _merge_research({}, value.get("research", {})),
        "verdict": dict(value.get("verdict", {})),
        "rounds": list(value.get("rounds", [])),
        "stagnant_rounds": int(value.get("stagnant_rounds", 0)),
        "dependency_signature": str(value.get("dependency_signature", "")),
        "error": value.get("error"),
    }


def _empty_research() -> dict[str, list[Any]]:
    return {"queries_attempted": [], "observations": [], "candidates": [],
            "verified_evidence": [], "rejected_candidates": [],
            "unresolved_criteria": []}


def _merge_research(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = _empty_research()
    for field in result:
        if field == "unresolved_criteria" and field in update:
            values = list(update.get(field, []))
        else:
            values = list(base.get(field, [])) + list(update.get(field, []))
        seen, merged = set(), []
        for item in values:
            marker = json.dumps(item, ensure_ascii=False, sort_keys=True)
            if marker not in seen:
                seen.add(marker)
                merged.append(item)
        result[field] = merged[-60:]
    return result


def _research_signature(value: dict[str, Any]) -> str:
    # Query count alone is not progress. Material progress means candidates,
    # evidence, eliminations or unresolved constraints changed.
    return _digest({key: value.get(key, []) for key in (
        "candidates", "verified_evidence", "rejected_candidates",
        "unresolved_criteria",
    )})


def _verdict(value: Any, subtask: SubtaskSpec) -> dict[str, Any]:
    if not isinstance(value, dict):
        return _failed_verdict(subtask, "Verifier output was not an object")
    failed = [str(item) for item in value.get("failed_criteria", [])]
    accepted = bool(value.get("accepted", False)) and not failed
    return {"accepted": accepted, "critique": str(value.get("critique", "")),
            "satisfied_criteria": [str(x) for x in value.get("satisfied_criteria", [])],
            "failed_criteria": failed,
            "verified_evidence": list(value.get("verified_evidence", [])),
            "confidence": max(0.0, min(1.0, float(value.get("confidence", 0.0))))}


def _failed_verdict(subtask: SubtaskSpec, reason: str) -> dict[str, Any]:
    return {"accepted": False, "critique": reason, "satisfied_criteria": [],
            "failed_criteria": list(subtask.success_criteria),
            "verified_evidence": [], "confidence": 0.0}


def _topological(plan: SubtaskPlanV1) -> list[SubtaskSpec]:
    pending, ordered = list(plan.subtasks), []
    done: set[str] = set()
    while pending:
        ready = [item for item in pending if set(item.dependencies) <= done]
        if not ready:
            raise RuntimeError("validated DAG has no executable frontier")
        for item in ready:
            pending.remove(item)
            ordered.append(item)
            done.add(item.id)
    return ordered


def _round_cap(ctx: OrchestrationContext, subtask: SubtaskSpec) -> int:
    configured = ctx.metadata.get("max_rounds_per_subtask")
    if isinstance(configured, int) and configured > 0:
        return configured
    return _ROUND_CAP[subtask.difficulty]


def _tools(subtask: SubtaskSpec) -> list[str]:
    return [item for item in subtask.tools if item in _KNOWN_TOOLS]


def _summary(state: dict[str, Any]) -> dict[str, Any]:
    research = state.get("research", {})
    return {"status": state.get("status"), "solution": state.get("solution", ""),
            "critique": state.get("critique", ""),
            "candidates": research.get("candidates", []),
            "verified_evidence": research.get("verified_evidence", []),
            "unresolved_criteria": research.get("unresolved_criteria", []),
            "verdict": state.get("verdict", {})}


def _digest(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


async def _save(ctx: OrchestrationContext, plan: SubtaskPlanV1,
                states: dict[str, dict[str, Any]], *,
                final_answer: str | None = None) -> None:
    await ctx.checkpoint("reference_v6_state", {
        "plan": plan.to_dict(), "states": states, "final_answer": final_answer,
    })


def _identity(ctx: OrchestrationContext) -> dict[str, Any]:
    return {"script_id": ctx.metadata.get("script_id", "reference_v6"),
            "version": ctx.metadata.get("script_version", 6),
            "source_hash": ctx.metadata.get("script_source_hash", "")}


def _decomposer_prompt(task: str, critique: str) -> str:
    return (
        "将原题拆成最多3个互补子问题DAG。不得增加原题不存在的数量、穷举规模、"
        "证据门槛或输出要求。每个节点必须能够在上游incomplete时利用已有线索并自行补全。"
        "success_criteria只复述原题的原子要求。tools只允许web_search、web_fetch、"
        f"run_python_code。上一版计划审查意见：{critique or '无'}\n\n原题：\n{task}"
    )


def _review_prompt(task: str, plan: SubtaskPlanV1) -> str:
    return (f"原题：\n{task}\n\n待审查DAG：\n"
            f"{json.dumps(plan.to_dict(), ensure_ascii=False)}\n\n"
            "重点检查是否擅自增加穷举数量、遗漏条件、重复完整原题或制造不必要阻塞。")


def _solver_prompt(task: str, subtask: SubtaskSpec,
                   dependencies: dict[str, dict[str, Any]], state: dict[str, Any],
                   round_index: int) -> str:
    return (
        f"原题：\n{task}\n\n当前子问题：\n"
        f"{json.dumps(subtask.__dict__, ensure_ascii=False)}\n\n"
        f"依赖结果只是研究线索，不是执行许可；即使incomplete也要主动补全：\n"
        f"{json.dumps({k: _summary(v) for k, v in dependencies.items()}, ensure_ascii=False)}\n\n"
        f"持久研究状态（禁止重复其中查询）：\n{json.dumps(state['research'], ensure_ascii=False)}\n\n"
        f"上一轮solution：{state['solution']}\nVerifier critique：{state['critique']}\n"
        f"当前第{round_index + 1}轮。优先使用最有区分度的联合约束检索；返回严格JSON。"
    )


def _verifier_prompt(task: str, subtask: SubtaskSpec, solution: str,
                     research: dict[str, Any]) -> str:
    return (f"原题：\n{task}\n\n只验证当前子问题：\n{subtask.question}\n"
            f"成功标准：{json.dumps(subtask.success_criteria, ensure_ascii=False)}\n"
            f"solution：{solution}\n结构化研究状态："
            f"{json.dumps(research, ensure_ascii=False)}\n"
            "逐项验证并给出可执行critique；不得因JSON合法而accepted。")


def _synthesis_prompt(task: str, plan: SubtaskPlanV1,
                      states: dict[str, dict[str, Any]]) -> str:
    compact = {key: _summary(value) for key, value in states.items()}
    return (f"原题：\n{task}\n\nDAG：{json.dumps(plan.to_dict(), ensure_ascii=False)}\n"
            f"各节点压缩状态：{json.dumps(compact, ensure_ascii=False)}\n\n"
            "使用accepted事实和incomplete中的已验证局部事实；排除明确证伪候选。"
            "只输出benchmark要求的最短答案；确无依据才输出信息未查全。")
