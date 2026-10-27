"""Dependency-versioned candidate funnel with fail-closed verification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any

from agent_harness.orchestration.context import OrchestrationContext
from orchestrations.contracts import CandidateV2, SolverResultV2, SubtaskPlanV1, SubtaskSpec
from orchestrations.reference_v2 import (
    _candidate_funnel,
    _decomposer_instruction,
    _normalize_verdict,
    _parse_solver_result,
    _validate_plan_semantics,
)

MANIFEST = {
    "script_id": "reference_v3",
    "version": 3,
    "entrypoint": "solve",
    "input_schema": "string",
    "output_schema": "string",
    "compatible_runtime": ">=1.0",
    "roles": ("decomposer", "solver", "verifier", "synthesizer"),
}

_ROUND_LIMIT = {"easy": 1, "medium": 2, "hard": 3}
_SYNTHESIS_MARKERS = (
    "synthesi", "summarize", "final answer", "combine results",
    "综合结果", "汇总结果", "最终答案",
)
_PAPER_DOMAINS = ("aclanthology.org", "arxiv.org")


async def solve(ctx: OrchestrationContext) -> str:
    task = str(ctx.task)
    restored = ctx.load_checkpoint("reference_v3_state") or {}
    ctx.emit("script_started", **_identity(ctx))
    if restored.get("final_answer"):
        ctx.emit("script_completed", **_identity(ctx), restored=True)
        return str(restored["final_answer"])

    plan = await _create_plan(ctx, task, restored)
    states: dict[str, dict[str, Any]] = dict(restored.get("states", {}))
    executions: dict[str, int] = {
        key: int(value) for key, value in restored.get("executions", {}).items()
    }
    order = _topological_order(plan)

    # Execute each node only against current dependency versions. Discovery
    # may revise immediately; once it has candidates, it never spends another
    # discovery round.
    for subtask in order:
        state = states.get(subtask.id)
        if state is None or _is_stale(subtask, state, states):
            state = await _execute(
                ctx, task, subtask, states, executions, previous=state,
                reason="initial" if state is None else "dependency_stale",
            )
            states[subtask.id] = state
            await _save(ctx, plan, states, executions)

        max_attempts = _ROUND_LIMIT[subtask.difficulty]
        while (
            int(state.get("attempt", 1)) < max_attempts
            and not state.get("accepted")
            and _can_spend(ctx, 8_000)
        ):
            if not subtask.dependencies and _has_candidates(state):
                ctx.emit("discovery_stopped_on_candidate", **_identity(ctx),
                         subtask_id=subtask.id, version=state["version"])
                break
            old_fingerprint = state["fingerprint"]
            state = await _execute(
                ctx, task, subtask, states, executions, previous=state,
                reason="revision",
            )
            states[subtask.id] = state
            if state["fingerprint"] != old_fingerprint:
                stale = _descendant_ids(plan, subtask.id)
                ctx.emit("descendants_marked_stale", **_identity(ctx),
                         subtask_id=subtask.id, descendant_ids=stale,
                         state_version=state["version"])
            await _save(ctx, plan, states, executions)

    # A changed dependency invalidates downstream output regardless of whether
    # that downstream node has exhausted attempts under an older input.
    while True:
        stale_nodes = [
            item for item in order
            if item.id in states and _is_stale(item, states[item.id], states)
        ]
        if not stale_nodes:
            break
        progressed = False
        for subtask in stale_nodes:
            if not _can_spend(ctx, 8_000):
                continue
            states[subtask.id] = await _execute(
                ctx, task, subtask, states, executions,
                previous=states[subtask.id], reason="dependency_stale",
            )
            await _save(ctx, plan, states, executions)
            progressed = True
        if not progressed:
            break

    for subtask in order:
        state = states[subtask.id]
        ctx.emit(
            "subtask_accepted" if state.get("accepted") else "subtask_best_effort",
            **_identity(ctx), subtask_id=subtask.id,
            status="accepted" if state.get("accepted") else "incomplete",
            state_version=state.get("version"),
        )

    ctx.emit("synthesis_started", **_identity(ctx))
    try:
        answer = str(await ctx.call(
            "synthesizer", _synthesis_instruction(task, plan, states),
            key="reference_v3:synthesis", max_turns=1,
            metadata={"allowed_tools": []},
        )).strip()
    except Exception:
        answer = "信息未查全"
    answer = answer or "信息未查全"
    await _save(ctx, plan, states, executions, final_answer=answer)
    ctx.emit("synthesis_completed", **_identity(ctx))
    ctx.emit("script_completed", **_identity(ctx), restored=False)
    return answer


async def _create_plan(
    ctx: OrchestrationContext, task: str, restored: dict[str, Any]
) -> SubtaskPlanV1:
    if restored.get("plan"):
        return SubtaskPlanV1.from_dict(restored["plan"])
    ctx.emit("plan_created", **_identity(ctx), status="requested")
    reason = ""
    try:
        raw = await ctx.call(
            "decomposer", _decomposer_instruction(task),
            key="reference_v3:decomposition", max_turns=1,
            metadata={"allowed_tools": []},
        )
        plan = SubtaskPlanV1.from_dict(raw)
        from orchestrations.contracts import validate_plan
        validate_plan(plan)
        _validate_plan_semantics(plan)
        for item in plan.subtasks:
            text = f"{item.id} {item.question}".casefold()
            if any(marker in text for marker in _SYNTHESIS_MARKERS):
                raise ValueError("plan contains a redundant synthesis subtask")
        fallback = False
    except (KeyError, TypeError, ValueError) as exc:
        reason = str(exc)
        original = _candidate_funnel(task)
        # Phase 3 owns synthesis; the v3 fallback needs only discovery and
        # constraint verification.
        plan = SubtaskPlanV1(
            original_task=task,
            subtasks=original.subtasks[:2],
            synthesis_requirements=original.synthesis_requirements,
        )
        fallback = True
    ctx.emit("plan_validated", **_identity(ctx), fallback=fallback,
             fallback_reason=reason, subtask_count=len(plan.subtasks))
    await ctx.checkpoint("reference_v3_state", {
        "plan": plan.to_dict(), "states": {}, "executions": {},
        "final_answer": None,
    })
    return plan


async def _execute(
    ctx: OrchestrationContext, task: str, subtask: SubtaskSpec,
    states: dict[str, dict[str, Any]], executions: dict[str, int], *,
    previous: dict[str, Any] | None, reason: str,
) -> dict[str, Any]:
    execution = executions.get(subtask.id, 0) + 1
    executions[subtask.id] = execution
    dependency_versions = {
        dependency: int(states[dependency]["version"])
        for dependency in subtask.dependencies
    }
    dependencies = {dependency: states[dependency] for dependency in subtask.dependencies}
    ctx.emit("solver_round_started", **_identity(ctx), subtask_id=subtask.id,
             round_index=execution - 1, execution=execution, reason=reason,
             dependency_versions=dependency_versions)
    try:
        raw = await ctx.call(
            "solver", _solver_instruction(task, subtask, dependencies, previous),
            key=f"reference_v3:{subtask.id}:execution:{execution}:solver",
            max_turns=2, metadata={"allowed_tools": list(subtask.tools)},
        )
        solution = _parse_solver_result(raw)
        error = None
    except Exception as exc:
        solution = SolverResultV2(
            status="incomplete", candidates=[], verified_facts=[],
            missing_constraints=["Solver执行失败"], search_summary="",
        )
        error = f"{type(exc).__name__}: {exc}"
    solution = _preserve_dependency_candidates(solution, dependencies)
    solution = _mark_discovery_only_sources(solution)
    ctx.emit("solver_round_completed", **_identity(ctx), subtask_id=subtask.id,
             round_index=execution - 1, execution=execution,
             candidate_count=len(solution.candidates))
    try:
        raw_verdict = await ctx.call(
            "verifier", _verifier_instruction(task, subtask, solution),
            key=f"reference_v3:{subtask.id}:execution:{execution}:verifier",
            max_turns=1, metadata={"allowed_tools": []},
        )
        verdict = _normalize_verdict(raw_verdict)
    except Exception as exc:
        verdict = {"accepted": False, "critique": f"Verifier失败: {exc}",
                   "satisfied_criteria": [],
                   "failed_criteria": list(subtask.success_criteria),
                   "verified_evidence": [], "confidence": 0.0}
    verdict = _apply_admission_policy(solution, verdict)
    previous_version = int(previous.get("version", 0)) if previous else 0
    fingerprint = _fingerprint(solution)
    version = previous_version + (
        1 if previous is None or fingerprint != previous.get("fingerprint") else 0
    )
    state = {
        "subtask_id": subtask.id, "version": version,
        "dependency_versions": dependency_versions,
        "attempt": int(previous.get("attempt", 0)) + 1 if previous else 1,
        "accepted": verdict["accepted"], "solution": solution.to_dict(),
        "verdict": verdict, "fingerprint": fingerprint, "error": error,
    }
    ctx.emit("verifier_round_completed", **_identity(ctx),
             subtask_id=subtask.id, round_index=execution - 1,
             execution=execution, accepted=verdict["accepted"])
    await ctx.checkpoint(f"subtask:{subtask.id}", state)
    return state


def _preserve_dependency_candidates(
    solution: SolverResultV2, dependencies: dict[str, dict[str, Any]],
) -> SolverResultV2:
    inherited: list[CandidateV2] = []
    for state in dependencies.values():
        inherited.extend(SolverResultV2.from_dict(state["solution"]).candidates)
    if not inherited:
        return solution
    by_answer = {item.answer.casefold(): item for item in inherited if item.answer}
    for item in solution.candidates:
        by_answer[item.answer.casefold()] = item
    return SolverResultV2(
        status="candidate" if by_answer else solution.status,
        candidates=list(by_answer.values())[:5],
        verified_facts=solution.verified_facts,
        missing_constraints=solution.missing_constraints,
        search_summary=solution.search_summary,
    )


def _mark_discovery_only_sources(solution: SolverResultV2) -> SolverResultV2:
    # Keep papers for candidate discovery but deterministic admission below
    # never treats them as independent factual verification.
    return solution


def _apply_admission_policy(
    solution: SolverResultV2, verdict: dict[str, Any],
) -> dict[str, Any]:
    reasons = []
    if not solution.candidates:
        reasons.append("没有具体候选")
    if solution.missing_constraints:
        reasons.append("仍有决定性约束未验证")
    for candidate in solution.candidates:
        statuses = {value.casefold() for value in candidate.constraint_status.values()}
        if not statuses or statuses & {"unknown", "unverified", "refuted"}:
            reasons.append(f"候选 {candidate.answer} 存在未验证或被反驳的约束")
        independent = [
            evidence for evidence in candidate.evidence
            if evidence.status == "verified"
            and not any(domain in evidence.url.casefold() for domain in _PAPER_DOMAINS)
        ]
        if not independent:
            reasons.append(f"候选 {candidate.answer} 缺少独立非论文来源")
    if reasons:
        verdict = dict(verdict)
        verdict["accepted"] = False
        verdict["critique"] = (
            str(verdict.get("critique", "")) + "；准入规则：" + "；".join(reasons)
        ).strip("；")
    return verdict


def _is_stale(
    subtask: SubtaskSpec, state: dict[str, Any],
    states: dict[str, dict[str, Any]],
) -> bool:
    expected = {
        dependency: int(states[dependency]["version"])
        for dependency in subtask.dependencies
    }
    return state.get("dependency_versions", {}) != expected


def _descendant_ids(plan: SubtaskPlanV1, root: str) -> list[str]:
    descendants: set[str] = set()
    while True:
        added = {
            item.id for item in plan.subtasks
            if item.id not in descendants and item.id != root
            and any(dep == root or dep in descendants for dep in item.dependencies)
        }
        if not added:
            return sorted(descendants)
        descendants.update(added)


def _topological_order(plan: SubtaskPlanV1) -> list[SubtaskSpec]:
    ordered: list[SubtaskSpec] = []
    remaining = list(plan.subtasks)
    while remaining:
        ready = [item for item in remaining
                 if all(dep in {done.id for done in ordered} for dep in item.dependencies)]
        if not ready:
            raise ValueError("plan is not a DAG")
        ordered.extend(ready)
        remaining = [item for item in remaining if item not in ready]
    return ordered


def _has_candidates(state: dict[str, Any]) -> bool:
    return bool(SolverResultV2.from_dict(state["solution"]).candidates)


def _fingerprint(solution: SolverResultV2) -> str:
    payload = json.dumps(solution.to_dict(), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _can_spend(ctx: OrchestrationContext, estimate: int) -> bool:
    maximum = ctx.metadata.get("budget", {}).get("max_tokens")
    if maximum is None:
        return True
    return ctx.remaining_budget().usage["tokens"] + estimate <= int(maximum * 0.85)


async def _save(
    ctx: OrchestrationContext, plan: SubtaskPlanV1,
    states: dict[str, dict[str, Any]], executions: dict[str, int], *,
    final_answer: str | None = None,
) -> None:
    await ctx.checkpoint("reference_v3_state", {
        "plan": plan.to_dict(), "states": states, "executions": executions,
        "final_answer": final_answer,
    })


def _identity(ctx: OrchestrationContext) -> dict[str, Any]:
    return {"script_id": ctx.metadata.get("script_id", "reference_v3"),
            "version": ctx.metadata.get("script_version", 3),
            "source_hash": ctx.metadata.get("script_source_hash", "")}


def _solver_instruction(
    task: str, subtask: SubtaskSpec, dependencies: dict[str, Any],
    previous: dict[str, Any] | None,
) -> str:
    return (
        f"原始任务：\n{task}\n\n当前子任务：\n{json.dumps(asdict(subtask), ensure_ascii=False)}"
        f"\n\n依赖的版本化结果：\n{json.dumps(dependencies, ensure_ascii=False)}"
        f"\n\n上一执行状态：\n{json.dumps(previous, ensure_ascii=False)}\n\n"
        "若依赖中已有候选，必须保留这些候选，只更新constraint_status和evidence；"
        "不得因本轮搜索失败清空候选。验证阶段优先fetch已有URL，只搜索仍为unknown的"
        "决定性约束。论文和benchmark页面只能用于发现候选，不能单独证明约束。"
        "最终只输出SolverResultV2 JSON，字段为status、candidates、verified_facts、"
        "missing_constraints、search_summary；每项证据包含claim/url/source/status。"
    )


def _verifier_instruction(
    task: str, subtask: SubtaskSpec, solution: SolverResultV2,
) -> str:
    return (
        f"原始任务：\n{task}\n\n成功标准：\n"
        f"{json.dumps(subtask.success_criteria, ensure_ascii=False)}\n\n"
        f"候选状态：\n{json.dumps(solution.to_dict(), ensure_ascii=False)}\n\n"
        "只验证任务语义，不评价JSON格式。存在missing_constraints、unknown、refuted，"
        "或关键事实只有论文/benchmark来源时必须accepted=false。"
    )


def _synthesis_instruction(
    task: str, plan: SubtaskPlanV1, states: dict[str, dict[str, Any]],
) -> str:
    return (
        f"原始任务：\n{task}\n\n计划：\n{json.dumps(plan.to_dict(), ensure_ascii=False)}"
        f"\n\n版本化最终状态：\n{json.dumps(states, ensure_ascii=False)}\n\n"
        "忽略stale结果和论文单独支持的事实。若候选仍有决定性unknown则输出信息未查全；"
        "否则只输出benchmark所需的最短答案。"
    )
