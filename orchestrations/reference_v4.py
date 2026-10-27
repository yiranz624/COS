"""Constraint-driven retrieval with deterministic evidence admission.

Unlike the generic DAG workflows in reference_v1-v3, this script makes the
retrieval state machine explicit: extract constraints, discover candidates by
two complementary routes, verify every required criterion, then synthesize.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from agent_harness.orchestration.context import OrchestrationContext
from orchestrations.contracts import CandidateV2, SubtaskPlanV1
from orchestrations.reference_v2 import _parse_solver_result

MANIFEST = {
    "script_id": "reference_v4",
    "version": 4,
    "entrypoint": "solve",
    "input_schema": "string",
    "output_schema": "string",
    "compatible_runtime": ">=1.0",
    "roles": ("decomposer", "solver", "verifier", "synthesizer"),
}

_PAPER_DOMAINS = ("arxiv.org", "aclanthology.org", "doi.org")
_CRITERION_ID = re.compile(r"\bC\d+\b", re.I)
_DISCOVERY_ROUTES = (
    ("rare_constraint", "用两个至三个最稀有的数量、时间和身份约束做联合查询"),
    ("entity_pivot", "从稀有身份或事件实体切入，发现候选后改用候选名称反向查询"),
)


@dataclass(frozen=True)
class ConstraintV4:
    id: str
    description: str
    required_evidence: str


async def solve(ctx: OrchestrationContext) -> str:
    task = str(ctx.task)
    identity = _identity(ctx)
    restored = ctx.load_checkpoint("reference_v4_state") or {}
    ctx.emit("script_started", **identity)
    if restored.get("final_answer"):
        ctx.emit("script_completed", **identity, restored=True)
        return str(restored["final_answer"])

    constraints = await _extract_constraints(ctx, task, restored)
    candidates = _candidates_from_dicts(restored.get("candidates", []))
    if not candidates:
        candidates = await _discover(ctx, task, constraints)
        await _save(ctx, constraints, candidates, [], None)

    # Verification without a concrete candidate is meaningless. Reassign the
    # downstream budget to one route-changing fallback rather than launching
    # empty dependent jobs.
    if not candidates and _can_spend(ctx, 10_000):
        ctx.emit("discovery_fallback_started", **identity,
                 reason="primary_routes_returned_no_candidate")
        candidates = await _discovery_call(
            ctx, task, constraints, "fallback_dimension_shift",
            "此前路线没有候选。改换检索维度：使用题目中的事件短语、统计同义词和"
            "可能的权威数据库；不得重复此前查询。",
            key="reference_v4:discovery:fallback",
        )
        await _save(ctx, constraints, candidates, [], None)

    if not candidates:
        ctx.emit("verification_blocked", **identity,
                 reason="no_candidate", candidate_count=0)
        answer = "信息未查全"
        await _save(ctx, constraints, [], [], answer)
        ctx.emit("script_completed", **identity, restored=False)
        return answer

    ctx.emit("discovery_stopped_on_candidate", **identity,
             candidate_count=len(candidates))
    verifications = list(restored.get("verifications", []))
    if not verifications:
        verifications = await ctx.map(
            lambda candidate: _verify_candidate(
                ctx, task, constraints, candidate,
            ),
            candidates,
        )
        await _save(ctx, constraints, candidates, verifications, None)

    accepted = [item for item in verifications if item["accepted"]]
    if not accepted:
        ctx.emit("synthesis_blocked", **identity,
                 reason="no_fully_verified_candidate")
        answer = "信息未查全"
    else:
        ctx.emit("synthesis_started", **identity,
                 accepted_candidates=[item["answer"] for item in accepted])
        try:
            answer = str(await ctx.call(
                "synthesizer",
                _synthesis_instruction(task, constraints, accepted),
                key="reference_v4:synthesis", max_turns=1,
                metadata={"allowed_tools": []},
            )).strip() or "信息未查全"
        except Exception:
            answer = "信息未查全"
        ctx.emit("synthesis_completed", **identity)
    await _save(ctx, constraints, candidates, verifications, answer)
    ctx.emit("script_completed", **identity, restored=False)
    return answer


async def _extract_constraints(
    ctx: OrchestrationContext, task: str, restored: dict[str, Any],
) -> list[ConstraintV4]:
    if restored.get("constraints"):
        return [ConstraintV4(**item) for item in restored["constraints"]]
    ctx.emit("constraint_extraction_started", **_identity(ctx))
    try:
        raw = await ctx.call(
            "decomposer", _constraint_instruction(task),
            key="reference_v4:constraints", max_turns=1,
            metadata={"allowed_tools": []},
        )
        plan = SubtaskPlanV1.from_dict(raw)
        descriptions: list[tuple[str, str]] = []
        for subtask in plan.subtasks:
            evidence = "；".join(subtask.required_evidence) or "可核验来源URL"
            for criterion in subtask.success_criteria:
                descriptions.append((criterion.strip(), evidence))
        if not descriptions:
            raise ValueError("decomposer returned no success criteria")
        constraints = [
            ConstraintV4(f"C{index}", description, evidence)
            for index, (description, evidence) in enumerate(descriptions[:8], 1)
        ]
        fallback = False
    except (KeyError, TypeError, ValueError):
        constraints = [ConstraintV4(
            "C1", "候选必须满足原题中的全部显式身份、数量、时间和事件约束",
            "能够核验最终答案关键事实的非论文来源URL",
        )]
        fallback = True
    ctx.emit("constraint_extraction_completed", **_identity(ctx),
             constraint_count=len(constraints), fallback=fallback)
    await _save(ctx, constraints, [], [], None)
    return constraints


async def _discover(
    ctx: OrchestrationContext, task: str, constraints: list[ConstraintV4],
) -> list[CandidateV2]:
    ctx.emit("discovery_routes_started", **_identity(ctx),
             routes=[item[0] for item in _DISCOVERY_ROUTES])
    results = await ctx.map(
        lambda route: _discovery_call(
            ctx, task, constraints, route[0], route[1],
            key=f"reference_v4:discovery:{route[0]}",
        ),
        list(_DISCOVERY_ROUTES),
    )
    merged = _merge_candidates([item for result in results for item in result])
    ctx.emit("discovery_routes_completed", **_identity(ctx),
             candidate_count=len(merged))
    return merged


async def _discovery_call(
    ctx: OrchestrationContext, task: str, constraints: list[ConstraintV4],
    route_id: str, route_instruction: str, *, key: str,
) -> list[CandidateV2]:
    ctx.emit("discovery_route_started", **_identity(ctx), route_id=route_id)
    try:
        raw = await ctx.call(
            "solver", _discovery_instruction(
                task, constraints, route_id, route_instruction,
            ),
            key=key, max_turns=4,
            metadata={"allowed_tools": ["web_search", "web_fetch"]},
        )
        result = _parse_solver_result(raw)
        candidates = [item for item in result.candidates if item.answer][:5]
    except Exception as exc:
        candidates = []
        ctx.emit("discovery_route_failed", **_identity(ctx), route_id=route_id,
                 error=f"{type(exc).__name__}: {exc}")
    ctx.emit("discovery_route_completed", **_identity(ctx), route_id=route_id,
             candidate_count=len(candidates))
    return candidates


async def _verify_candidate(
    ctx: OrchestrationContext, task: str, constraints: list[ConstraintV4],
    candidate: CandidateV2,
) -> dict[str, Any]:
    answer = candidate.answer
    ctx.emit("candidate_verification_started", **_identity(ctx), answer=answer)
    try:
        raw = await ctx.call(
            "verifier", _verification_instruction(task, constraints, candidate),
            key=f"reference_v4:verify:{_safe_key(answer)}", max_turns=3,
            metadata={"allowed_tools": ["web_search", "web_fetch"]},
        )
        verdict = _normalize_verification(raw)
    except Exception as exc:
        verdict = {"satisfied_criteria": [], "failed_criteria": [],
                   "verified_evidence": [], "critique": f"Verifier失败: {exc}",
                   "confidence": 0.0}
    admitted, statuses, reasons = _programmatic_admission(
        constraints, candidate, verdict,
    )
    result = {
        "answer": answer, "accepted": admitted,
        "criterion_status": statuses, "reasons": reasons,
        "verdict": verdict, "candidate": asdict(candidate),
    }
    ctx.emit("candidate_verification_completed", **_identity(ctx),
             answer=answer, accepted=admitted, criterion_status=statuses,
             rejection_reasons=reasons)
    await ctx.checkpoint(f"candidate:{_safe_key(answer)}", result)
    return result


def _programmatic_admission(
    constraints: list[ConstraintV4], candidate: CandidateV2,
    verdict: dict[str, Any],
) -> tuple[bool, dict[str, str], list[str]]:
    satisfied = _criterion_ids(verdict.get("satisfied_criteria", []))
    failed = _criterion_ids(verdict.get("failed_criteria", []))
    statuses = {
        item.id: ("contradicted" if item.id in failed else
                  "supported" if item.id in satisfied else "unknown")
        for item in constraints
    }
    reasons = [f"{key}={value}" for key, value in statuses.items()
               if value != "supported"]
    evidence = list(candidate.evidence) + [
        item for item in verdict.get("verified_evidence", [])
        if isinstance(item, dict)
    ]
    valid_urls = [
        str(item.url if hasattr(item, "url") else item.get("url", ""))
        for item in evidence
        if str(item.url if hasattr(item, "url") else item.get("url", "")).startswith("http")
    ]
    independent = [url for url in valid_urls
                   if not any(domain in url.casefold() for domain in _PAPER_DOMAINS)]
    if not independent:
        reasons.append("缺少独立非论文事实来源")
    return not reasons, statuses, reasons


def _normalize_verification(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("verifier result must be an object")
    return {
        "satisfied_criteria": [str(x) for x in value.get("satisfied_criteria", [])],
        "failed_criteria": [str(x) for x in value.get("failed_criteria", [])],
        "verified_evidence": [x for x in value.get("verified_evidence", [])
                              if isinstance(x, dict)],
        "critique": str(value.get("critique", "")),
        "confidence": max(0.0, min(1.0, float(value.get("confidence", 0)))),
    }


def _criterion_ids(values: list[Any]) -> set[str]:
    return {match.group(0).upper() for value in values
            if (match := _CRITERION_ID.search(str(value)))}


def _merge_candidates(candidates: list[CandidateV2]) -> list[CandidateV2]:
    merged: dict[str, CandidateV2] = {}
    for candidate in candidates:
        key = " ".join(candidate.answer.casefold().split())
        if not key:
            continue
        previous = merged.get(key)
        if previous is None:
            merged[key] = candidate
            continue
        evidence = {item.url: item for item in previous.evidence}
        evidence.update({item.url: item for item in candidate.evidence})
        merged[key] = CandidateV2(
            answer=previous.answer,
            constraint_status={**previous.constraint_status,
                               **candidate.constraint_status},
            evidence=list(evidence.values()),
        )
    # Three candidates keep worst-case verification inside the 72k envelope.
    # Discovery may report five for recall, but only its highest-ranked unique
    # rows enter the expensive fetch-and-verify stage.
    return list(merged.values())[:3]


def _candidates_from_dicts(values: list[Any]) -> list[CandidateV2]:
    return [CandidateV2.from_dict(item) for item in values if isinstance(item, dict)]


def _discovery_instruction(
    task: str, constraints: list[ConstraintV4], route_id: str,
    route_instruction: str,
) -> str:
    return (
        f"原始问题：\n{task}\n\n必须覆盖的约束：\n"
        f"{json.dumps([asdict(item) for item in constraints], ensure_ascii=False)}\n\n"
        f"检索路线 {route_id}：{route_instruction}\n"
        "执行固定状态机：联合约束宽搜→发现具体候选→立即用候选名称定向搜索→"
        "fetch候选页面→输出候选表。出现候选后停止继续宽泛发现。不得添加题目未给出的"
        "年份限制；连续两次查询无新增候选或证据时停止。论文只能发现候选。"
        "你有足够的turn完成search→fetch→结构化输出。最终只输出SolverResultV2 JSON："
        "status、candidates、verified_facts、missing_constraints、search_summary；"
        "candidate包含answer、constraint_status和evidence，证据保留URL。"
    )


def _constraint_instruction(task: str) -> str:
    return (
        "使用既有SubtaskPlan schema提取原题约束，不设计最终答案或综合节点。最多3个"
        "subtask；每条success_criteria必须是原题中的一个原子、必需、可核查约束，"
        "例如身份、数量、时间、事件或输出格式，不得添加推测年份。若相关约束很多，"
        "可在一个subtask中列出多条success_criteria，总数最多8条。required_evidence"
        "说明验证该组约束所需的来源。输出严格的SubtaskPlan JSON。\n\n"
        f"原始任务：\n{task}"
    )


def _verification_instruction(
    task: str, constraints: list[ConstraintV4], candidate: CandidateV2,
) -> str:
    return (
        f"原始问题：\n{task}\n\n待核查候选：\n"
        f"{json.dumps(asdict(candidate), ensure_ascii=False)}\n\n逐项约束：\n"
        f"{json.dumps([asdict(item) for item in constraints], ensure_ascii=False)}\n\n"
        "只核查这个候选，不重新做无边界候选发现。优先fetch已有URL，并对缺失约束"
        "使用候选名称做定向搜索。satisfied_criteria必须逐项写成'C1: ...'；"
        "failed_criteria写明确矛盾或无法验证的ID。verified_evidence每项必须含claim、"
        "url、source、status。论文或搜索摘要不能单独支持事实。模型的accepted字段"
        "不会被采信，Python将依据全部C编号与URL重新计算。"
    )


def _synthesis_instruction(
    task: str, constraints: list[ConstraintV4], accepted: list[dict[str, Any]],
) -> str:
    return (
        f"原始问题：\n{task}\n\n约束：\n"
        f"{json.dumps([asdict(item) for item in constraints], ensure_ascii=False)}\n\n"
        f"通过程序化逐项验证的候选：\n{json.dumps(accepted, ensure_ascii=False)}\n\n"
        "只能从这些已接受候选生成benchmark要求的最短答案，不要解释，不要增加新事实。"
    )


async def _save(
    ctx: OrchestrationContext, constraints: list[ConstraintV4],
    candidates: list[CandidateV2], verifications: list[dict[str, Any]],
    final_answer: str | None,
) -> None:
    await ctx.checkpoint("reference_v4_state", {
        "constraints": [asdict(item) for item in constraints],
        "candidates": [asdict(item) for item in candidates],
        "verifications": verifications, "final_answer": final_answer,
    })


def _can_spend(ctx: OrchestrationContext, estimate: int) -> bool:
    maximum = ctx.metadata.get("budget", {}).get("max_tokens")
    if maximum is None:
        return True
    return ctx.remaining_budget().usage["tokens"] + estimate <= int(maximum * 0.85)


def _identity(ctx: OrchestrationContext) -> dict[str, Any]:
    return {"script_id": ctx.metadata.get("script_id", "reference_v4"),
            "version": ctx.metadata.get("script_version", 4),
            "source_hash": ctx.metadata.get("script_source_hash", "")}


def _safe_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")[:60] or "candidate"
