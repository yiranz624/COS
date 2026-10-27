"""Comparable single, fixed multi-agent, and dynamically patched strategies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any, Protocol

from agent_harness.orchestration.contracts import (
    BudgetSpec,
    BudgetUsage,
    StrategyKind,
)
from agent_harness.orchestration.patching import (
    OrchestrationPatchV1,
    OrchestrationPlan,
    PatchOperation,
    PatchValidator,
    PlanNode,
)
from agent_harness.orchestration.runtime import ScriptRuntime
from agent_harness.orchestration.trace import RunTraceV1, TraceEventV1
from agent_harness.orchestration.types import RunLimits, ScriptRunResult

_ROLES = {
    "researcher", "solver", "verifier", "synthesizer", "general",
}


@dataclass(frozen=True)
class StrategyRunResult:
    run_id: str
    strategy: str
    value: Any
    runtime_result: ScriptRunResult
    trace: RunTraceV1
    budget_violations: tuple[str, ...]
    patches_proposed: int = 0
    patches_applied: int = 0


@dataclass(frozen=True)
class SolutionBranch:
    """One independent route for answering the complete original task."""

    objective: str
    requires_web: bool
    approach: str

    @property
    def role(self) -> str:
        return "researcher" if self.requires_web else "solver"


def _branch_job_key(
    stage: str, iteration: int, branch: SolutionBranch,
) -> str:
    """Build a collision-resistant key for one logical branch invocation."""
    identity_source = "\0".join((
        branch.role, branch.objective, branch.approach,
    ))
    entity_id = hashlib.sha256(
        identity_source.encode("utf-8"),
    ).hexdigest()[:12]
    return f"{stage}:{iteration}:branch-{entity_id}"


@dataclass(frozen=True)
class StageBudgetPolicy:
    """Token reservations for the dynamic PoC's stage boundaries."""

    discovery_limit: float = 0.60
    verification_limit: float = 0.78
    revision_limit: float = 0.88
    synthesis_reserve: float = 0.12


class OrchestrationStrategy(Protocol):
    kind: StrategyKind

    async def run(
        self,
        executor: Any,
        *,
        task: str,
        run_id: str,
        budget: BudgetSpec,
        metadata: dict[str, Any] | None = None,
    ) -> StrategyRunResult: ...


class _BaseStrategy:
    kind: StrategyKind

    async def run(
        self,
        executor: Any,
        *,
        task: str,
        run_id: str,
        budget: BudgetSpec,
        metadata: dict[str, Any] | None = None,
    ) -> StrategyRunResult:
        stage_events: list[dict[str, Any]] = []
        trace_dir_raw = (metadata or {}).get("trace_dir")
        trace_dir = Path(trace_dir_raw) if trace_dir_raw else None
        live_path = trace_dir / "orchestration_live.jsonl" if trace_dir else None
        stage_path = trace_dir / "stage_trace.jsonl" if trace_dir else None
        if live_path is not None:
            live_path.unlink(missing_ok=True)
        if stage_path is not None:
            stage_path.unlink(missing_ok=True)

        def stage(name: str, event_type: str = "stage_boundary", **payload: Any) -> None:
            record = {
                "timestamp": time.time(),
                "stage": name,
                "event_type": event_type,
                "payload": payload,
            }
            stage_events.append(record)
            if stage_path is not None:
                descriptions = {
                    "stage_boundary": "到达编排阶段边界",
                    "patch_proposed": "提出编排Patch",
                    "patch_decision": "完成Patch校验并作出决定",
                }
                stage_path.parent.mkdir(parents=True, exist_ok=True)
                localized = {
                    "sequence": len(stage_events),
                    "事件类型": event_type,
                    "事件": descriptions.get(event_type, "编排阶段事件"),
                    "阶段": name,
                    "timestamp": record["timestamp"],
                    "详细信息": payload,
                }
                with stage_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(
                        localized, ensure_ascii=False, default=str,
                    ) + "\n")
                    stream.flush()

        runtime = ScriptRuntime(
            executor,
            limits=RunLimits(
                max_parallel_jobs=budget.max_parallel_jobs,
                max_total_jobs=budget.max_jobs,
                wall_time_seconds=budget.wall_time_seconds,
            ),
            live_trace_path=live_path,
        )
        script = self._make_script(stage, budget)
        result = await runtime.run(
            script, task=task, run_id=run_id, metadata=metadata,
        )
        usage = usage_from_run(result)
        violations = budget_violations(budget, usage)
        trace = build_trace(
            result,
            strategy=self.kind.value,
            usage=usage,
            stage_events=stage_events,
        )
        proposed = sum(event.event_type == "patch_proposed" for event in trace.events)
        applied = sum(
            event.event_type == "patch_decision"
            and event.payload.get("accepted") is True
            for event in trace.events
        )
        return StrategyRunResult(
            run_id=run_id,
            strategy=self.kind.value,
            value=result.value,
            runtime_result=result,
            trace=trace,
            budget_violations=violations,
            patches_proposed=proposed,
            patches_applied=applied,
        )

    def _make_script(self, stage, budget: BudgetSpec):
        raise NotImplementedError


class SingleAgentStrategy(_BaseStrategy):
    """Orchestration-side offline baseline; benchmark keeps react_base."""

    kind = StrategyKind.SINGLE_AGENT

    def _make_script(self, stage, budget: BudgetSpec):
        async def script(ctx):
            stage("single_agent_started")
            value = await ctx.call(
                "general",
                f"Solve the task and return the final answer:\n{ctx.task}",
                key="single:answer",
                max_turns=12,
            )
            stage("single_agent_completed")
            return value

        return script


class FixedMultiAgentStrategy(_BaseStrategy):
    kind = StrategyKind.FIXED_MULTI_AGENT

    def _make_script(self, stage, budget: BudgetSpec):
        async def script(ctx):
            branches = _enforce_capability_routes(_default_branches())
            stage(
                "subtask_list_prepared",
                subproblem_count=len(branches),
                web_branches=sum(branch.requires_web for branch in branches),
            )

            async def solve_one(item):
                index, branch = item
                solution = await ctx.call(
                    branch.role,
                    _branch_instruction(ctx.task, branch),
                    key=_branch_job_key("solve", index, branch), max_turns=3,
                )
                return {
                    "branch_index": index,
                    "branch": asdict(branch),
                    "solution": solution,
                }

            solved = await ctx.map(solve_one, enumerate(branches))
            candidates = _collect_candidates(solved)
            if not candidates:
                fallback_handle = ctx.spawn(
                    "solver", _fallback_candidate_instruction(ctx.task, solved),
                    key="fallback:no-candidate", max_turns=2,
                )
                fallback_outcome = (await ctx.gather(
                    fallback_handle, failure_policy="collect",
                ))[0]
                if fallback_outcome.succeeded:
                    solved.append({
                        "branch_index": len(solved),
                        "branch": {"role": "solver", "approach": "fallback recovery"},
                        "solution": fallback_outcome.result.value,
                    })
                    candidates = _collect_candidates(solved)
            if not candidates:
                stage("no_candidate", internal_status="execution_failed")
                return {"status": "no_candidate", "execution_failed": True}
            audits, degraded = await _audit_candidates(ctx, candidates)
            if degraded:
                stage("verification_degraded", degraded_candidates=degraded)
            stage("verification_completed", candidate_count=len(candidates))
            stage("synthesizer_starting")
            synthesized = await ctx.call(
                "synthesizer",
                _synthesis_instruction(ctx.task, _synthesis_payload(candidates, audits)),
                key="synthesize", max_turns=3,
            )
            return synthesized

        return script


class DynamicMultiAgentStrategy(_BaseStrategy):
    kind = StrategyKind.DYNAMIC_MULTI_AGENT

    def __init__(self, *, max_subproblems: int = 3) -> None:
        self.max_subproblems = max_subproblems
        self.stage_budget = StageBudgetPolicy()

    def _make_script(self, stage, budget: BudgetSpec):
        validator = PatchValidator(allowed_roles=_ROLES, budget=budget)

        async def script(ctx):
            branches = _default_branches()
            normalized = _enforce_capability_routes(
                branches, limit=self.max_subproblems,
            )
            plan = _initial_plan(branches, budget.max_parallel_jobs)
            stage(
                "subtask_list_prepared",
                subproblem_count=len(branches),
                web_branches=sum(branch.requires_web for branch in branches),
                stage_budget={
                    "discovery_limit": self.stage_budget.discovery_limit,
                    "verification_limit": self.stage_budget.verification_limit,
                    "revision_limit": self.stage_budget.revision_limit,
                    "synthesis_reserve": self.stage_budget.synthesis_reserve,
                },
            )
            if len(normalized) != len(branches):
                keep = set(range(len(normalized)))
                operations = tuple(
                    PatchOperation(op="delete", node_id=f"solve:{index}")
                    for index in range(len(branches)) if index not in keep
                ) + (PatchOperation(
                    op="update",
                    node_id="synthesize",
                    changes={
                        "dependencies": tuple(
                            f"solve:{index}" for index in range(len(normalized))
                        ),
                    },
                ),)
                patch = OrchestrationPatchV1(
                    base_version=plan.version,
                    trigger_stage="subtask_list_prepared",
                    evidence=("duplicate_or_excess_subproblems",),
                    operations=operations,
                    reason="deduplicate and cap decomposition fan-out",
                )
                plan = _record_patch(stage, validator, plan, patch)
            branches = normalized

            handles = [
                ctx.spawn(
                    branch.role,
                    _branch_instruction(ctx.task, branch),
                    key=_branch_job_key("solve", index, branch), max_turns=3,
                    metadata={
                        "plan_node_id": f"solve:{index}",
                        "requires_web": branch.requires_web,
                        "approach": branch.approach,
                    },
                )
                for index, branch in enumerate(branches)
            ]
            outcomes = await ctx.gather(*handles, failure_policy="collect")
            discovery_usage = ctx.usage()
            stage(
                "first_solver_round_completed",
                usage=discovery_usage,
                allocation=self.stage_budget.discovery_limit,
                allocation_exceeded=not _within_stage_budget(
                    discovery_usage, budget,
                    fraction=self.stage_budget.discovery_limit,
                ),
            )
            records: list[dict[str, Any]] = []
            for index, (branch, outcome) in enumerate(zip(branches, outcomes)):
                if outcome.succeeded:
                    solution = outcome.result.value
                else:
                    solution = _incomplete_solution(
                        f"Branch execution failed: {outcome.failure.error_type}"
                    )
                records.append({
                    "branch_index": index,
                    "branch": asdict(branch),
                    "solution": solution,
                    "verdict": None,
                    "revision": None,
                })

            candidates = _collect_candidates(records)
            if not candidates:
                fallback = ctx.spawn(
                    "solver", _fallback_candidate_instruction(ctx.task, records),
                    key="fallback:no-candidate", max_turns=2,
                )
                fallback_outcome = (await ctx.gather(
                    fallback, failure_policy="collect",
                ))[0]
                if fallback_outcome.succeeded:
                    records.append({
                        "branch_index": len(records),
                        "branch": {"role": "solver", "approach": "fallback recovery"},
                        "solution": fallback_outcome.result.value,
                    })
                    candidates = _collect_candidates(records)
            if not candidates:
                stage("no_candidate", internal_status="execution_failed")
                return {"status": "no_candidate", "execution_failed": True}

            audits: list[dict[str, Any]] = []
            if _within_stage_budget(
                ctx.usage(), budget,
                fraction=self.stage_budget.verification_limit,
            ):
                audits, degraded = await _audit_candidates(ctx, candidates)
                if degraded:
                    stage("verification_degraded", degraded_candidates=degraded)
            else:
                stage(
                    "verification_skipped_budget",
                    usage=ctx.usage(),
                    reason="为Synthesizer保留题目级Token预算",
                )
                audits = [
                    _fallback_verdict(candidate, "Verifier skipped to preserve synthesis budget")
                    for candidate in candidates
                ]
            stage(
                "verification_completed", usage=ctx.usage(),
                candidate_count=len(candidates),
            )

            stage("synthesizer_starting", plan_version=plan.version)
            synthesized = await ctx.call(
                "synthesizer",
                _synthesis_instruction(ctx.task, _synthesis_payload(candidates, audits)),
                key="synthesize", max_turns=3,
            )
            return synthesized

        return script


def _initial_plan(
    branches: list[SolutionBranch], parallel: int,
) -> OrchestrationPlan:
    nodes = [
        PlanNode(f"solve:{index}", branch.role, branch.objective)
        for index, branch in enumerate(branches)
    ]
    nodes.append(PlanNode(
        "synthesize", "synthesizer", "Synthesize results",
        dependencies=tuple(node.node_id for node in nodes),
    ))
    return OrchestrationPlan(tuple(nodes), max_parallel_jobs=parallel)


def _record_patch(stage, validator, plan, patch) -> OrchestrationPlan:
    stage(
        patch.trigger_stage, "patch_proposed",
        patch=asdict(patch), plan_version=plan.version,
    )
    decision = validator.apply(plan, patch)
    stage(
        patch.trigger_stage, "patch_decision",
        accepted=decision.accepted,
        reason=decision.reason,
        before_version=plan.version,
        after_version=decision.plan.version,
    )
    return decision.plan


def _decomposition_instruction(task: str) -> str:
    return (
        "Create independent solution branches for the COMPLETE original task. "
        "Each branch must be capable of producing a candidate final answer on "
        "its own, preserve every constraint, and use a distinct evidence or "
        "reasoning route. Do not turn sequential filters into separate parallel "
        "tasks. Mark requires_web=true whenever current external facts or source "
        "lookup are needed. The only available research tools are web_search, "
        "web_fetch, and run_python_code; do not propose private APIs, Kaggle "
        "datasets, local databases, or credentials that are not supplied. For "
        "web tasks, use genuinely distinct routes: exact rare-clue phrase "
        "search, entity/referee enumeration followed by match-sheet checking, "
        "and source-specific archival corroboration.\n\nOriginal task:\n" + task
    )


def _branch_instruction(task: str, branch: SolutionBranch) -> str:
    web_requirement = (
        "Use web_search/web_fetch to collect evidence before answering."
        if branch.requires_web
        else "Solve from the supplied information; web research is not required."
    )
    return (
        f"Original complete task:\n{task}\n\n"
        f"Independent branch objective:\n{branch.objective}\n\n"
        f"Distinct approach:\n{branch.approach}\n\n"
        f"Research requirement: {web_requirement}\n\n"
        "Produce a candidate answer to the complete original task, not merely "
        "an intermediate filtering step. Preserve all original constraints. "
        "Maintain a candidate table with one column per constraint. Search the "
        "joint constraints, not each condition in isolation. Stop after two "
        "consecutive searches add no evidence. Preserve every concrete candidate "
        "already found and name missing constraints separately; incomplete status "
        "must not erase a candidate answer."
    )


async def _audit_candidates(
    ctx: Any, candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Audit each candidate independently and preserve degraded audits."""
    handles = [
        ctx.spawn(
            "verifier",
            _verification_instruction(ctx.task, candidate),
            key=f"verify:{candidate['candidate_id']}",
            max_turns=3,
            metadata={"candidate_id": candidate["candidate_id"]},
        )
        for candidate in candidates
    ]
    outcomes = await ctx.gather(*handles, failure_policy="collect")
    audits: list[dict[str, Any]] = []
    degraded: list[str] = []
    for candidate, outcome in zip(candidates, outcomes):
        if outcome.succeeded:
            verdict = outcome.result.value
        else:
            degraded.append(candidate["candidate_id"])
            verdict = _fallback_verdict(candidate, outcome.failure.error_type)
        audits.append({
            "candidate_id": candidate["candidate_id"],
            "candidate": candidate,
            "verdict": verdict,
        })
    return audits, degraded


def _verification_instruction(
    task: str, candidate: dict[str, Any],
) -> str:
    return (
        f"Original complete task:\n{task}\n\n"
        "Candidate to audit (this is the only candidate you may consider):\n"
        f"{json.dumps(candidate, ensure_ascii=False, default=str)}\n\n"
        "Audit this candidate independently against every constraint in the original "
        "task. Do not discover, name, compare, rank, or recommend another candidate. "
        "Accept only if it is correct, complete, and satisfies every requirement. "
        "Otherwise reject it and give specific actionable criticism. Return only "
        "the prescribed acceptance decision and critique."
    )


def _synthesis_instruction(task: str, results: Any) -> str:
    return (
        f"Original task:\n{task}\n\nCandidates and independent audits:\n"
        f"{json.dumps(results, ensure_ascii=False, default=str)}\n\n"
        "Compare candidates yourself using the independent audit attached to each "
        "candidate. Prefer accepted candidates and use each rejected candidate's "
        "critique to avoid unsupported claims. Do not invent a replacement candidate "
        "or present a rejected candidate as verified. "
        "Return only the concise final answer requested by the original task."
    )


def _synthesis_payload(
    candidates: list[dict[str, Any]], audits: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "all_candidates": candidates,
        "independent_verifier_audits": audits,
    }


def _collect_candidates(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recover concrete candidates without using status as an admission gate."""
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record_index, record in enumerate(records):
        solution = record.get("solution")
        branch = record.get("branch", {})
        answers: list[tuple[str, str]] = []
        evidence: list[Any] = []
        unverified: list[Any] = []
        negative: list[Any] = []
        status = _solution_status(solution)
        if isinstance(solution, dict):
            answer = str(solution.get("answer", "")).strip()
            table = solution.get("candidate_table", [])
            if answer and not _is_placeholder_answer(answer):
                answers.append(("answer", answer))
            if isinstance(table, list):
                answers.extend(
                    (f"table-{index}", str(item).strip())
                    for index, item in enumerate(table)
                    if str(item).strip() and not _is_placeholder_answer(str(item))
                )
            evidence = solution.get("evidence", []) or []
            unverified = solution.get(
                "missing_constraints", solution.get("unverified_criteria", []),
            ) or []
            negative = solution.get(
                "incorrect_claims", solution.get("contradicted_criteria", []),
            ) or []
        else:
            answer = str(solution or "").strip()
            if answer and not _is_placeholder_answer(answer):
                answers.append(("answer", answer))
        for suffix, answer in answers:
            identity = answer.casefold()
            if identity in seen:
                continue
            seen.add(identity)
            candidates.append({
                "candidate_id": f"branch-{record.get('branch_index', record_index)}-{suffix}",
                "answer": answer,
                "source_branch": branch,
                "source_model": (
                    branch.get("role", "unknown")
                    if isinstance(branch, dict) else "unknown"
                ),
                "status": status,
                "evidence": evidence if isinstance(evidence, list) else [evidence],
                "unverified_criteria": unverified if isinstance(unverified, list) else [unverified],
                "negative_evidence": negative if isinstance(negative, list) else [negative],
            })
    return candidates


def _is_placeholder_answer(answer: str) -> bool:
    normalized = answer.strip().casefold().rstrip(".。")
    return normalized in {
        "", "insufficient information", "information insufficient",
        "信息不足", "无法判断", "unknown", "no candidate", "not found",
    }


def _fallback_candidate_instruction(task: str, records: Any) -> str:
    return (
        f"Original task:\n{task}\n\nRaw researcher and solver outputs:\n"
        f"{json.dumps(records, ensure_ascii=False, default=str)}\n\n"
        "Recover every concrete answer mentioned in answer, candidate_table, "
        "evidence, or prose. Produce the strongest concrete candidate even when "
        "some conditions remain unverified. Do not answer only with an "
        "insufficient-information placeholder."
    )


def _fallback_verdict(
    candidate: dict[str, Any], reason: str,
) -> dict[str, Any]:
    """Return a conservative audit when a verifier cannot run."""
    missing = [str(item) for item in candidate.get("unverified_criteria", [])]
    critique = reason
    if missing:
        critique += " Missing criteria: " + "; ".join(missing)
    return {
        "accepted": False,
        "critique": critique,
    }


def _revision_instruction(
    task: str,
    branch: SolutionBranch,
    previous: Any,
    verdict: dict[str, Any],
) -> str:
    return (
        _branch_instruction(task, branch)
        + f"\n\nRejected candidate:\n{previous}\n\n"
        + f"Verifier critique:\n{verdict.get('critique', '')}\n\n"
        + "Perform one targeted correction. Do not repeat a disproven candidate. "
        "Search only for evidence needed to resolve the verifier's critique. "
        "If it cannot be resolved within this attempt, return status=incomplete."
    )


def _incomplete_solution(reason: str) -> dict[str, Any]:
    return {
        "status": "incomplete",
        "answer": "",
        "candidate_table": [],
        "evidence": [],
        "missing_constraints": [reason],
    }


def _solution_status(solution: Any) -> str:
    if isinstance(solution, dict):
        return str(solution.get("status", "incomplete"))
    text = str(solution or "").strip()
    if not text or "Insufficient information" in text or "DSML" in text:
        return "incomplete"
    return "candidate"


def _normalize_branches(value: Any, task: str) -> list[SolutionBranch]:
    if not isinstance(value, list):
        return [_fallback_branch(task)]
    branches: list[SolutionBranch] = []
    for item in value:
        if isinstance(item, dict):
            objective = str(item.get("objective", "")).strip()
            approach = str(item.get("approach", "")).strip()
            requires_web = item.get("requires_web", False) is True
            if objective:
                branches.append(SolutionBranch(
                    objective=objective,
                    requires_web=requires_web,
                    approach=approach or objective,
                ))
            continue
        # Compatibility with older externally supplied branch lists.
        text = str(item).strip()
        if text:
            branches.append(SolutionBranch(text, False, text))
    return branches or [_fallback_branch(task)]


def _fallback_branch(task: str) -> SolutionBranch:
    return SolutionBranch(
        objective="Answer the complete original task",
        requires_web=True,
        approach=(
            "Search for the full conjunction of constraints, identify likely "
            "candidates, then verify every condition with reliable sources."
        ),
    )


def _default_branches() -> list[SolutionBranch]:
    """Return bounded complementary subtasks without spending an agent job."""
    return [
        SolutionBranch(
            objective="Discover concrete candidates for the complete task",
            requires_web=True,
            approach=(
                "Search the rarest conjunctions of constraints, preserve source "
                "URLs, and return a candidate table without guessing."
            ),
        ),
        SolutionBranch(
            objective="Independently test every constraint against candidates",
            requires_web=True,
            approach=(
                "Start from structured records and primary sources, eliminate "
                "any candidate that fails one required condition."
            ),
        ),
        SolutionBranch(
            objective="Resolve evidence gaps and produce a supported candidate",
            requires_web=True,
            approach=(
                "Focus on missing links and conflicting claims, then report the "
                "strongest answer with explicit unresolved constraints."
            ),
        ),
    ]


def _deduplicate_branches(
    branches: list[SolutionBranch],
) -> list[SolutionBranch]:
    seen: set[tuple[str, str]] = set()
    result: list[SolutionBranch] = []
    for branch in branches:
        key = (branch.objective.casefold(), branch.approach.casefold())
        if key not in seen:
            seen.add(key)
            result.append(branch)
    return result


_WEB_ROUTE_TEMPLATES = (
    (
        "Search exact combinations of the rarest clues and quoted event "
        "phrases. When a near-exact clue or benchmark/paper excerpt appears, "
        "pivot immediately to identifying the named entity or match."
    ),
    (
        "Enumerate plausible referees or entities from public web sources, "
        "then inspect their match lists and match sheets against every card "
        "and substitution constraint."
    ),
    (
        "Search source-specific public archives and historical match reports, "
        "then use web_fetch to corroborate the strongest candidate across "
        "independent pages. Do not assume access to private APIs or datasets."
    ),
)


def _enforce_capability_routes(
    branches: list[SolutionBranch], *, limit: int = 3,
) -> list[SolutionBranch]:
    """Make generated routes distinct and executable with registered tools."""
    unique = _deduplicate_branches(branches)[:limit]
    web_index = 0
    routed: list[SolutionBranch] = []
    for branch in unique:
        if not branch.requires_web:
            routed.append(branch)
            continue
        routed.append(SolutionBranch(
            objective=branch.objective,
            requires_web=True,
            approach=_WEB_ROUTE_TEMPLATES[
                min(web_index, len(_WEB_ROUTE_TEMPLATES) - 1)
            ],
        ))
        web_index += 1
    return routed or [_fallback_branch("")]


def _within_stage_budget(
    usage: dict[str, int], budget: BudgetSpec, *, fraction: float,
) -> bool:
    checks = (
        (usage.get("tokens", 0), budget.max_tokens),
        (usage.get("llm_calls", 0), budget.max_llm_calls),
        (usage.get("tool_calls", 0), budget.max_tool_calls),
    )
    return all(
        limit is None or actual < int(limit * fraction)
        for actual, limit in checks
    )


def usage_from_run(run: ScriptRunResult) -> BudgetUsage:
    tokens = llm_calls = tool_calls = 0
    for event in run.events:
        if event.event_type.value == "job_succeeded":
            usage = event.payload.get("usage", {})
            tokens += int(usage.get("total_tokens", 0) or usage.get("tokens", 0) or (
                usage.get("prompt_tokens", usage.get("input_tokens", 0))
                + usage.get("completion_tokens", usage.get("output_tokens", 0))
            ))
        if (
            event.event_type.value == "worker_event"
            and event.payload.get("name") == "agent_loop_succeeded"
        ):
            data = event.payload.get("data", {})
            llm_calls += int(data.get("turns", 0))
            tool_calls += int(data.get("tool_calls", 0))
        if event.event_type.value == "tool_call_started":
            tool_calls += 1
    duration = sum(job.elapsed_seconds for job in run.jobs)
    return BudgetUsage(
        tokens=tokens,
        llm_calls=llm_calls,
        tool_calls=tool_calls,
        jobs=len(run.jobs),
        duration_seconds=duration,
    )


def budget_violations(budget: BudgetSpec, usage: BudgetUsage) -> tuple[str, ...]:
    checks = (
        ("tokens", budget.max_tokens, usage.tokens),
        ("llm_calls", budget.max_llm_calls, usage.llm_calls),
        ("tool_calls", budget.max_tool_calls, usage.tool_calls),
        ("jobs", budget.max_jobs, usage.jobs),
    )
    return tuple(
        f"{name}:{actual}>{limit}"
        for name, limit, actual in checks
        if limit is not None and actual > limit
    )


def build_trace(
    run: ScriptRunResult,
    *,
    strategy: str,
    usage: BudgetUsage,
    stage_events: list[dict[str, Any]],
) -> RunTraceV1:
    jobs = {job.job_id: job for job in run.jobs}
    projected: list[TraceEventV1] = []
    for source in run.events:
        job = jobs.get(source.job_id)
        event_type = source.event_type.value
        if event_type == "script_event":
            event_type = str(source.payload.get("script_event", event_type))
        projected.append(TraceEventV1(
            sequence=0,
            event_type=event_type,
            run_id=run.run_id,
            timestamp=source.timestamp,
            job_id=source.job_id,
            parent_job_id=getattr(job, "parent_job_id", None),
            role=getattr(job, "role", None),
            status=(
                getattr(getattr(job, "status", None), "value", None)
                if job else None
            ),
            payload=source.payload,
        ))
    for item in stage_events:
        projected.append(TraceEventV1(
            sequence=0,
            event_type=item["event_type"],
            run_id=run.run_id,
            timestamp=item["timestamp"],
            stage=item["stage"],
            payload=item["payload"],
        ))
    projected.sort(key=lambda event: event.timestamp)
    events = tuple(
        TraceEventV1(**{**event.__dict__, "sequence": index})
        for index, event in enumerate(projected, 1)
    )
    return RunTraceV1(run.run_id, strategy, events, usage)


STRATEGIES: dict[str, type[_BaseStrategy]] = {
    StrategyKind.SINGLE_AGENT.value: SingleAgentStrategy,
    StrategyKind.FIXED_MULTI_AGENT.value: FixedMultiAgentStrategy,
    StrategyKind.DYNAMIC_MULTI_AGENT.value: DynamicMultiAgentStrategy,
}


def create_strategy(name: str) -> OrchestrationStrategy:
    try:
        return STRATEGIES[name]()
    except KeyError as exc:
        raise ValueError(
            f"unknown strategy {name!r}; choose from {sorted(STRATEGIES)}"
        ) from exc
