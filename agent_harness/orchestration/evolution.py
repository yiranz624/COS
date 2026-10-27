"""Offline, non-deploying diagnosis and candidate evolution reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from agent_harness.orchestration.trace import RunTraceV1


@dataclass(frozen=True)
class EvolutionReport:
    run_id: str
    failure_categories: tuple[str, ...]
    bottlenecks: tuple[str, ...]
    candidate_changes: tuple[dict[str, Any], ...]
    replay_validation: dict[str, Any]
    auto_deploy: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def diagnose_trace(trace: RunTraceV1) -> EvolutionReport:
    failures: set[str] = set()
    durations: list[tuple[str, float]] = []
    for event in trace.events:
        if event.event_type in {
            "job_failed", "job_timed_out", "job_cancelled", "run_failed",
        }:
            failures.add(str(event.payload.get("error_type", event.event_type)))
        if event.event_type == "job_succeeded" and event.role:
            durations.append((event.role, float(event.payload.get("elapsed_seconds", 0))))
            result = event.payload.get("result")
            if (
                event.role in {"researcher", "solver"}
                and isinstance(result, dict)
                and result.get("status") == "incomplete"
            ):
                failures.add("incomplete_branch")
        if event.stage == "verification_skipped_budget":
            failures.add("verification_skipped_budget")
        if event.role == "synthesizer" and event.event_type == "job_succeeded":
            if str(event.payload.get("result", "")).strip() == "Insufficient information":
                failures.add("no_accepted_candidate")
    candidates: list[dict[str, Any]] = []
    if "TimeoutError" in failures:
        candidates.append({"change": "reduce_branch_budget", "reason": "timeouts"})
    if any("InvalidRoleOutput" in item for item in failures):
        candidates.append({"change": "strengthen_output_contract", "reason": "format"})
    if "verification_skipped_budget" in failures:
        candidates.append({
            "change": "reserve_verification_budget",
            "reason": "discovery_consumed_stage_allocation",
        })
    if "incomplete_branch" in failures:
        candidates.append({
            "change": "improve_candidate_discovery_routes",
            "reason": "research_ended_without_concrete_candidate",
        })
    if "no_accepted_candidate" in failures:
        candidates.append({
            "change": "retain_accepted_only_admission",
            "reason": "no_verified_answer_available",
        })
    if not candidates and failures:
        candidates.append({"change": "targeted_retry", "reason": "worker_failure"})
    return EvolutionReport(
        run_id=trace.run_id,
        failure_categories=tuple(sorted(failures)),
        bottlenecks=tuple(role for role, _ in sorted(
            durations, key=lambda item: item[1], reverse=True,
        )[:3]),
        candidate_changes=tuple(candidates),
        replay_validation={
            "schema_valid": True,
            "budget_valid": not any(
                event.event_type == "budget_violation" for event in trace.events
            ),
            "event_count": len(trace.events),
        },
    )
