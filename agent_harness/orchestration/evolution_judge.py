"""Terminal contracts for the V3 self-evolving orchestration gate."""

from __future__ import annotations

import json
import re
from typing import Any, Iterable

from agent_harness.orchestration.self_evolving import normalize_failure_report


PROPOSAL_KEYS = frozenset({
    "status", "selected_candidate", "selected_verifier", "facts", "issues",
    "synthesis_context",
})
JUDGE_DECISIONS = frozenset({"approve_synthesis", "evolve"})
_SECRET_KEY = re.compile(
    r"(?:api[_-]?key|authorization|password|secret|access[_-]?token)",
    re.IGNORECASE,
)


def normalize_round_proposal(value: Any) -> dict[str, Any]:
    """Normalize a generated V3 script result and fail closed on ambiguity."""
    if not isinstance(value, dict):
        return _incomplete_proposal(
            "generated script did not return a candidate proposal object",
        )
    try:
        json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return _incomplete_proposal(
            "generated script returned a non-JSON-compatible candidate proposal",
        )
    candidate = value.get("selected_candidate", "")
    verifier = value.get("selected_verifier", {})
    if not isinstance(verifier, dict):
        verifier = {}
    facts = value.get("facts", [])
    issues = value.get("issues", [])
    context = value.get("synthesis_context", {})
    normalized = {
        "status": str(value.get("status", "incomplete")),
        "selected_candidate": candidate,
        # Candidate eligibility and the selected verifier conclusion are terminal
        # gate inputs, so retain their complete JSON value rather than applying a
        # trace-summary string limit.
        "selected_verifier": _json_value(verifier, string_limit=None),
        "facts": _json_value(
            facts if isinstance(facts, list) else [], string_limit=None,
        ),
        "issues": _json_value(
            issues if isinstance(issues, list) else [str(issues)],
            string_limit=None,
        ),
        "synthesis_context": _json_value(context, string_limit=24_000),
    }
    if normalized["status"] not in {"ready", "incomplete"}:
        normalized["issues"].append("proposal status must be ready or incomplete")
        normalized["status"] = "incomplete"
    if not proposal_is_verifier_eligible(normalized):
        normalized["status"] = "incomplete"
    return normalized


def proposal_is_verifier_eligible(proposal: dict[str, Any]) -> bool:
    verifier = proposal.get("selected_verifier", {})
    return bool(
        proposal.get("status") == "ready"
        and _has_candidate(proposal.get("selected_candidate"))
        and isinstance(verifier, dict)
        and verifier.get("accepted") is True
    )


def normalize_judge_decision(
    value: Any,
    *,
    proposal: dict[str, Any],
    failure_stage: str,
    summary: str,
    previous_report: dict[str, Any] | None,
    observed_urls: set[str],
) -> dict[str, Any]:
    """Validate veto-only authority and source-bind judge facts."""
    raw = value if isinstance(value, dict) else {}
    requested = str(raw.get("decision", "evolve"))
    from agent_harness.orchestration.agent_executor import parse_role_output
    from agent_harness.orchestration.evolution_roles import EVOLUTION_JUDGE_ROLE
    from agent_harness.orchestration.exceptions import InvalidRoleOutput

    try:
        parse_role_output(json.dumps(raw, allow_nan=False), EVOLUTION_JUDGE_ROLE.result_schema)
    except (InvalidRoleOutput, TypeError, ValueError):
        raw = {}
    eligible = proposal_is_verifier_eligible(proposal)
    decision = (
        "approve_synthesis"
        if raw.get("decision") == "approve_synthesis" and eligible
        else "evolve"
    )
    rationale = str(raw.get("rationale", "")).strip()[:8000]
    if requested not in JUDGE_DECISIONS:
        rationale = (
            "Evolution Judge returned an invalid decision. " + rationale
        ).strip()
    elif requested == "approve_synthesis" and not eligible:
        rationale = (
            "Evolution Judge approval was downgraded because the selected "
            "candidate was not ordinary-Verifier-eligible. " + rationale
        ).strip()
    confirmed_facts = _source_bound_facts(
        raw.get("confirmed_facts", []), observed_urls,
    )
    report = None
    if decision == "evolve":
        report = normalize_failure_report(
            raw.get("failure_report"),
            failure_stage=failure_stage,
            summary=rationale or summary,
            previous_report=previous_report,
        )
        # V3 deliberately runs all five rounds unless synthesis succeeds.
        report["continue_evolution"] = True
    return {
        "decision": decision,
        "requested_decision": requested,
        "eligible_before_judge": eligible,
        "rationale": rationale or summary,
        "confirmed_facts": confirmed_facts,
        "failure_report": report,
    }


def build_structured_round_trace(
    *,
    task: str,
    round_number: int,
    evolution_context: dict[str, Any],
    natural_language_plan: str,
    generation_status: dict[str, Any],
    proposal: dict[str, Any],
    events: Iterable[Any],
    jobs: Iterable[Any],
    execution_error: str,
    usage: dict[str, Any],
) -> dict[str, Any]:
    """Return a complete, bounded, JSON-only representation of one round."""
    job_events: dict[str, dict[str, Any]] = {}
    event_values = list(events)
    for event in event_values:
        job_id = str(getattr(event, "job_id", "") or "")
        if not job_id:
            continue
        event_type = _enum_value(getattr(event, "event_type", ""))
        payload = getattr(event, "payload", {})
        payload = payload if isinstance(payload, dict) else {}
        document = job_events.setdefault(job_id, {})
        if event_type == "job_created":
            document["input"] = _json_value(
                payload.get("instruction", ""), string_limit=24_000,
            )
        elif event_type == "job_succeeded":
            document["result"] = _json_value(
                payload.get("result"), string_limit=24_000,
            )
            document["usage"] = _json_value(
                payload.get("usage", {}), string_limit=2000,
            )
            metadata = payload.get("metadata", {})
            if isinstance(metadata, dict) and metadata.get("trace_summary"):
                document["trace_summary"] = _json_value(
                    metadata["trace_summary"], string_limit=16_000,
                )
        elif event_type in {
            "job_failed", "job_timed_out", "job_cancelled",
        }:
            document["error"] = {
                "type": str(payload.get("error_type", event_type)),
                "message": str(payload.get("message", "")),
            }
    job_documents = []
    for job in jobs:
        job_id = str(getattr(job, "job_id", ""))
        linked = job_events.get(job_id, {})
        job_documents.append({
            "job_id": job_id,
            "key": str(getattr(job, "key", "")),
            "role": str(getattr(job, "role", "")),
            "status": _enum_value(getattr(job, "status", "")),
            "input": linked.get("input", ""),
            "result": linked.get("result"),
            "error": linked.get("error"),
            "usage": linked.get("usage", {}),
            "trace_summary": linked.get("trace_summary", {}),
            "elapsed_seconds": getattr(job, "elapsed_seconds", 0),
            "max_turns": getattr(job, "max_turns", None),
            "parent_job_id": getattr(job, "parent_job_id", None),
            "depth": getattr(job, "depth", 0),
            "branch_path": list(getattr(job, "branch_path", ()) or ()),
            "metadata": _json_value(
                getattr(job, "metadata", {}), string_limit=4000,
            ),
        })
    event_documents = []
    for event in event_values:
        event_documents.append({
            "sequence": getattr(event, "sequence", 0),
            "event_type": _enum_value(getattr(event, "event_type", "")),
            "timestamp": getattr(event, "timestamp", 0),
            "job_id": getattr(event, "job_id", None),
            "key": getattr(event, "key", None),
            "payload": _json_value(
                getattr(event, "payload", {}), string_limit=8000,
            ),
        })
    return {
        "original_task": task,
        "round": round_number,
        "inherited_evolution_context": _json_value(
            evolution_context, string_limit=16_000,
        ),
        "natural_language_plan": natural_language_plan[:32_000],
        "generation": _json_value(generation_status, string_limit=8000),
        "candidate_proposal": proposal,
        "execution_error": execution_error,
        "jobs": job_documents,
        "events": event_documents,
        "usage": _json_value(usage, string_limit=2000),
    }


def build_prior_round_failures(round_records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Snapshot every prior failed round without filtering faults or truncating reports.

    The Runner calls this before appending the current round. History is separate
    from the Planner's latest-report context and contains no old plan or source.
    """
    history = []
    for record in round_records:
        proposal = record.get("proposal") or {}
        history.append(_json_value({
            "round": record.get("round"),
            "status": record.get("status"),
            "execution_error": record.get("execution_error"),
            "candidate_proposal": {
                key: proposal.get(key)
                for key in ("status", "selected_candidate", "selected_verifier")
            },
            "evolution_judge": record.get("evolution_judge"),
            "failure_report": record.get("failure_report"),
        }, string_limit=None))
    return history


def render_judge_instruction(trace: dict[str, Any]) -> str:
    instruction = (
        "Audit this completed orchestration round. You are a veto-only terminal "
        "gate: approve_synthesis cannot make an ineligible candidate eligible, "
        "cannot replace the selected candidate, and cannot override an ordinary "
        "Verifier rejection. Use evolve for every unsupported identity/answer "
        "binding, reliable contradiction, broken entity chain, or material process "
        "failure. When decision=evolve, failure_report must be complete and useful "
        "to the next Planner."
    )
    return (
        instruction + "\n\n# Complete structured round trace\n"
        + json.dumps(trace, ensure_ascii=False, indent=2, default=str)
    )


def render_synthesis_instruction(
    task: str,
    proposal: dict[str, Any],
    judge: dict[str, Any],
    active_facts: list[dict[str, Any]],
) -> str:
    payload = {
        "selected_candidate": proposal.get("selected_candidate", ""),
        "selected_verifier": proposal.get("selected_verifier", {}),
        "synthesis_context": proposal.get("synthesis_context", {}),
        "active_facts": active_facts,
        "evolution_judge": {
            "decision": judge.get("decision"),
            "rationale": judge.get("rationale", ""),
        },
    }
    return (
        "Original task:\n" + task
        + "\n\nReturn the concise final answer in the format requested by the "
        "original task. Use only this ordinary-Verifier-eligible candidate; do "
        "not introduce or substitute another answer.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    )


def urls_in_value(value: Any) -> set[str]:
    return set(re.findall(
        r"https?://[^\s\]\[()<>\"']+",
        json.dumps(value, ensure_ascii=False, default=str),
    ))


def urls_in_tool_trace(events: Iterable[Any]) -> set[str]:
    """Extract URLs observed by a worker without trusting its answer payload."""
    values = []
    for event in events:
        event_type = _enum_value(getattr(event, "event_type", ""))
        payload = getattr(event, "payload", {})
        if not isinstance(payload, dict):
            continue
        if event_type == "job_succeeded":
            metadata = payload.get("metadata", {})
            if isinstance(metadata, dict):
                values.append(metadata.get("trace_summary", {}))
        elif event_type in {
            "worker_event", "tool_call_started", "tool_call_succeeded",
            "tool_call_failed",
        }:
            values.append(payload)
    return urls_in_value(values)


def _incomplete_proposal(issue: str) -> dict[str, Any]:
    return {
        "status": "incomplete",
        "selected_candidate": "",
        "selected_verifier": {},
        "facts": [],
        "issues": [issue],
        "synthesis_context": {},
    }


def _has_candidate(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, tuple)):
        return bool(value)
    return True


def _source_bound_facts(value: Any, observed_urls: set[str]) -> list[dict[str, Any]]:
    result = []
    if not isinstance(value, list):
        return result
    for item in value[:32]:
        if not isinstance(item, dict):
            continue
        claim = str(item.get("claim", "")).strip()
        urls = item.get("source_urls", [])
        if not isinstance(urls, list):
            continue
        urls = [
            str(url).strip() for url in urls[:8]
            if str(url).strip() in observed_urls
        ]
        if not claim or not urls:
            continue
        candidate_ids = item.get("candidate_ids", [])
        if not isinstance(candidate_ids, list):
            candidate_ids = []
        result.append({
            "claim": claim,
            "source_urls": urls,
            "candidate_ids": [str(item) for item in candidate_ids[:8]],
        })
    return result


def _json_value(value: Any, *, string_limit: int | None) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            name = str(key)
            result[name] = (
                "[REDACTED]" if _SECRET_KEY.search(name)
                else _json_value(item, string_limit=string_limit)
            )
        return result
    if isinstance(value, (list, tuple)):
        return [_json_value(item, string_limit=string_limit) for item in value]
    if isinstance(value, str):
        return value if string_limit is None else value[:string_limit]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    rendered = str(value)
    return rendered if string_limit is None else rendered[:string_limit]


def _enum_value(value: Any) -> str:
    return str(getattr(value, "value", value))


__all__ = [
    "JUDGE_DECISIONS", "PROPOSAL_KEYS", "build_structured_round_trace",
    "build_prior_round_failures",
    "normalize_judge_decision", "normalize_round_proposal",
    "proposal_is_verifier_eligible", "render_judge_instruction",
    "render_synthesis_instruction", "urls_in_tool_trace", "urls_in_value",
]
