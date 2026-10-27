"""Fusion-only two-stage local review and third-round final output contracts."""
from __future__ import annotations

import copy
import json
from typing import Any

from agent_harness.orchestration.evolution_judge import proposal_is_verifier_eligible
from agent_harness.orchestration.evolution_roles import EVOLUTION_JUDGE_ROLE
from agent_harness.orchestration.self_evolving import normalize_failure_report
from agent_harness.models.agent_definition import AgentDefinition

MAX_ROLECHANGE_EVOLUTION_ROUNDS = 3
FACT_REPORT_FIELDS = {"inheritable_facts", "disputed_facts"}
LOCAL_JUDGE_PROMPT = (
    "Audit the current candidate as the final evidence review before synthesis. "
    "STAGE ONE: Read only task, candidate_answer and the frozen facts table. "
    "The candidate is the object of review, not independent evidence. Use only facts as evidence. "
    "Enumerate task constraints; for each give satisfied, unsatisfied or unknown, supporting "
    "fact_ids (empty when unsupported), and a concrete explanation. Preserve disputed status "
    "and consider conflicts and sources. Missing evidence is unknown, not proof of correctness. "
    "Recommend approve_synthesis only for a nonempty candidate adequately supported by facts "
    "without material identity or answer-binding gaps or reliable contradictions. Otherwise "
    "return evolve and rough issues. The runner separately enforces ordinary verifier eligibility; "
    "you cannot infer its verdict or unseen execution failures. Do not request traces, plans, "
    "code, tool transcripts, verifier commentary or historical reports. Do not write a detailed "
    "report in this stage. Use only the supplied table and facts_index/facts_query, which reads the same "
    "frozen snapshot. Never search, fetch websites, execute code or modify facts. Return JSON only."
)
REPORT_JUDGE_PROMPT = (
    "STAGE TWO: Write evidence-based feedback using only task, candidate_answer, the frozen "
    "facts table and the first-stage model assessment. The assessment may be empty if unavailable. "
    "Ground conclusions in facts, not candidate assertions or unseen execution. Explain missing "
    "evidence, conflicts and candidate weaknesses. Do not invent failed routes or verifier opinions: "
    "use empty lists for unsupported list fields and an empty string for verifier_critique. "
    "Describe failure_stage as evidence review or unknown, not an inferred runtime failure. "
    "planner_instructions MUST specify priority_checks, actions, avoid_repeating and "
    "acceptance_conditions; also populate next_round_priorities with actionable evidence checks. "
    "Never request traces, plans, code, tool transcripts, verifier commentary or historical reports. "
    "Use only supplied facts and facts_index/facts_query on the same frozen snapshot. Never search, fetch "
    "websites, execute code or modify facts. Return JSON only."
)
FINAL_ROUND_POLICY = (
    "ROUND THREE FINAL OUTPUT POLICY: no fourth round is allowed. Return decision=best_effort "
    "and a nonempty final_answer, plus the detailed report and hypothetical Planner improvements. "
    "Use only the current candidate and accumulated frozen facts to choose the most likely answer. "
    "If candidate_answer is empty, infer your best answer from the task and facts, explaining "
    "uncertainty. Do not infer historical candidates or request previous traces or reports. "
    "Never return an empty answer. This best_effort path overrides ordinary verifier acceptance."
)


def local_judge_role(round_number: int, *, stage: str = "assessment", reflector: bool = False) -> AgentDefinition:
    if stage not in {"assessment", "report"}:
        raise ValueError(f"unknown judge stage: {stage}")
    strings = {"type": "array", "items": {"type": "string"}}
    if stage == "assessment":
        constraint = {
            "type": "object", "required": ["constraint", "status", "fact_ids", "explanation"],
            "properties": {
                "constraint": {"type": "string"},
                "status": {"type": "string", "enum": ["satisfied", "unsatisfied", "unknown"]},
                "fact_ids": strings, "explanation": {"type": "string"},
            },
        }
        properties = {
            "decision": {"type": "string", "enum": ["approve_synthesis", "evolve"]},
            "rationale": {"type": "string"},
            "constraints": {"type": "array", "items": constraint}, "issues": strings,
        }
        prompt = LOCAL_JUDGE_PROMPT
        if round_number >= 3:
            prompt += (
                "\n\nV4 persistent relaxed review, round 3 only: judge the current candidate "
                "using the same frozen facts. Distinguish minor uncertainty from material gaps; "
                "do not invent support. The runner still enforces verifier eligibility. "
                "If approval is unsupported, return evolve; the separate report stage must "
                "choose a best-effort answer from the current candidate and accumulated facts."
            )
    else:
        report = copy.deepcopy(EVOLUTION_JUDGE_ROLE.result_schema["properties"]["failure_report"]["anyOf"][1])
        for field in FACT_REPORT_FIELDS:
            report["properties"].pop(field)
            report["required"].remove(field)
        instruction_fields = ["priority_checks", "actions", "avoid_repeating", "acceptance_conditions"]
        report["properties"]["planner_instructions"] = {
            "type": "object", "required": instruction_fields, "additionalProperties": False,
            "properties": {field: strings for field in instruction_fields},
        }
        report["required"].append("planner_instructions")
        final = round_number == MAX_ROLECHANGE_EVOLUTION_ROUNDS
        properties = {
            "decision": {"type": "string", "enum": ["best_effort" if final else "evolve"]},
            "rationale": {"type": "string"}, "failure_report": report,
        }
        prompt = REPORT_JUDGE_PROMPT
        if reflector:
            from agent_harness.orchestration.reflection import REFLECTOR_PROMPT, report_schema
            properties["failure_report"] = report_schema(report, round_number)
            prompt = REFLECTOR_PROMPT
        if final:
            properties["final_answer"] = {"type": "string", "minLength": 1}
            prompt += "\n\n" + (FINAL_ROUND_POLICY if not reflector else (
                "ROUND THREE: Return decision=best_effort and nonempty final_answer from the "
                "current candidate and frozen facts only. Diagnose execution using trace, but do not "
                "use trace assertions as answer evidence. Set evolution_mode=STOP and continue_evolution=false."))
    schema = {
        "type": "object", "properties": properties, "required": list(properties),
        "x-rolechange-judge-stage": stage, "additionalProperties": False,
    }
    if reflector and stage == "report":
        schema["x-rolechange-reflector"] = True
    return EVOLUTION_JUDGE_ROLE.model_copy(deep=True, update={
        "system_prompt": prompt, "allowed_tools": ["facts_index", "facts_query"] + (
            ["trace_index", "trace_query"] if reflector and stage == "report" else []), "result_schema": schema,
    })


def validate_local_judge(value: dict, schema: dict) -> None:
    """Semantic validation scoped to the fusion contracts; also runs during repair."""
    from agent_harness.orchestration.exceptions import InvalidRoleOutput

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise InvalidRoleOutput(message)

    require(value["decision"] in schema["properties"]["decision"]["enum"], "invalid judge decision")
    require(bool(value["rationale"].strip()), "judge rationale must be nonempty")
    if schema["x-rolechange-judge-stage"] == "assessment":
        require(bool(value["constraints"]), "judge must enumerate the task constraints")
        for item in value["constraints"]:
            require(item["status"] in {"satisfied", "unsatisfied", "unknown"}, "invalid constraint status")
            require(bool(item["constraint"].strip() and item["explanation"].strip()), "empty constraint assessment")
        if value["decision"] == "evolve":
            require(bool(value["issues"]) and all(i.strip() for i in value["issues"]), "rejection requires rough issues")
    else:
        report = value["failure_report"]
        require(bool(report["summary"].strip()), "failure report requires a summary")
        require(bool(report["next_round_priorities"]) and all(i.strip() for i in report["next_round_priorities"]),
                "failure report requires next-round priorities")
        for field, entries in report["planner_instructions"].items():
            require(bool(entries) and all(item.strip() for item in entries), f"Planner instruction {field} must be nonempty")
        actions = report["planner_instructions"]["actions"]
        require(not all(a.strip().lower().rstrip(".") in {"continue searching", "continue search", "继续搜索"}
                        for a in actions), "Planner actions must be specific")


def local_failure_report(value: Any, **kwargs: Any) -> dict:
    report = normalize_failure_report(value, **kwargs)
    for field in FACT_REPORT_FIELDS:
        report.pop(field, None)
    raw = value if isinstance(value, dict) else {}
    report["planner_instructions"] = raw.get("planner_instructions") or {
        "priority_checks": [f"Resolve the recorded failure: {report['summary']}"],
        "actions": ["Check the latest fact IDs against each task constraint and obtain evidence for the unresolved bindings."],
        "avoid_repeating": report["failed_routes"] or ["Do not reuse an unsupported candidate without addressing the recorded failure."],
        "acceptance_conditions": ["All task constraints have concrete support and the selected candidate passes ordinary verification."],
    }
    if not report["next_round_priorities"]:
        report["next_round_priorities"] = report["planner_instructions"]["priority_checks"]
    return report


def normalize_local_judge(
    value: Any, *, round_number: int, proposal: dict, failure_stage: str,
    summary: str, previous_report: dict | None,
) -> dict:
    from agent_harness.orchestration.agent_executor import parse_role_output
    from agent_harness.orchestration.exceptions import InvalidRoleOutput

    raw = value if isinstance(value, dict) else {}
    requested = raw.get("decision", "evolve")
    try:
        parse_role_output(json.dumps(raw, allow_nan=False), local_judge_role(round_number).result_schema,
                          role="evolution_judge")
    except (InvalidRoleOutput, TypeError, ValueError):
        raw = {}
    eligible = proposal_is_verifier_eligible(proposal)
    approved = raw.get("decision") == "approve_synthesis" and eligible
    rationale = str(raw.get("rationale") or summary)
    if requested == "approve_synthesis" and not eligible:
        rationale = "Current proposal was not ordinary-Verifier-eligible. " + rationale
    return {
        "decision": "approve_synthesis" if approved else "evolve",
        "requested_decision": requested, "eligible_before_judge": eligible,
        "rationale": rationale, "constraints": raw.get("constraints", []),
        "issues": raw.get("issues", []) if approved else (raw.get("issues") or [rationale]),
        "final_answer": "", "failure_report": None,
    }

def answer_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict) and value:
        return json.dumps(value, ensure_ascii=False)
    return ""


def fallback_answer(proposals: list[dict], traces: list[dict] | None = None) -> str:
    # Prefer an authentically eligible proposal even over a newer unverified candidate.
    for proposal in reversed(proposals):
        answer = answer_text(proposal.get("selected_candidate"))
        if answer and proposal_is_verifier_eligible(proposal):
            return answer
    if traces:
        for trace in reversed(traces):
            answer = answer_text(trace.get("candidate_proposal", {}).get("selected_candidate"))
            if answer:
                return answer
            # A script may fail after its solver returned, before submitting a proposal.
            for job in reversed(trace.get("jobs", [])):
                if job.get("role") == "solver" and job.get("status") == "succeeded":
                    answer = answer_text(job.get("result"))
                    if answer:
                        return answer
    for proposal in reversed(proposals):
        answer = answer_text(proposal.get("selected_candidate"))
        if answer:
            return answer
    return ""
