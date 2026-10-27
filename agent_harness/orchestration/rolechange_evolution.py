"""Adapters between unchanged rolechange workers and the v4 terminal gate."""
from __future__ import annotations

import ast
import json
import re
import uuid
from dataclasses import asdict, replace
from functools import wraps
from pathlib import Path
from typing import Any

from agent_harness.core.loop_types import BaseObserver

ROLECHANGE_STRATEGY = "rolechange_self_evolving_v4"
ROLECHANGE_V6_STRATEGY = "rolechange_self_evolving_v6"
ROLECHANGE_V5_STRATEGY = "rolechange_self_evolving_v5"
SCRIPT_ROLES = frozenset({"decomposer", "researcher", "solver", "verifier", "synthesizer"})
PROPOSAL_KEYS = frozenset({
    "status", "selected_candidate", "selected_verifier", "facts", "issues", "synthesis_context",
})

PLANNING_RULES = """This run uses outer self-evolution. Plan a fresh task-specific workflow.
Choose among the supplied five business agents; decomposition and intermediate synthesis are optional.
Before submitting the selected answer candidate, have the ordinary verifier independently check
that exact candidate against the original task and accumulated evidence. If later work changes the
candidate, verify it again. A rejection or failed verification cannot be converted to acceptance.
The completed round submits a candidate and its verification to the external Evolution Judge.
Only after that Judge approves does the runner ask the synthesizer for the formal final answer.
Do not call the Evolution Judge from the planned workflow. Intermediate synthesis is permitted.
Workers may use facts_index for the latest shared fact summaries and facts_query for full facts.
Each worker independently decides whether and when to query; neither tool is a mandatory stage
or prerequisite. Do not reject a plan merely because it omits fact queries.
Pass the task, relevant prior role results and related fact IDs to workers, not a round-start
fact snapshot repeated as the current table. Worker queries see committed same-round updates.
Each worker
chooses whether to call facts_submit with one supported fact and its own summary immediately.
The tool waits for authenticated evidence maintenance; successful submissions are visible in
later same-round queries. Do not call evidence_storesubagent directly or evidence_store_request.
Planner reads current full facts before every plan. A round-end full-table maintenance precedes
Evolution Judge. Only explicitly submitted facts enter the table; ordinary outputs and proposals
are not fact sources. The host binds provenance and evidence maintenance decides admission.
The inherited facts and last failure report below guide a NEW plan; they are not a previous plan
to replay. Treat disputed claims as unresolved and evidence, not repetition, as support.
These terminal requirements qualify the general statement that no role is mandatory.
"""

CODE_RULES = """Return a JSON-compatible object with exactly these fields:
status ('ready' or 'incomplete'), selected_candidate (nonempty answer text or a JSON object),
selected_verifier (the actual unmodified verifier result), facts (list of source-backed claims),
issues (list), synthesis_context (JSON object with evidence and intermediate synthesis).
The proposal facts field is diagnostic only and may be empty; it never writes the shared table.
Do not cache a round-start fact snapshot and replay it in worker instructions or synthesis_context
as the current fact table. Pass task, relevant prior role results and related fact IDs instead.
Leave whether and when to query facts_index/facts_query to each worker; do not force fact queries.
Workers explicitly submit facts through facts_submit, supplying claim, summary, source_urls and
nonempty evidence_excerpt, optionally related fact_ids. Empty source_urls is allowed for local
 evidence. Do not assign status or provenance. Never invent evidence or role result fields.
ctx.call returns the ordinary role result directly, without a facts envelope.
Ordinary verifier returns ONLY accepted and critique; do not read or invent classification.
For the final verification, include str(selected_candidate) verbatim in the verifier instruction,
alongside the original task and evidence. Copy that actual returned verdict into selected_verifier.
Set ready only when the concrete selected candidate's latest actual verification accepted is True.
Otherwise return incomplete, preserving available results and issues. Never fabricate a verdict.
Any subsequent change to the selected answer requires fresh verification of that changed answer.
Call the five business roles, including synthesizer for intermediate work; evidence maintenance is host-owned. Never call
general or evolution_judge. The runner owns formal final synthesis after terminal approval.
Use rolechange role defaults unless the plan requires a max_turns override. Do not use v4's
fixed 70/30/20 budgets or runtime Planner APIs. Do not load a previous round's checkpoint.
Generate a new script for this round; do not recreate a prior source from feedback.
"""


def is_fusion(request: Any) -> bool:
    return request.metadata.get("prompt_profile") in {ROLECHANGE_STRATEGY, ROLECHANGE_V5_STRATEGY, ROLECHANGE_V6_STRATEGY}


def evolution_system_prompt(base: str, request: Any) -> str:
    if request.metadata.get("prompt_profile") == ROLECHANGE_V6_STRATEGY:
        base += "\nWhen passing relevant shared facts to downstream roles, preserve their exact stable fact IDs in the instruction. IDs support provenance tracking; fact queries remain optional."
    if not is_fusion(request):
        return base
    return base + (
        "\n\nFor this outer-evolution strategy, the general optional-role guidance has "
        "one exception: a submitted final candidate requires actual ordinary verifier "
        "acceptance before the external Evolution Judge may approve it. The script "
        "submits a candidate proposal, not the formal final answer. Intermediate "
        "synthesis is optional; the runner owns formal final synthesis after approval. "
        "Follow the strategy's supplied return contract and cross-round input rules. "
        "All other role selection, planning and review responsibilities remain as stated."
    )


def adapt_generation_request(request: Any) -> Any:
    if not is_fusion(request):
        return request
    constraints = tuple(
        item for item in request.generation_constraints
        if item not in {
            "Return a concise answer on the normal planned execution path.",
            "Choose roles only when their declared responsibilities match a stage in the approved plan; no role is mandatory for every task.",
        }
    )
    extra = ()
    if request.metadata.get("prompt_profile") in {ROLECHANGE_V5_STRATEGY, ROLECHANGE_V6_STRATEGY}:
        from agent_harness.orchestration.reflection import PLANNER_MODES
        extra = (PLANNER_MODES,)
    return replace(request, generation_constraints=constraints + (PLANNING_RULES, CODE_RULES) + extra)


def with_evolution_context(render: Any, *, planning: bool = False) -> Any:
    @wraps(render)
    def wrapped(request: Any, *args: Any, **kwargs: Any) -> str:
        prompt = render(request, *args, **kwargs)
        if not is_fusion(request):
            return prompt
        # Select fields explicitly: caller metadata cannot smuggle a prior source/trace.
        context = request.metadata.get("evolution_context", {})
        inherited = {key: context.get(key) for key in (
            "round", "active_facts", "disputed_facts", "previous_failure_report", "fact_store_revision",
        )}
        modes = ""
        if request.metadata.get("prompt_profile") in {ROLECHANGE_V5_STRATEGY, ROLECHANGE_V6_STRATEGY}:
            from agent_harness.orchestration.reflection import PLANNER_MODES
            modes = PLANNER_MODES
        return (
            prompt + modes + "\n\n## Outer evolution requirements\n" + PLANNING_RULES
            + ("" if planning else "\n" + CODE_RULES)
            + "\n## Cross-round facts and feedback\n"
            + json.dumps(inherited, ensure_ascii=False, indent=2)
        )
    return wrapped


def validate_proposal_source(tree: ast.AST) -> list[tuple[str, str]]:
    issues = []
    has_verifier = False
    has_proposal = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            keys = {key.value for key in node.keys if isinstance(key, ast.Constant)}
            has_proposal |= PROPOSAL_KEYS.issubset(keys)
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "ctx"):
            continue
        if node.func.attr in {"call", "spawn"} and node.args:
            role = node.args[0].value if isinstance(node.args[0], ast.Constant) else None
            if role not in SCRIPT_ROLES:
                issues.append(("fusion_role", "fusion scripts may call only the five business roles"))
            has_verifier |= role == "verifier"
    if not has_verifier:
        issues.append(("missing_final_verification", "include an actual verifier call for the candidate"))
    if not has_proposal:
        issues.append(("missing_candidate_proposal", "construct the six-field candidate proposal"))
    return issues


def validate_proposal_shape(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != PROPOSAL_KEYS:
        raise ValueError("fusion solve must return the six-field candidate proposal")
    json.dumps(value, allow_nan=False)
    if value["status"] not in {"ready", "incomplete"}:
        raise ValueError("proposal status must be ready or incomplete")
    if not isinstance(value["selected_candidate"], (str, dict)):
        raise ValueError("selected_candidate must be text or an object")
    if not isinstance(value["selected_verifier"], dict):
        raise ValueError("selected_verifier must be an actual verifier result object")
    if not all(isinstance(value[k], list) for k in ("facts", "issues")):
        raise ValueError("facts and issues must be lists")
    if not isinstance(value["synthesis_context"], dict):
        raise ValueError("synthesis_context must be an object")


def _event_type(event: Any) -> str:
    value = event.event_type
    return str(getattr(value, "value", value))


def _candidate_in_instruction(candidate: Any, instruction: str) -> bool:
    if not candidate:
        return False
    variants = [str(candidate)]
    if isinstance(candidate, dict):
        variants += [json.dumps(candidate, ensure_ascii=False, sort_keys=sort) for sort in (True, False)]
    normalized = " ".join(instruction.split())
    # Boundaries prevent a short answer such as '12' matching '123'. The Judge
    # separately checks semantic binding and that verification was substantive.
    return any(re.search(r"(?<!\w)" + re.escape(" ".join(item.split())) + r"(?!\w)", normalized)
               for item in variants if item.strip())


def bind_executed_verifier(
    proposal: dict[str, Any], events: Any, jobs: Any, execution_error: str = "",
) -> None:
    """Require an authentic, latest verdict for the exact submitted candidate."""
    verifier_ids = {job.job_id for job in jobs if job.role == "verifier"}
    inputs = {}
    original_results = {}
    relevant = []
    for event in events:
        if event.job_id not in verifier_ids:
            continue
        kind = _event_type(event)
        if kind == "job_created":
            inputs[event.job_id] = str(event.payload.get("instruction", ""))
        elif kind == "worker_event" and event.payload.get("name") == "evolution_role_result":
            original_results[event.job_id] = event.payload.get("data", {}).get("result")
        if kind in {"job_succeeded", "job_failed", "job_cancelled", "job_timed_out"}:
            if _candidate_in_instruction(proposal["selected_candidate"], inputs.get(event.job_id, "")):
                relevant.append(event)
    last = relevant[-1] if relevant else None
    actual = original_results.get(last.job_id) if last is not None else None
    eligible = (
        not execution_error and last is not None and _event_type(last) == "job_succeeded"
        and isinstance(actual, dict) and set(actual) == {"accepted", "critique"}
        and actual.get("accepted") is True and actual == proposal["selected_verifier"]
    )
    if not eligible:
        proposal["status"] = "incomplete"
        proposal["issues"].append(
            "No matching, successful latest verifier acceptance for the submitted candidate; "
            "fabricated, stale, rejected and failed verdicts are ineligible."
        )


class EvidenceObserver(BaseObserver):
    """Persist actual outputs through existing worker events before compaction."""
    critical = True

    def __init__(self, emit: Any) -> None:
        self.emit = emit

    async def on_llm_response(self, ctx: Any) -> None:
        await self.emit("evolution_llm_evidence", {
            "turn": ctx.turn, "content": ctx.ai_text, "tool_calls": ctx.tool_calls,
        })

    async def on_tool_result(self, ctx: Any, result: Any) -> None:
        await self.emit("evolution_tool_evidence", {"turn": ctx.turn, **asdict(result)})


def attach_worker_evidence(trace: dict[str, Any], events: Any) -> None:
    trace["worker_evidence"] = [
        {"job_id": event.job_id, "sequence": event.sequence, **event.payload}
        for event in events
        if _event_type(event) == "worker_event"
        and event.payload.get("name") in {
            "evolution_llm_evidence", "evolution_tool_evidence",
            "evolution_role_result", "worker_facts",
        }
    ]


def archive_previous_attempt(generated_root: Path, trial_dir: Path) -> None:
    """A retry starts fresh, while preserving previous attempt artifacts."""
    paths = [generated_root, trial_dir / "agent_traces", trial_dir / "agent"]
    existing = [path for path in paths if path.exists()]
    if not existing:
        return
    archive = trial_dir / "previous_attempts" / uuid.uuid4().hex
    archive.mkdir(parents=True)
    for path in existing:
        path.rename(archive / path.name)
