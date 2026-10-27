"""V5 diagnostics and evidence-grounded reflection contracts."""
from __future__ import annotations

import copy

TARGETS = ["SEARCH", "ORCHESTRATION", "REASONING", "VERIFICATION", "FACT_STATE",
           "CONTEXT_PROPAGATION", "EXECUTION", "UNKNOWN"]
MODES = ["REFINE", "REPLAN", "EXPLORE", "STOP"]
REFLECTOR_PROMPT = """STAGE TWO: Diagnose the failed execution using the frozen facts,
stage-one assessment, diagnostic bundle and read-only trace tools. This is one reflective
agent invocation with multiple internal tool turns. Form concrete competing failure
hypotheses; locate relevant events with trace_index; inspect selected evidence with
trace_query and facts_query; mark hypotheses supported, rejected or uncertain. Cite exact
fact IDs and trace IDs/references. Trace shows execution, not proof of answer correctness.
Do not modify facts, browse the web, execute code, or copy raw traces/programs into feedback.
Stop when evidence supports an actionable diagnosis, when two queries yield no new evidence,
or when retrieval budget is exhausted. If evidence is unavailable, say UNKNOWN; do not invent
execution failures. Provide concise diagnostic conclusions, not a reasoning transcript.
Choose REFINE to preserve a supported direction and fix a local failure; REPLAN to redesign
roles/order/information flow; EXPLORE to change candidates or search hypotheses. STOP is only
for round three. Every non-UNKNOWN root cause must cite resolvable evidence. Populate the
existing summary, next_round_priorities and planner_instructions (priority_checks, actions,
avoid_repeating, acceptance_conditions) with specific actionable feedback. Return JSON only.
"""
PLANNER_MODES = """
V5 reflection modes: REFINE preserves supported hypotheses and fixes the specified local failure;
REPLAN retains facts but changes role decomposition, ordering or information flow;
EXPLORE changes candidate families, search hypotheses or evidence paths.
Always generate a fresh script, never reconstruct prior code. Trace references in feedback
are audit citations, not permission to read prior traces. Facts remain the answer evidence.
"""


def report_schema(report: dict, round_number: int) -> dict:
    report = copy.deepcopy(report)
    strings = {"type": "array", "items": {"type": "string"}, "maxItems": 20}
    fields = {"root_cause": {"type": "string"},
              "failure_target": {"type": "string", "enum": TARGETS},
              "evolution_mode": {"type": "string", "enum": ["STOP"] if round_number == 3 else MODES[:3]},
              "fact_refs": strings, "trace_refs": strings,
              "diagnostic_hypotheses": {"type": "array", "maxItems": 8, "items": {
                  "type": "object", "additionalProperties": False,
                  "required": ["hypothesis", "verdict", "fact_refs", "trace_refs"],
                  "properties": {"hypothesis": {"type": "string"},
                                 "verdict": {"type": "string", "enum": ["supported", "rejected", "uncertain"]},
                                 "fact_refs": strings, "trace_refs": strings}}}}
    report["properties"].update(fields)
    report["required"] = list(dict.fromkeys([*report["required"], *fields]))
    return report


def normalize_reflection(report: dict, raw: dict | None, *, round_number: int,
                         scope=None, snapshot=None, error="") -> dict:
    raw = raw if isinstance(raw, dict) and not error else {}
    strings = lambda value: [x for x in value[:20] if isinstance(x, str)] if isinstance(value, list) else []
    fact_ids = {f["id"] for f in (snapshot or {}).get("facts", [])}
    invalid = []

    def refs(values, kind):
        result = []
        for ref in strings(values):
            known = ref in fact_ids if kind == "fact" else (
                scope is not None and (ref in scope.records or bool(scope.aliases.get(ref))))
            if known:
                result.append(ref)
            else:
                invalid.append(ref)
        return result

    hypotheses = []
    for item in raw.get("diagnostic_hypotheses", [])[:8]:
        if not isinstance(item, dict):
            continue
        fr, tr = refs(item.get("fact_refs"), "fact"), refs(item.get("trace_refs"), "trace")
        verdict = item.get("verdict", "uncertain")
        if verdict not in {"supported", "rejected", "uncertain"} or not (fr or tr):
            verdict = "uncertain"
        hypotheses.append({"hypothesis": str(item.get("hypothesis", ""))[:1500],
                           "verdict": verdict, "fact_refs": fr, "trace_refs": tr})
    fr, tr = refs(raw.get("fact_refs"), "fact"), refs(raw.get("trace_refs"), "trace")
    target = raw.get("failure_target", "UNKNOWN")
    if target not in TARGETS or not (fr or tr):
        target = "UNKNOWN"
    mode = raw.get("evolution_mode", "REFINE")
    if mode not in MODES[:3]:
        mode = "REFINE"
    report.update(root_cause=str(raw.get("root_cause") or "Diagnosis unavailable; inspect recorded errors.")[:3000],
                  failure_target=target, evolution_mode="STOP" if round_number == 3 else mode,
                  diagnostic_hypotheses=hypotheses, fact_refs=fr, trace_refs=tr,
                  continue_evolution=round_number < 3)
    if invalid:
        report["invalid_evidence_refs"] = list(dict.fromkeys(invalid))[:20]
        report["root_cause"] = "Diagnosis uncertain: some evidence references could not be resolved."
        report["failure_target"] = "UNKNOWN"
    if error:
        report["report_error"] = error[:2000]
    return report


def bounded(value, limit=2000):
    if isinstance(value, str):
        return value[:limit]
    if isinstance(value, dict):
        return {str(k): bounded(v, limit) for k, v in list(value.items())[:30]}
    if isinstance(value, (list, tuple)):
        return [bounded(v, limit) for v in value[:30]]
    return value


def build_diagnostic_bundle(*, task, round_number, proposal, assessment, gate, trace,
                            frozen_facts, previous_snapshot, previous_report, usage,
                            synthesis_error, trace_limits, errors=None):
    before = {f["id"]: f for f in previous_snapshot.get("facts", [])}
    after = {f["id"]: f for f in frozen_facts.get("facts", [])}
    return {"task": task, "round": round_number, "remaining_rounds": 3 - round_number,
            "candidate": bounded(proposal.get("selected_candidate", "")),
            "stage1_assessment": bounded(assessment), "actual_gate": bounded(gate),
            "generation_status": bounded(trace.get("generation", {})),
            "execution_error": bounded(trace.get("execution_error", "")),
            "synthesis_error": bounded(synthesis_error),
            "ordinary_verifier": bounded(proposal.get("selected_verifier", {})),
            "jobs": [{k: bounded(j.get(k)) for k in ("job_id", "role", "status", "error")}
                     for j in trace.get("jobs", [])[:50]],
            "fact_revision": frozen_facts.get("revision"),
            "fact_delta": {"added": sorted(after.keys() - before.keys()),
                           "removed": sorted(before.keys() - after.keys()),
                           "updated": sorted(k for k in before.keys() & after.keys() if before[k] != after[k])},
            "previous_failure_report": bounded(previous_report), "usage": usage,
            "trace_limits": trace_limits, "trace_errors": (errors or [])[:10]}


SLICE_PROMPT = """
V6: A failure-relevant Diagnostic Slice may be provided. Start trace retrieval from
scope=slice, then expand to scope=round, then scope=run only to test an important
unresolved hypothesis. Supply expansion_reason naming the concrete missing evidence;
provide it also for direct trace_query references outside the slice. If the bundle
reports a slice failure, use current-round retrieval and acknowledge the coverage gap.
The slice is a retrieval prior, not proof of causality. READ_FULL, READ_SUMMARY and
TRANSFERRED record exposure, not semantic use or verification. Absence of a flow event
alone does not establish information loss: equivalent information can be in ordinary
role results or instructions. Inspect those records before diagnosing propagation failure.
"""
