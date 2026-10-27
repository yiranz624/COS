"""Legacy fact-envelope readers for historical artifacts.

The live v4 executor no longer invokes these helpers. New facts enter only through
facts_submit; round-end maintenance reads committed submissions and the fact table.
"""
from __future__ import annotations

import copy
import json
from typing import Any

WORKER_ROLES = {"decomposer", "researcher", "solver", "verifier", "synthesizer"}
WORKER_FACT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "claim": {"type": "string"},
        "source_urls": {"type": "array", "items": {"type": "string"}},
        "fact_ids": {"type": "array", "items": {"type": "string"}},
        "evidence_excerpt": {"type": "string", "minLength": 1},
    },
    "required": ["claim", "source_urls", "evidence_excerpt"],
}
WORKER_FACT_PROMPT = (
    "\nFUSION FACT ENVELOPE: Return {\"result\": your ordinary role result, \"facts\": [...]}. "
    "The executor unwraps result for the script; use its original type and contract. "
    "Extract task-relevant factual observations with claim, source_urls and a nonempty "
    "evidence_excerpt. Include concrete supporting evidence, source correspondence, dates "
    "and qualifications so maintenance can evaluate it without the full trace. Keep it concise "
    "but complete; it is not truncated. Do not assign status or decide conflicts, validity or "
    "replacement relationships. fact_ids optionally identifies related existing facts for "
    "retrieval only. Do not supply job IDs, role, round or trace_refs; the executor attaches "
    "provenance. Use facts=[] when "
    "there are no observations. In intermediate messages you may emit a JSON object with "
    "facts to preserve evidence before compaction or failure; never repeat entire webpages."
)


def worker_definition(definition):
    schema = {"type": "object", "additionalProperties": False,
              "properties": {"result": definition.result_schema or {"type": "string"},
                             "facts": {"type": "array", "items": WORKER_FACT_SCHEMA}},
              "required": ["result", "facts"], "x-fusion-facts": True}
    return definition.model_copy(deep=True, update={
        "system_prompt": definition.system_prompt + WORKER_FACT_PROMPT, "result_schema": schema,
    })


def parse_fact_message(text: str) -> Any:
    if not isinstance(text, str):
        return None
    text = text.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    elif text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def fact_fields(value: Any) -> list:
    """Read only explicitly supported locations, never arbitrary nested keys."""
    if not isinstance(value, dict):
        return []
    if isinstance(value.get("facts"), list):
        return value["facts"]
    return []


def normalize_facts(values: list, *, round_number: int, role: str, job_id: str,
                    trace_ref: str) -> tuple[list[dict], int]:
    facts = []
    invalid = 0
    for value in values:
        if not isinstance(value, dict) or not isinstance(value.get("claim"), str) or not value["claim"].strip():
            invalid += 1
            continue
        fields = {}
        valid = True
        for key in ("source_urls", "fact_ids"):
            items = value.get(key, [])
            if not isinstance(items, list) or not all(isinstance(i, str) for i in items):
                valid = False
                break
            fields[key] = list(dict.fromkeys(items))
        if not valid:
            invalid += 1
            continue
        excerpt = value.get("evidence_excerpt", "")
        if not isinstance(excerpt, str) or not excerpt.strip():
            invalid += 1
            continue
        facts.append({"claim": value["claim"].strip(),
                      **fields, "evidence_excerpt": excerpt,
                      "trace_refs": [trace_ref],
                      "origins": [{"round": round_number, "role": role,
                                   "job_id": job_id, "trace_ref": trace_ref}]})
    return facts, invalid


def collect_facts(trace: dict, round_number: int) -> tuple[list[dict], dict]:
    collected = []
    diagnostics = {"fact_events": 0, "invalid_facts": 0, "missing_fact_events": 0}
    jobs = {j.get("job_id", j.get("id")): j.get("role") for j in trace.get("jobs", [])}
    # Runtime events include job_created even when a worker fails before final output.
    runtime_events = trace.get("events", [])
    for event in runtime_events:
        if event.get("event_type") == "job_created":
            jobs[event.get("job_id")] = event.get("payload", {}).get("role")
    events = trace.get("worker_evidence")
    if events is None:
        events = [{"job_id": e.get("job_id"), "sequence": e.get("sequence"), **e.get("payload", {})}
                  for e in runtime_events if e.get("event_type") == "worker_event"]
    canonical_jobs = {e.get("job_id") for e in events if e.get("name") == "worker_facts"}
    for event in events:
        job_id = event.get("job_id") or "unknown-worker"
        data = event.get("data", {})
        role = jobs.get(job_id) or data.get("role")
        if role not in WORKER_ROLES:
            continue
        name = event.get("name")
        if name == "worker_facts":
            values = data.get("facts", [])
            trace_ref = data.get("trace_ref")
            # Generated by the executor, not a model result field.
            if not isinstance(trace_ref, str):
                continue
        elif job_id not in canonical_jobs and name in {"evolution_llm_evidence", "evolution_role_result"}:
            value = (parse_fact_message(data.get("content", "")) if name == "evolution_llm_evidence"
                     else data.get("result"))
            values = fact_fields(value)
            trace_ref = f"round:{round_number}:job:{job_id}:event:{event.get('sequence', 0)}"
        else:
            continue
        diagnostics["fact_events"] += 1
        facts, invalid = normalize_facts(values, round_number=round_number, role=role,
                                         job_id=job_id, trace_ref=trace_ref)
        diagnostics["invalid_facts"] += invalid + data.get("invalid_facts", 0)
        if not values:
            diagnostics["missing_fact_events"] += 1
        collected.extend(facts)
    proposal = trace.get("candidate_proposal", {})
    facts, invalid = normalize_facts(
        fact_fields(proposal), round_number=round_number, role="proposal", job_id="proposal",
        trace_ref=f"round:{round_number}:proposal:facts",
    )
    collected.extend(facts)
    diagnostics["invalid_facts"] += invalid
    unique = {}
    for fact in collected:
        # Merge exact claims while preserving all evidence and provenance.
        key = " ".join(fact["claim"].casefold().split())
        if key not in unique:
            unique[key] = copy.deepcopy(fact)
        else:
            target = unique[key]
            for field in ("source_urls", "fact_ids", "trace_refs", "origins"):
                for item in fact[field]:
                    if item not in target[field]:
                        target[field].append(item)
            if fact["evidence_excerpt"] and fact["evidence_excerpt"] not in target["evidence_excerpt"]:
                target["evidence_excerpt"] = target["evidence_excerpt"] + "\n" + fact["evidence_excerpt"]
    diagnostics["collected_facts"] = len(unique)
    return list(unique.values()), diagnostics


def compact_packet(trace: dict, snapshot: dict, round_number: int) -> dict:
    new, diagnostics = collect_facts(trace, round_number)
    ids = {fid for fact in new for fid in fact["fact_ids"]}
    claims = {" ".join(f["claim"].casefold().split()) for f in new}
    related = [copy.deepcopy(f) for f in snapshot["facts"]
               if f["status"] in {"active", "disputed"} and (
                   f["id"] in ids
                   or " ".join(f["claim"].casefold().split()) in claims)]
    return {"round": round_number, "store_revision": snapshot["revision"], "new_facts": new,
            "related_existing_facts": related, "collection_diagnostics": diagnostics}
