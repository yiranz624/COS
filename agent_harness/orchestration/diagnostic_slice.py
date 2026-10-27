"""Deterministic failure-relevant dependency slicing; no semantic causality claims."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict

DEFAULT_LIMITS = {"depth": 2, "fact_nodes": 30, "job_nodes": 20, "trace_refs": 100}


def event_data(record):
    try:
        obj = json.loads(record["content"])
    except (ValueError, TypeError):
        return {}, record.get("event_type", "")
    if not isinstance(obj, dict):
        return {}, record.get("event_type", "")
    payload = obj.get("详细信息", obj.get("payload", obj))
    if not isinstance(payload, dict):
        payload = {}
    kind = payload.get("name", record.get("event_type", ""))
    return payload.get("data", payload), kind


def build_diagnostic_slice(*, round_number, assessment, gate, proposal, frozen_facts,
                           fact_flow, trace_store, limits=None):
    limits = {**DEFAULT_LIMITS, **(limits or {})}
    facts = {f["id"]: f for f in frozen_facts.get("facts", [])}
    records = {tid: row for tid, row in trace_store.records.items()
               if row["round"] <= round_number and row["stage"] != "report"}
    jobs, edges, reasons = {}, set(), defaultdict(set)
    unresolved, warnings = [], list(fact_flow.get("coverage_warnings", []))
    seed_facts, seed_jobs, seed_traces = set(), set(), set()
    results = {}
    for tid, row in records.items():
        job = row.get("job_id")
        data, kind = event_data(row)
        if not isinstance(data, dict):
            data = {}
        if job:
            jobs.setdefault(job, {"role": row.get("role", ""), "round": row["round"]})
            if row.get("role"):
                jobs[job]["role"] = row["role"]
            parent = data.get("parent_job_id") or data.get("source_job_id")
            if parent:
                edges.add(("parent_of", "job:" + parent, "job:" + job))
            if kind == "evolution_role_result":
                results[job] = data.get("result")
            if row["round"] == round_number and any(x in kind for x in ("failed", "timed_out", "cancelled")):
                seed_jobs.add(job)
                reasons["job:" + job].add("failed_job")
                seed_traces.add(tid)
        if row["round"] == round_number and row["stage"] in {"diagnostics", "generation"}:
            seed_traces.add(tid)
    for event in fact_flow.get("events", []):
        if event["round"] > round_number or event["stage"] in {"report", "assessment"}:
            continue
        fid, source, target = event.get("fact_id"), event.get("source_job_id"), event.get("target_job_id")
        kind = event["event_type"]
        if fid and source and kind == "COMMITTED":
            edges.add(("produces", "job:" + source, "fact:" + fid))
        if fid and target and kind in {"READ_FULL", "READ_SUMMARY", "TRANSFERRED"}:
            edges.add((kind, "fact:" + fid, "job:" + target))
        if fid and source and kind in {"DISPUTED", "DELETED", "UPDATED"}:
            edges.add((kind, "job:" + source, "fact:" + fid))
    for fid, fact in facts.items():
        for other in fact.get("conflicts_with", []):
            edges.add(("conflicts_with", "fact:" + fid, "fact:" + other))
    for constraint in assessment.get("constraints", []):
        if constraint.get("status") not in {"unsatisfied", "unknown"}:
            continue
        ids = set(constraint.get("fact_ids", [])) & facts.keys()
        if not ids:
            unresolved.append({"constraint": constraint.get("constraint", ""), "reason": "no_resolvable_fact"})
        for fid in ids:
            seed_facts.add(fid)
            reasons["fact:" + fid].add("constraint:" + constraint.get("constraint", ""))
    # Exact whole-result matches only; ambiguity is retained, never choose the last job.
    for role, value in (("solver", proposal.get("selected_candidate")),
                        ("verifier", proposal.get("selected_verifier"))):
        candidates = {job for job, meta in jobs.items() if meta["round"] == round_number and meta["role"] == role}
        matched = {job for job in candidates if value is not None and job in results and results[job] == value}
        selected = matched or candidates
        if len(matched) != 1:
            warnings.append({"role": role, "reason": "ambiguous_or_unknown_candidate_origin", "jobs": sorted(selected)})
        seed_jobs.update(selected)
        for job in selected:
            reasons["job:" + job].add("exact_result" if matched else "conservative_candidate_origin")
    for job, meta in jobs.items():
        if meta["round"] == round_number and meta["role"] == "synthesizer":
            seed_jobs.add(job)
            reasons["job:" + job].add("candidate_synthesis")
    neighbors = defaultdict(set)
    for kind, source, target in sorted(edges):
        neighbors[source].add(target)
        neighbors[target].add(source)
    seeds = {"fact:" + f for f in seed_facts} | {"job:" + j for j in seed_jobs}
    selected, omitted = set(), set()
    frontier = sorted(seeds)
    counts = {"fact": 0, "job": 0}
    for depth in range(limits["depth"] + 1):
        next_frontier = set()
        for node in frontier:
            if node in selected:
                continue
            kind = node.split(":", 1)[0]
            if counts[kind] >= limits[kind + "_nodes"]:
                omitted.add(node)
                continue
            selected.add(node)
            counts[kind] += 1
            if depth < limits["depth"]:
                for neighbor in neighbors[node] - selected:
                    # Do not turn an orchestration root into all sibling jobs.
                    if neighbor.startswith("job:") and neighbor[4:] not in jobs:
                        continue
                    reasons[neighbor].add("neighbor_of:" + node)
                    next_frontier.add(neighbor)
        frontier = sorted(next_frontier)
    fact_ids = sorted(n[5:] for n in selected if n.startswith("fact:"))
    job_ids = sorted(n[4:] for n in selected if n.startswith("job:"))
    trace_ids = set(seed_traces)
    for tid, row in records.items():
        if row.get("job_id") in job_ids:
            trace_ids.add(tid)
    for fid in fact_ids:
        for ref in facts.get(fid, {}).get("trace_refs", []):
            trace_ids.update(t for t in trace_store.aliases.get(ref, [ref]) if t in records)
    for event in fact_flow.get("events", []):
        if event.get("fact_id") in fact_ids:
            trace_ids.update(t for t in trace_store.aliases.get(event["event_id"], []) if t in records)
    def rank(tid):
        row = records[tid]
        _, kind = event_data(row)
        # Keep job input/result and one complete trajectory ahead of repetitive turns.
        if tid in seed_traces:
            priority = 0
        elif kind in {"agent_loop_started", "evolution_role_result", "fact_submission", "evidence_store_committed"} or any(x in kind for x in ("failed", "timed_out")):
            priority = 1
        elif row["stage"] == "agent_trace" and row["source"].endswith(".json"):
            priority = 2
        elif row["stage"] == "fact_flow":
            priority = 3
        else:
            priority = 4
        return priority, tid
    ordered = sorted(trace_ids, key=rank)
    result = {"schema_version": "diagnostic_slice_v1", "round": round_number,
              "seed": {"fact_ids": sorted(seed_facts), "job_ids": sorted(seed_jobs)},
              "fact_ids": fact_ids, "job_ids": job_ids,
              "edges": [{"type": k, "source": a, "target": b} for k, a, b in sorted(edges)
                        if a in selected and b in selected],
              "trace_refs": ordered[:limits["trace_refs"]],
              "inclusion_reasons": {k: sorted(reasons[k]) for k in sorted(selected)},
              "limits": limits, "truncation": {"nodes": len(omitted), "trace_refs": max(0, len(ordered)-limits["trace_refs"])},
              "unresolved_seeds": unresolved, "coverage_warnings": warnings,
              "full_round_trace_count": sum(r["round"] == round_number for r in records.values())}
    result["slice_id"] = "slice-" + hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()[:24]
    return result
