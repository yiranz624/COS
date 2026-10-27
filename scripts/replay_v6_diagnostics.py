#!/usr/bin/env python3
"""Read historical V5 traces without changing them; emit partial V6 diagnostics.

Only recorded submissions, returned fact-tool results and exact input IDs are recovered.
Historical gaps are explicitly retained, not inferred as absence of information.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from agent_harness.orchestration.diagnostic_slice import build_diagnostic_slice, event_data
from agent_harness.orchestration.fact_flow import FACT_TOKEN, FactFlowLedger
from agent_harness.orchestration.trace_store import TraceStore


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else (default or {})


def recover_flow(store, snapshot, ledger):
    ledger.errors.append({"operation": "historical_replay", "error":
                          "Partial reconstruction: unrecorded reads/transfers and historical exposure revisions are unknown."})
    facts = {f["id"]: f for f in snapshot.get("facts", [])}
    for history in snapshot.get("history", []):
        for change in history.get("changes", []):
            if change.get("before"):
                facts.setdefault(change["fact_id"], change["before"])
    for row in store.records.values():
        data, kind = event_data(row)
        if not isinstance(data, dict) or row["stage"] == "fact_flow":
            continue
        common = {"round_number": row["round"], "runtime_id": row["runtime"],
                  "stage": row["stage"], "role": row.get("role", ""), "trace_refs": [row["trace_id"]]}
        if kind == "agent_loop_started":
            instruction = data.get("user_message", "")
            for fid in sorted(set(FACT_TOKEN.findall(instruction)) & facts.keys()):
                ledger.record("TRANSFERRED", **common, fact_id=fid,
                              target_job_id=row["job_id"], identity=[row["trace_id"], fid],
                              details={"reconstructed": True, "historical_revision": "unknown"})
        if kind == "fact_submission" and data.get("submission_id"):
            ledger.record("SUBMITTED", **common, source_job_id=data.get("job_id"),
                          submission_id=data["submission_id"], identity=data["submission_id"])
        if kind == "evolution_tool_evidence" and data.get("name") in {"facts_index", "facts_query"} and not data.get("is_error"):
            result = data.get("result")
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except ValueError:
                    try:
                        result = ast.literal_eval(result)
                    except (ValueError, SyntaxError):
                        result = None
            if isinstance(result, dict):
                full = data["name"] == "facts_query"
                for fact in result.get("facts" if full else "fact_index", []):
                    ledger.record("READ_FULL" if full else "READ_SUMMARY", **common,
                                  fact_id=fact["id" if full else "fact_id"], target_job_id=row["job_id"],
                                  revision=result.get("revision"), identity=[row["trace_id"], fact.get("id", fact.get("fact_id"))])
    for submission in snapshot.get("submissions", []):
        refs = set(submission.get("trace_refs", []))
        retained = [fid for fid, fact in facts.items() if refs & set(fact.get("trace_refs", []))]
        for fid in retained:
            ledger.record("COMMITTED", round_number=submission["round"], stage="evidence",
                          fact_id=fid, submission_id=submission["submission_id"],
                          source_job_id=submission["job_id"], role=submission.get("role", ""),
                          trace_refs=submission.get("trace_refs", []), identity=[submission["submission_id"], fid],
                          details={"reconstructed": True, "source": "persisted_authenticated_submission"})


def replay(trial, output, round_number):
    source_root = trial / "generated"
    rd = source_root / f"evolution_round_{round_number:02d}"
    review = rd / "terminal_review"
    structured = read(review / "structured_trace.json")
    snapshot = structured.get("fact_store_snapshot")
    if snapshot is None:
        raise ValueError("Round-local frozen snapshot unavailable; refuse to substitute future facts")
    bundle = read(review / "diagnostic_bundle.json")
    assessment = read(review / "assessment.json")
    report_input = read(review / "report_input.json")
    store = TraceStore(output)
    store.ingest(trial, round_number, snapshot, source_root=source_root)
    store.add(round_number=round_number, stage="diagnostics", runtime="replay", sequence="outcome",
              content={"gate": assessment, "execution_error": bundle.get("execution_error"),
                       "synthesis_error": bundle.get("synthesis_error")}, event_type="round_outcome")
    ledger = FactFlowLedger(output, store.attempt_id)
    recover_flow(store, snapshot, ledger)
    flow = ledger.freeze(round_number)
    store.ingest_flow(flow)
    proposal = {"selected_candidate": report_input.get("candidate_answer", bundle.get("candidate")),
                "selected_verifier": bundle.get("ordinary_verifier", {})}
    result = build_diagnostic_slice(round_number=round_number, assessment=assessment, gate=assessment,
                                   proposal=proposal, frozen_facts=snapshot, fact_flow=flow, trace_store=store)
    target = output / f"evolution_round_{round_number:02d}"
    ledger.persist_snapshot(target, round_number)
    (target / "terminal_review").mkdir(parents=True, exist_ok=True)
    (target / "terminal_review/diagnostic_slice.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    # Existing reflection is comparison-only, loaded strictly after slice construction.
    report = read(rd / "failure_report.json")
    old_index = read(source_root / "trace_store/index.json", [])
    old_records = {row["trace_id"]: row for row in old_index} if isinstance(old_index, list) else {}
    selected_sources = {(store.records[t]["runtime"], str(store.records[t]["sequence"])) for t in result["trace_refs"]}
    refs = [old_records[t] for t in report.get("trace_refs", []) if t in old_records]
    metrics = {"question": trial.name, "round": round_number, "partial_historical_reconstruction": True,
               "flow_events": len(flow["events"]), "fact_count": len(snapshot.get("facts", [])),
               "facts_with_producers": len({e["fact_id"] for e in flow["events"] if e["event_type"] == "COMMITTED"}),
               "slice_trace_count": len(result["trace_refs"]), "total_trace_count": len(store.records),
               "full_round_trace_count": result["full_round_trace_count"],
               "historical_reference_count": len(refs),
               "historical_reference_hits": sum((r["runtime"], str(r["sequence"])) in selected_sources for r in refs),
               "job_ids": result["job_ids"], "fact_ids": result["fact_ids"]}
    (target / "replay_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="V5 run directory containing trials")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--question-ids", default="1,6,13,14,39,43,51,54,61,64")
    args = parser.parse_args()
    source, output = args.source.resolve(), args.out.resolve()
    if source == output or source in output.parents or output in source.parents:
        parser.error("Output and source must be separate directory trees")
    metrics = []
    for question in args.question_ids.split(","):
        trial = source / "trials" / question
        for rd in sorted((trial / "generated").glob("evolution_round_*")):
            if (rd / "terminal_review/diagnostic_bundle.json").exists():
                number = int(rd.name.rsplit("_", 1)[1])
                metric = replay(trial, output / question / rd.name, number)
                metrics.append(metric)
                print(json.dumps(metric), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
