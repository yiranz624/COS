"""Attempt-local raw trace index and bounded, read-only reflector tools."""
from __future__ import annotations

import copy
import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any

from agent_harness.core.tool import Tool

DEFAULT_LIMITS = {"index_items": 30, "query_items": 5, "item_chars": 8000,
                  "query_chars": 20000, "calls": 12, "queries": 8, "total_chars": 60000}


def encoded(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


class TraceStore:
    """Immutable record copies with rebuild-stable identifiers and provenance aliases."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.path = self.root / "trace_store"
        self.path.mkdir(parents=True, exist_ok=True)
        attempt_file = self.path / "attempt.json"
        if not attempt_file.exists():
            attempt_file.write_text(encoded({"attempt_id": uuid.uuid4().hex}), encoding="utf-8")
        self.attempt_id = json.loads(attempt_file.read_text())["attempt_id"]
        self.records: dict[str, dict] = {}
        self.aliases: dict[str, list[str]] = {}
        self.errors: list[dict] = []

    def add(self, *, round_number: int, stage: str, runtime: str, sequence: Any,
            content: Any, job_id: str = "", role: str = "", event_type: str = "",
            source: str = "") -> str:
        key = encoded([self.attempt_id, round_number, runtime, sequence])
        tid = "trace-" + hashlib.sha256(key.encode()).hexdigest()[:32]
        text = content if isinstance(content, str) else encoded(content)
        self.records[tid] = {"trace_id": tid, "round": round_number, "stage": stage,
                             "runtime": runtime, "sequence": sequence, "job_id": job_id,
                             "role": role, "event_type": event_type, "source": source,
                             "content": text}
        return tid

    def ingest(self, trial_dir: Path, through_round: int, snapshot: dict, *, source_root: Path | None = None) -> None:
        """Read only current-attempt paths. Report traces never enter the query corpus."""
        trial_dir = Path(trial_dir)
        for number in range(1, through_round + 1):
            rd = (source_root or self.root) / f"evolution_round_{number:02d}"
            job_roles = {}
            for rel, stage in [("orchestration_live.jsonl", "execution"),
                               ("evidence_maintenance/live.jsonl", "evidence"),
                               ("terminal_review/judge_live.jsonl", "assessment"),
                               ("terminal_review/synthesis_live.jsonl", "synthesis")]:
                path = rd / rel
                if not path.is_file():
                    continue
                runtime = str(path.relative_to(trial_dir))
                events = []
                for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    try:
                        event = json.loads(line)
                        if not isinstance(event, dict):
                            raise TypeError("event must be an object")
                        events.append((line_no, event))
                    except (ValueError, TypeError) as exc:
                        self.errors.append({"source": runtime, "line": line_no, "error": str(exc)})
                roles = {}
                for _, event in events:
                    payload = event.get("详细信息", event.get("payload", {}))
                    data = payload.get("data", payload) if isinstance(payload, dict) else {}
                    if isinstance(data, dict) and data.get("role"):
                        roles[event.get("job_id")] = data["role"]
                job_roles.update(roles)
                for line_no, event in events:
                    self.add(round_number=number, stage=stage, runtime=runtime,
                             sequence=event.get("sequence", line_no), content=event,
                             job_id=event.get("job_id") or "", role=roles.get(event.get("job_id"), ""),
                             event_type=event.get("事件类型", event.get("event_type", "")), source=runtime)
            # Planner scripts and generation artifacts are available for diagnosis only.
            for name in ("solve.py", "plan.txt", "generation.json", "generation_status.json"):
                for path in sorted(rd.rglob(name)):
                    if path.is_symlink():
                        continue
                    source = str(path.relative_to(trial_dir))
                    self.add(round_number=number, stage="generation", runtime=source, sequence=0,
                             content=path.read_text(encoding="utf-8"), event_type="artifact", source=source)
            agent_dir = trial_dir / "agent_traces" / f"evolution_round_{number:02d}"
            if agent_dir.exists():
                for path in sorted(agent_dir.rglob("*.json*")):
                    if "report" in path.relative_to(agent_dir).parts or path.is_symlink():
                        continue
                    source = str(path.relative_to(trial_dir))
                    candidates = [job for job, role in job_roles.items() if job and
                                  path.stem == (job + "_" + role).replace("/", "__").replace(":", "_")]
                    job = candidates[0] if len(candidates) == 1 else ""
                    self.add(round_number=number, stage="agent_trace", runtime=source, sequence=0,
                             content=path.read_text(encoding="utf-8"), job_id=job,
                             role=job_roles.get(job, ""), event_type="agent_trace", source=source)
        # The evidence store persists authenticated submissions, even if their facts were removed.
        for index, submission in enumerate(snapshot.get("submissions", [])):
            if not isinstance(submission, dict):
                continue
            number = int(submission.get("round", through_round))
            job = submission.get("job_id", "")
            sid = submission.get("submission_id", str(index))
            tid = self.add(round_number=number, stage="evidence", runtime="submissions",
                           sequence=sid, content=submission, job_id=job,
                           role=submission.get("role", ""), event_type="fact_submission")
            for ref in submission.get("trace_refs", []):
                linked = [tid] + [key for key, row in self.records.items()
                                  if row["round"] == number and row["job_id"] == job and key != tid]
                self.aliases[ref] = list(dict.fromkeys([*self.aliases.get(ref, []), *linked]))
        # Also resolve legacy references by their recorded job, without inventing a submission.
        for fact in snapshot.get("facts", []):
            for ref in fact.get("trace_refs", []):
                match = re.match(r"^round:(\d+):job:(.+):submission-[^:]+$", ref)
                if ref not in self.aliases and match:
                    number, job = int(match[1]), match[2]
                    self.aliases[ref] = [key for key, row in self.records.items()
                                         if row["round"] == number and row["job_id"] == job]
        self.persist()

    def persist(self) -> None:
        with (self.path / "raw_trace.jsonl").open("w", encoding="utf-8") as stream:
            for record in self.records.values():
                stream.write(encoded(record) + "\n")
        index = [{k: v for k, v in row.items() if k != "content"} for row in self.records.values()]
        (self.path / "index.json").write_text(encoded(index), encoding="utf-8")
        (self.path / "manifest.json").write_text(encoded({"attempt_id": self.attempt_id,
            "aliases": self.aliases, "records": len(index), "errors": self.errors}), encoding="utf-8")

    def ingest_flow(self, snapshot: dict) -> None:
        for event in snapshot.get("events", []):
            if event["stage"] == "report":
                continue
            tid = self.add(round_number=event["round"], stage="fact_flow", runtime="fact_flow",
                           sequence=event["event_id"], content=event,
                           job_id=(event.get("target_job_id") if event["event_type"] in
                                   {"READ_FULL", "READ_SUMMARY", "TRANSFERRED"} else event.get("source_job_id")) or "",
                           role=event.get("role", ""), event_type=event["event_type"],
                           source="fact_flow/events.jsonl")
            self.aliases[event["event_id"]] = [tid]
        self.persist()

    def freeze(self, current_round: int, preferred_trace_ids=None) -> TraceScope:
        return TraceScope(copy.deepcopy(self.records), copy.deepcopy(self.aliases), current_round,
                          preferred_trace_ids=preferred_trace_ids)


class TraceScope:
    def __init__(self, records: dict, aliases: dict, current_round: int, preferred_trace_ids=None):
        self.records, self.aliases, self.current_round = records, aliases, current_round
        self.limits = dict(DEFAULT_LIMITS)
        self.calls = self.queries = self.returned_chars = 0
        self.seen: set[str] = set()
        self.no_new_results = 0
        self.slice_ids = set(preferred_trace_ids or []) & records.keys()
        self.expansions = []
        self.slice_queries = self.queried_inside_slice = self.queried_outside_slice = 0
        self.queried_ids: set[str] = set()

    def _begin(self, query: bool = False) -> bool:
        if (self.calls >= self.limits["calls"] or self.returned_chars >= self.limits["total_chars"]
                or (query and self.queries >= self.limits["queries"])):
            return False
        self.calls += 1
        self.queries += int(query)
        return True

    def _finish(self, result: dict) -> dict:
        remaining = self.limits["total_chars"] - self.returned_chars
        if len(encoded(result)) > remaining:
            result = {"status": "budget_exhausted", "instruction": "Finish using existing evidence."}
        else:
            self.returned_chars += len(encoded(result))
        return result

    async def index(self, *, rounds=None, stages=None, roles=None, job_id="", keyword="",
                    offset=0, limit=30, scope=None, expansion_reason="") -> dict:
        if offset < 0 or not 1 <= limit <= self.limits["index_items"]:
            return {"status": "invalid_request", "message": "Invalid offset or index limit"}
        scope = scope or ("run" if rounds == [] else "round" if rounds is not None
                          else "slice" if self.slice_ids else "round")
        if scope not in {"slice", "round", "run"}:
            return {"status": "invalid_request", "message": "Unknown scope"}
        legacy_rounds = rounds is not None
        if self.slice_ids and scope != "slice" and not expansion_reason and not legacy_rounds:
            return {"status": "invalid_request", "message": "expansion_reason is required"}
        if not self._begin():
            return {"status": "budget_exhausted"}
        if self.slice_ids and scope != "slice":
            self.expansions.append({"scope": scope, "reason": expansion_reason or "legacy rounds filter"})
        if scope == "slice":
            self.slice_queries += 1
        rounds = ([self.current_round] if scope == "round" else []) if rounds is None else rounds
        rows = [r for r in self.records.values() if (not rounds or r["round"] in rounds)
                and (scope != "slice" or r["trace_id"] in self.slice_ids)
                and (scope != "round" or r["round"] == self.current_round)
                and r["round"] <= self.current_round
                and (not stages or r["stage"] in stages) and (not roles or r["role"] in roles)
                and (not job_id or r["job_id"] == job_id)
                and (not keyword or keyword.casefold() in r["content"].casefold())]
        items = [{**{k: v for k, v in r.items() if k != "content"}, "summary": r["content"][:240]}
                 for r in rows[offset:offset + limit]]
        return self._finish({"status": "ok", "items": items, "total": len(rows),
                             "next_offset": offset + limit if offset + limit < len(rows) else None})

    async def query(self, *, trace_ids: list[str], offset=0, max_chars=8000, expansion_reason="") -> dict:
        if (not trace_ids or len(trace_ids) > self.limits["query_items"] or offset < 0
                or not 1 <= max_chars <= self.limits["item_chars"]):
            return {"status": "invalid_request", "message": "Invalid IDs, offset or length"}
        resolved = [ref if ref in self.records else (self.aliases.get(ref) or [None])[0]
                    for ref in trace_ids]
        outside = [tid for tid in resolved if tid in self.records and tid not in self.slice_ids]
        if self.slice_ids and outside and not expansion_reason:
            return {"status": "invalid_request", "message": "expansion_reason is required outside slice"}
        if not self._begin(True):
            return {"status": "budget_exhausted"}
        if self.slice_ids and outside:
            scope = "run" if any(self.records[t]["round"] < self.current_round for t in outside) else "round"
            self.expansions.append({"scope": scope, "reason": expansion_reason})
        elif self.slice_ids:
            self.slice_queries += 1
        items = []
        available = self.limits["query_chars"]
        new_ids = set()
        for ref in trace_ids:
            ids = [ref] if ref in self.records else self.aliases.get(ref, [])
            if not ids:
                items.append({"reference": ref, "status": "missing"})
                continue
            # An alias reads its primary record; associated event IDs are bounded and pageable via index.
            row = self.records[ids[0]]
            if row["round"] > self.current_round:
                items.append({"reference": ref, "status": "missing"})
                continue
            self.queried_ids.add(row["trace_id"])
            if row["trace_id"] in self.slice_ids:
                self.queried_inside_slice += 1
            else:
                self.queried_outside_slice += 1
            text = row["content"]
            end = min(len(text), offset + min(max_chars, available))
            fragment = text[offset:end]
            available -= len(fragment)
            items.append({"reference": ref, "trace_id": row["trace_id"], "source": row["source"],
                          "content": fragment, "offset": offset,
                          "next_offset": end if end < len(text) else None,
                          "truncated": end < len(text), "related_ids": ids[1:11],
                          "related_total": max(0, len(ids) - 1), "job_id": row["job_id"],
                          "round": row["round"]})
            # New fragments count as new evidence even when continuing the same event.
            new_ids.add(f"{row['trace_id']}:{offset}:{end}")
        self.no_new_results = 0 if new_ids - self.seen else self.no_new_results + 1
        self.seen.update(new_ids)
        return self._finish({"status": "ok", "items": items,
                             "stop_recommended": self.no_new_results >= 2})

    def usage(self, final_refs=()) -> dict:
        resolved = {ref if ref in self.records else (self.aliases.get(ref) or [None])[0]
                    for ref in final_refs}
        resolved.discard(None)
        return {"calls": self.calls, "queries": self.queries, "returned_chars": self.returned_chars,
                "slice_queries": self.slice_queries, "round_expansions": sum(e["scope"] == "round" for e in self.expansions),
                "run_expansions": sum(e["scope"] == "run" for e in self.expansions),
                "queried_inside_slice": self.queried_inside_slice,
                "queried_outside_slice": self.queried_outside_slice, "expansions": self.expansions,
                "final_reference_count": len(resolved), "slice_reference_hits": len(resolved & self.slice_ids)}

    def tools(self) -> list[Tool]:
        strings = {"type": "array", "items": {"type": "string"}}
        return [Tool("trace_index", "Locate execution records. Default slice if provided, otherwise current round. Expand scope with a concrete expansion_reason; legacy rounds=[] searches all allowed rounds.",
                     {"type": "object", "additionalProperties": False, "properties": {
                         "scope": {"type": "string", "enum": ["slice", "round", "run"]},
                         "expansion_reason": {"type": "string"},
                         "rounds": {"type": "array", "items": {"type": "integer"}},
                         "stages": strings, "roles": strings, "job_id": {"type": "string"},
                         "keyword": {"type": "string"}, "offset": {"type": "integer", "minimum": 0},
                         "limit": {"type": "integer", "minimum": 1, "maximum": 30}}}, self.index),
                Tool("trace_query", "Read selected raw records or Fact trace_refs. Use next_offset for continuation.",
                     {"type": "object", "required": ["trace_ids"], "additionalProperties": False,
                      "properties": {"expansion_reason": {"type": "string"}, "trace_ids": {**strings, "minItems": 1, "maxItems": 5},
                                     "offset": {"type": "integer", "minimum": 0},
                                     "max_chars": {"type": "integer", "minimum": 1, "maximum": 8000}}}, self.query)]
