"""Host-authenticated, append-only observations of explicit fact movement.

Exposure is not semantic use. Missing observations are not proof of absence.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

EVENT_TYPES = frozenset({"SUBMITTED", "COMMITTED", "REJECTED_SUBMISSION", "READ_SUMMARY",
                        "READ_FULL", "TRANSFERRED", "DISPUTED", "DELETED", "UPDATED"})
FACT_TOKEN = re.compile(r"(?<![\w-])fact-[A-Za-z0-9_-]+(?![\w-])")


@dataclass(frozen=True)
class FactFlowEvent:
    event_id: str
    attempt_id: str
    round: int
    runtime_id: str
    stage: str
    event_type: str
    fact_id: str | None = None
    submission_id: str | None = None
    source_job_id: str | None = None
    target_job_id: str | None = None
    role: str = ""
    revision: int | None = None
    trace_refs: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


class FactFlowLedger:
    def __init__(self, root: Path, attempt_id: str):
        self.path = Path(root) / "fact_flow"
        self.attempt_id = attempt_id
        self.events: list[dict] = []
        self.errors: list[dict] = []
        self._keys: set[str] = set()
        self._writable = True
        try:
            self.path.mkdir(parents=True, exist_ok=True)
            path = self.path / "events.jsonl"
            if path.exists():
                for line in path.read_text().splitlines():
                    row = json.loads(line)
                    if row["attempt_id"] != attempt_id:
                        self.events.clear()
                        self._keys.clear()
                        self._writable = False
                        raise ValueError("Fact flow attempt mismatch")
                    if row["event_id"] not in self._keys:
                        self.events.append(row)
                        self._keys.add(row["event_id"])
        except (OSError, ValueError, KeyError) as exc:
            self.errors.append({"operation": "load", "error": str(exc)})

    def record(self, event_type: str, *, round_number: int, runtime_id="", stage="execution",
               identity=None, **fields) -> str:
        if event_type not in EVENT_TYPES:
            raise ValueError(f"Unknown flow event: {event_type}")
        key = [self.attempt_id, round_number, runtime_id, event_type,
               identity if identity is not None else len(self.events)]
        eid = "flow-" + hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:32]
        if eid in self._keys:
            return eid
        row = asdict(FactFlowEvent(eid, self.attempt_id, round_number, runtime_id,
                                  stage, event_type, **fields))
        self.events.append(row)
        self._keys.add(eid)
        if not self._writable:
            return eid
        try:
            with (self.path / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as exc:
            self.errors.append({"operation": "append", "event_id": eid, "error": str(exc)})
        return eid

    def transfer(self, request, snapshot: dict, round_number: int, stage: str):
        known = {f["id"] for f in snapshot.get("facts", [])}
        for fid in sorted(set(FACT_TOKEN.findall(request.instruction)) & known):
            self.record("TRANSFERRED", round_number=round_number, runtime_id=request.run_id,
                        stage=stage, fact_id=fid, source_job_id=request.parent_job_id,
                        target_job_id=request.job_id, role=request.role,
                        revision=snapshot["revision"], identity=[request.job_id, fid],
                        details={"source_actor": request.parent_job_id or "orchestration"})

    def freeze(self, through_round: int) -> dict:
        events = [e for e in self.events if e["round"] <= through_round and e["stage"] != "report"]
        return copy.deepcopy({"schema_version": "fact_flow_v1", "attempt_id": self.attempt_id,
                              "through_round": through_round, "events": events,
                              "coverage_warnings": self.errors})

    def persist_snapshot(self, round_dir: Path, through_round: int):
        try:
            target = Path(round_dir) / "fact_flow"
            target.mkdir(parents=True, exist_ok=True)
            (target / "snapshot.json").write_text(json.dumps(self.freeze(through_round),
                                                            ensure_ascii=False, indent=2))
        except OSError as exc:
            self.errors.append({"operation": "snapshot", "error": str(exc)})
