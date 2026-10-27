"""Stable JSONL trace projection for orchestration runs."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from agent_harness.orchestration.contracts import BudgetUsage

TRACE_SCHEMA_VERSION = "1.0"


def _json_value(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {key: _json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True)
class TraceEventV1:
    sequence: int
    event_type: str
    run_id: str
    timestamp: float
    orchestration_version: int = 1
    stage: str | None = None
    job_id: str | None = None
    parent_job_id: str | None = None
    role: str | None = None
    status: str | None = None
    budget: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)
    schema_version: str = TRACE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return _json_value(self)


@dataclass(frozen=True)
class RunTraceV1:
    run_id: str
    strategy: str
    events: tuple[TraceEventV1, ...]
    usage: BudgetUsage
    orchestration_version: int = 1
    schema_version: str = TRACE_SCHEMA_VERSION

    def write_jsonl(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as stream:
            for event in self.events:
                stream.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        return target

    def to_summary(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "strategy": self.strategy,
            "orchestration_version": self.orchestration_version,
            "event_count": len(self.events),
            "usage": asdict(self.usage),
        }


class TraceRecorder:
    """Append stage/patch events and project native runtime events."""

    def __init__(self, run_id: str, strategy: str) -> None:
        self.run_id = run_id
        self.strategy = strategy
        self.version = 1
        self._events: list[TraceEventV1] = []

    def append(
        self,
        event_type: str,
        *,
        timestamp: float,
        stage: str | None = None,
        job_id: str | None = None,
        parent_job_id: str | None = None,
        role: str | None = None,
        status: str | None = None,
        budget: dict[str, Any] | None = None,
        payload: dict[str, Any] | None = None,
    ) -> TraceEventV1:
        event = TraceEventV1(
            sequence=len(self._events) + 1,
            event_type=event_type,
            run_id=self.run_id,
            timestamp=timestamp,
            orchestration_version=self.version,
            stage=stage,
            job_id=job_id,
            parent_job_id=parent_job_id,
            role=role,
            status=status,
            budget=dict(budget or {}),
            payload=dict(payload or {}),
        )
        self._events.append(event)
        return event

    def extend_runtime_events(
        self,
        events: Iterable[Any],
        jobs: Iterable[Any],
    ) -> None:
        jobs_by_id = {job.job_id: job for job in jobs}
        for source in events:
            job = jobs_by_id.get(source.job_id)
            self.append(
                source.event_type.value,
                timestamp=source.timestamp,
                job_id=source.job_id,
                parent_job_id=getattr(job, "parent_job_id", None),
                role=getattr(job, "role", None),
                status=(
                    getattr(getattr(job, "status", None), "value", None)
                    if job is not None else None
                ),
                payload=source.payload,
            )

    @property
    def events(self) -> tuple[TraceEventV1, ...]:
        return tuple(self._events)
