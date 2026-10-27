"""In-memory job ledger and append-only trace store."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any
from pathlib import Path

from agent_harness.orchestration.events import (
    OrchestrationEvent,
    OrchestrationEventType,
)
from agent_harness.orchestration.types import (
    JobOutcome,
    JobRequest,
    JobSnapshot,
    JobStatus,
)


@dataclass
class JobRecord:
    """Mutable runtime-owned job state.

    Scripts only receive immutable :class:`JobSnapshot` instances.
    """

    request: JobRequest
    status: JobStatus
    created_at: float
    completion: asyncio.Future[JobOutcome]
    started_at: float | None = None
    finished_at: float | None = None
    outcome: JobOutcome | None = None
    task: asyncio.Task[None] | None = None
    cancellation_requested: bool = False
    executor_cancel_requested: bool = False
    reuse_count: int = 0


class InMemoryOrchestrationStore:
    """Per-run state store used by the first runtime implementation."""

    def __init__(
        self,
        run_id: str,
        *,
        live_trace_path: str | Path | None = None,
    ) -> None:
        self.run_id = run_id
        self._records_by_id: dict[str, JobRecord] = {}
        self._records_by_key: dict[str, JobRecord] = {}
        self._events: list[OrchestrationEvent] = []
        self._live_trace_path = (
            Path(live_trace_path) if live_trace_path is not None else None
        )

    @property
    def job_count(self) -> int:
        return len(self._records_by_id)

    @property
    def events(self) -> tuple[OrchestrationEvent, ...]:
        return tuple(self._events)

    def add(self, record: JobRecord) -> None:
        self._records_by_id[record.request.job_id] = record
        self._records_by_key[record.request.key] = record

    def get(self, job_id: str) -> JobRecord:
        return self._records_by_id[job_id]

    def get_by_key(self, key: str) -> JobRecord | None:
        return self._records_by_key.get(key)

    def records(self) -> tuple[JobRecord, ...]:
        return tuple(self._records_by_id.values())

    def append_event(
        self,
        event_type: OrchestrationEventType,
        *,
        timestamp: float | None = None,
        job_id: str | None = None,
        key: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> OrchestrationEvent:
        event = OrchestrationEvent(
            sequence=len(self._events) + 1,
            event_type=event_type,
            run_id=self.run_id,
            timestamp=timestamp if timestamp is not None else time.time(),
            job_id=job_id,
            key=key,
            payload=dict(payload or {}),
        )
        self._events.append(event)
        if self._live_trace_path is not None:
            descriptions = {
                "run_started": "编排运行开始",
                "run_succeeded": "编排运行成功结束",
                "run_failed": "编排运行失败",
                "run_timed_out": "编排运行超时",
                "job_created": "创建Agent任务",
                "job_reused": "复用幂等Agent任务",
                "job_started": "Agent任务开始执行",
                "job_succeeded": "Agent任务执行成功",
                "job_failed": "Agent任务执行失败",
                "job_timed_out": "Agent任务执行超时",
                "job_cancel_requested": "请求取消Agent任务",
                "job_cancelled": "Agent任务已取消",
                "branch_started": "并行分支开始",
                "branch_succeeded": "并行分支成功",
                "branch_failed": "并行分支失败",
                "branch_cancelled": "并行分支取消",
                "worker_event": "Agent内部生命周期事件",
                "script_event": "编排脚本语义事件",
                "checkpoint_saved": "编排检查点已保存",
                "run_resumed": "从编排检查点恢复",
                "tool_call_started": "编排脚本直接调用工具",
                "tool_call_succeeded": "编排脚本工具调用成功",
                "tool_call_failed": "编排脚本工具调用失败",
            }
            self._live_trace_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "sequence": event.sequence,
                "事件类型": event.event_type.value,
                "事件": descriptions.get(
                    event.event_type.value, event.event_type.value,
                ),
                "run_id": event.run_id,
                "timestamp": event.timestamp,
                "job_id": event.job_id,
                "key": event.key,
                "详细信息": event.payload,
            }
            with self._live_trace_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
        return event

    def snapshot(self, record: JobRecord, *, now: float | None = None) -> JobSnapshot:
        current = now if now is not None else time.time()
        end = record.finished_at if record.finished_at is not None else current
        start = (
            record.started_at
            if record.started_at is not None
            else record.created_at
        )
        return JobSnapshot(
            job_id=record.request.job_id,
            run_id=record.request.run_id,
            key=record.request.key,
            role=record.request.role,
            status=record.status,
            created_at=record.created_at,
            started_at=record.started_at,
            finished_at=record.finished_at,
            elapsed_seconds=max(0.0, end - start),
            request_digest=record.request.request_digest,
            timeout=record.request.timeout,
            max_turns=record.request.max_turns,
            metadata=dict(record.request.metadata),
            parent_job_id=record.request.parent_job_id,
            depth=record.request.depth,
            branch_path=record.request.branch_path,
            reuse_count=record.reuse_count,
        )

    def snapshots(self) -> tuple[JobSnapshot, ...]:
        now = time.time()
        return tuple(self.snapshot(record, now=now) for record in self.records())
