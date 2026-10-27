"""Immutable lifecycle events emitted by the orchestration runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class OrchestrationEventType(str, Enum):
    RUN_STARTED = "run_started"
    RUN_SUCCEEDED = "run_succeeded"
    RUN_FAILED = "run_failed"
    RUN_TIMED_OUT = "run_timed_out"
    JOB_CREATED = "job_created"
    JOB_REUSED = "job_reused"
    JOB_STARTED = "job_started"
    JOB_SUCCEEDED = "job_succeeded"
    JOB_FAILED = "job_failed"
    JOB_TIMED_OUT = "job_timed_out"
    JOB_CANCEL_REQUESTED = "job_cancel_requested"
    JOB_CANCELLED = "job_cancelled"
    BRANCH_STARTED = "branch_started"
    BRANCH_SUCCEEDED = "branch_succeeded"
    BRANCH_FAILED = "branch_failed"
    BRANCH_CANCELLED = "branch_cancelled"
    WORKER_EVENT = "worker_event"
    SCRIPT_EVENT = "script_event"
    CHECKPOINT_SAVED = "checkpoint_saved"
    RUN_RESUMED = "run_resumed"
    TOOL_CALL_STARTED = "tool_call_started"
    TOOL_CALL_SUCCEEDED = "tool_call_succeeded"
    TOOL_CALL_FAILED = "tool_call_failed"


@dataclass(frozen=True)
class OrchestrationEvent:
    """One append-only trace event."""

    sequence: int
    event_type: OrchestrationEventType
    run_id: str
    timestamp: float
    job_id: str | None = None
    key: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
