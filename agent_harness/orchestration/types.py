"""Public data contracts for orchestration scripts and worker executors."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class JobStatus(str, Enum):
    """Lifecycle states owned by :class:`ScriptRuntime`."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in {
            JobStatus.SUCCEEDED,
            JobStatus.FAILED,
            JobStatus.TIMED_OUT,
            JobStatus.CANCELLED,
        }


@dataclass(frozen=True)
class RunLimits:
    """Hard limits enforced by the runtime rather than by script convention."""

    max_parallel_jobs: int = 8
    max_total_jobs: int = 50
    max_depth: int = 5
    wall_time_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.max_parallel_jobs <= 0:
            raise ValueError("max_parallel_jobs must be positive")
        if self.max_total_jobs <= 0:
            raise ValueError("max_total_jobs must be positive")
        if self.max_depth <= 0:
            raise ValueError("max_depth must be positive")
        if self.wall_time_seconds is not None and self.wall_time_seconds <= 0:
            raise ValueError("wall_time_seconds must be positive when set")


@dataclass(frozen=True)
class BudgetSnapshot:
    """Current run usage and the remaining configured hard limits."""

    usage: dict[str, int]
    remaining: dict[str, int | float | None]


@dataclass(frozen=True)
class JobRequest:
    """A fully resolved request passed to a :class:`WorkerExecutor`."""

    job_id: str
    run_id: str
    key: str
    role: str
    instruction: str
    request_digest: str
    timeout: float | None = None
    max_turns: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_job_id: str | None = None
    depth: int = 1
    branch_path: tuple[str, ...] = ()
    original_task: str = ""


@dataclass(frozen=True)
class AgentResult:
    """Structured result returned by a worker executor."""

    value: Any
    usage: dict[str, int | float] = field(default_factory=dict)
    artifacts: tuple[Any, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class JobFailure:
    """Serializable failure information retained by the job ledger."""

    job_id: str
    status: JobStatus
    error_type: str
    message: str


@dataclass(frozen=True)
class JobOutcome:
    """A terminal worker outcome that can be handled as data."""

    job_id: str
    status: JobStatus
    result: AgentResult | None = None
    failure: JobFailure | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == JobStatus.SUCCEEDED


@dataclass(frozen=True)
class JobSnapshot:
    """Immutable point-in-time view returned by ``ctx.inspect``."""

    job_id: str
    run_id: str
    key: str
    role: str
    status: JobStatus
    created_at: float
    started_at: float | None
    finished_at: float | None
    elapsed_seconds: float
    request_digest: str
    timeout: float | None
    max_turns: int | None
    metadata: dict[str, Any]
    parent_job_id: str | None
    depth: int
    branch_path: tuple[str, ...]
    reuse_count: int


@dataclass(frozen=True)
class ScriptRunResult:
    """Successful result of one orchestration-script invocation."""

    run_id: str
    value: Any
    jobs: tuple[JobSnapshot, ...]
    events: tuple[Any, ...]


FailurePolicy = Literal["fail_fast", "collect"]
