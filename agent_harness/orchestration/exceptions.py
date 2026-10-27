"""Exceptions raised by the orchestration-script runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_harness.orchestration.types import JobFailure


class OrchestrationError(RuntimeError):
    """Base class for orchestration runtime errors."""


class InvalidJobRequest(OrchestrationError):
    """A script attempted to schedule an invalid job."""


class UnknownRole(InvalidJobRequest):
    """A script requested a role unavailable to its worker executor."""


class InvalidRoleOutput(OrchestrationError):
    """A role returned content that does not match its output contract."""


class IdempotencyConflict(OrchestrationError):
    """A stable job key was reused with a different request."""

    def __init__(
        self,
        key: str,
        existing_digest: str,
        requested_digest: str,
    ) -> None:
        self.key = key
        self.existing_digest = existing_digest
        self.requested_digest = requested_digest
        super().__init__(
            f"job key {key!r} already exists with a different request "
            f"({existing_digest[:12]} != {requested_digest[:12]})",
        )


class JobLimitExceeded(OrchestrationError):
    """The script attempted to exceed a runtime-enforced job limit."""


class RunTimedOut(OrchestrationError):
    """The orchestration script exceeded its wall-time budget."""


class JobFailed(OrchestrationError):
    """Awaiting a worker job surfaced a terminal failure."""

    def __init__(self, failure: "JobFailure") -> None:
        self.failure = failure
        super().__init__(
            f"job {failure.job_id!r} failed with {failure.error_type}: "
            f"{failure.message}",
        )


class JobTimedOut(JobFailed):
    """Awaiting a worker job surfaced its per-job timeout."""


class JobCancelled(JobFailed):
    """Awaiting a worker job surfaced cancellation."""
