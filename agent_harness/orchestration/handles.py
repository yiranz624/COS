"""Awaitable handles returned by ``ctx.spawn``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agent_harness.orchestration.exceptions import (
    JobCancelled,
    JobFailed,
    JobTimedOut,
)
from agent_harness.orchestration.types import (
    AgentResult,
    JobOutcome,
    JobStatus,
)

if TYPE_CHECKING:
    from agent_harness.orchestration.runtime import _RunState


class JobHandle:
    """A stable reference to one runtime-managed worker job."""

    def __init__(self, state: "_RunState", job_id: str, key: str) -> None:
        self._state = state
        self.job_id = job_id
        self.key = key

    def __repr__(self) -> str:
        return f"JobHandle(job_id={self.job_id!r}, key={self.key!r})"

    def __await__(self):
        return self.result().__await__()

    @property
    def done(self) -> bool:
        """Whether the job has reached a terminal state."""
        return self._state.inspect(self).status.is_terminal

    async def outcome(self) -> JobOutcome:
        """Wait for a terminal outcome without raising worker failures."""
        return await self._state.wait_for_outcome(self)

    async def result_details(self) -> AgentResult:
        """Wait for the structured internal worker result."""
        outcome = await self.outcome()
        if outcome.succeeded:
            if outcome.result is None:
                raise RuntimeError(
                    f"job {self.job_id!r} succeeded without an AgentResult",
                )
            return outcome.result

        failure = outcome.failure
        if failure is None:
            raise RuntimeError(
                f"job {self.job_id!r} ended without result or failure",
            )
        if outcome.status == JobStatus.TIMED_OUT:
            raise JobTimedOut(failure)
        if outcome.status == JobStatus.CANCELLED:
            raise JobCancelled(failure)
        raise JobFailed(failure)

    async def result(self) -> Any:
        """Wait for success and return the role's plain Python value."""
        return (await self.result_details()).value


__all__ = ["JobHandle"]
