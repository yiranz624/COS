"""Public primitive surface presented to orchestration scripts."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable
from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

from agent_harness.orchestration.handles import JobHandle
from agent_harness.orchestration.types import (
    BudgetSnapshot,
    FailurePolicy,
    JobOutcome,
    JobSnapshot,
)

if TYPE_CHECKING:
    from agent_harness.orchestration.runtime import _RunState

T = TypeVar("T")


class OrchestrationContext:
    """Controlled interface through which a script schedules worker jobs."""

    def __init__(
        self,
        state: "_RunState",
        *,
        task: str,
        metadata: dict[str, Any] | None = None,
        parent_job_id: str | None = None,
        depth: int = 0,
    ) -> None:
        self._state = state
        self.task = task
        self.metadata = dict(metadata or {})
        self.run_id = state.run_id
        self._parent_job_id = parent_job_id
        self._depth = depth

    def spawn(
        self,
        role: str,
        instruction: str,
        *,
        key: str | None = None,
        max_turns: int | None = None,
        timeout: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> JobHandle:
        """Schedule a worker job and return immediately with its handle.

        ``max_turns`` overrides the role default; ``timeout`` limits elapsed
        wall-clock time independently.
        """
        return self._state.spawn(
            role=role,
            instruction=instruction,
            key=key,
            max_turns=max_turns,
            timeout=timeout,
            metadata=metadata,
            parent_job_id=self._parent_job_id,
            depth=self._depth + 1,
        )

    async def call(
        self,
        role: str,
        instruction: str,
        *,
        key: str | None = None,
        max_turns: int | None = None,
        timeout: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        """Run one role, wait, and return its predefined plain value.

        ``max_turns`` overrides the role default; ``timeout`` limits elapsed
        wall-clock time independently.
        """
        return await self.spawn(
            role,
            instruction,
            key=key,
            max_turns=max_turns,
            timeout=timeout,
            metadata=metadata,
        )

    @overload
    async def gather(
        self,
        *handles: JobHandle,
        failure_policy: Literal["fail_fast"] = "fail_fast",
        cancel_remaining: bool = True,
    ) -> list[Any]: ...

    @overload
    async def gather(
        self,
        *handles: JobHandle,
        failure_policy: Literal["collect"],
        cancel_remaining: bool = True,
    ) -> list[JobOutcome]: ...

    async def gather(
        self,
        *handles: JobHandle,
        failure_policy: FailurePolicy = "fail_fast",
        cancel_remaining: bool = True,
    ) -> list[Any] | list[JobOutcome]:
        """Wait for jobs concurrently, preserving the input handle order."""
        return await self._state.gather(
            handles,
            failure_policy=failure_policy,
            cancel_remaining=cancel_remaining,
        )

    async def parallel(self, *branches: Awaitable[T]) -> list[T]:
        """Run orchestration branches concurrently with fail-fast cleanup."""
        return await self._state.parallel(branches)

    async def map(
        self,
        function: Callable[[T], Awaitable[Any]],
        items: Iterable[T],
    ) -> list[Any]:
        """Apply an async orchestration function concurrently in input order."""
        return await self._state.map(function, items)

    def inspect(self, handle: JobHandle) -> JobSnapshot:
        """Return an immutable, non-blocking snapshot of a job."""
        return self._state.inspect(handle)

    def usage(self) -> dict[str, int]:
        """Return provider-reported usage for completed jobs in this run."""
        return self._state.usage()

    def remaining_budget(self) -> BudgetSnapshot:
        """Return an immutable view of usage and remaining hard limits."""
        return self._state.remaining_budget()

    async def checkpoint(self, name: str, state: Any) -> None:
        """Atomically persist JSON-compatible script state at a safe boundary."""
        await self._state.checkpoint(name, state)

    def load_checkpoint(self, name: str) -> Any | None:
        """Load the last saved value for ``name``, if the run was resumed."""
        return self._state.load_checkpoint(name)

    def emit(self, event_type: str, **payload: Any) -> None:
        """Append a script-semantic event to the run trace."""
        self._state.emit_script_event(event_type, payload)

    async def facts_index(self, *, keyword: str = "", statuses: list[str] | None = None,
                          rounds: list[int] | None = None) -> dict[str, Any]:
        """Read the summary directory, including disputes by default."""
        return await self._state.execute_tool("facts_index", {
            "keyword": keyword, "statuses": statuses, "rounds": rounds,
        }, role="decomposer", key=None)

    async def facts_query(
        self, *, keyword: str = "", fact_ids: list[str] | None = None,
        statuses: list[str] | None = None,
        rounds: list[int] | None = None,
    ) -> dict[str, Any]:
        """Read attempt-local facts; available only with an evidence-enabled executor."""
        return await self._state.execute_tool("facts_query", {
            "keyword": keyword, "fact_ids": fact_ids,
            "statuses": statuses, "rounds": rounds,
        }, role="decomposer", key=None)

    async def tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        role: str,
        key: str | None = None,
    ) -> Any:
        """Execute one permission-checked tool with budget and Trace tracking."""
        return await self._state.execute_tool(
            tool_name, arguments, role=role, key=key,
        )

    async def cancel(
        self,
        handle: JobHandle,
        *,
        reason: str = "cancelled by orchestration script",
    ) -> None:
        """Request cancellation and wait for the job to terminate."""
        await self._state.cancel_job(handle, reason=reason)

    async def cancel_all(
        self,
        handles: Iterable[JobHandle],
        *,
        reason: str = "cancelled by orchestration script",
    ) -> None:
        """Cancel a collection of handles and wait for their cleanup."""
        await self._state.cancel_handles(handles, reason=reason)


__all__ = ["OrchestrationContext"]
