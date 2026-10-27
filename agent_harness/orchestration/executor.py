"""Worker execution boundary used by orchestration scripts."""

from __future__ import annotations

from typing import Any, Awaitable, Callable, Protocol

from agent_harness.orchestration.types import AgentResult, JobRequest

WorkerEventEmitter = Callable[[str, dict], Awaitable[None]]


class WorkerExecutor(Protocol):
    """Execute and cancel runtime-managed worker jobs."""

    def validate_role(self, role: str) -> None:
        """Raise when ``role`` is unavailable to this executor."""

    async def execute(
        self,
        request: JobRequest,
        emit: WorkerEventEmitter,
    ) -> AgentResult:
        """Execute one worker request."""

    async def cancel(self, job_id: str) -> None:
        """Best-effort cancellation hook for an executing worker."""

    async def execute_tool(
        self, tool_name: str, arguments: dict[str, Any], *, role: str,
    ) -> Any:
        """Execute one permission-checked tool outside an Agent Loop."""
