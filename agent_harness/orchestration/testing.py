"""Deterministic worker executor utilities for runtime tests and examples."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from agent_harness.orchestration.executor import WorkerEventEmitter
from agent_harness.orchestration.types import AgentResult, JobRequest

FakeHandler = Callable[[JobRequest], Any]


class FakeWorkerExecutor:
    """In-process executor with observable scheduling and cancellation state."""

    def __init__(
        self,
        handlers: dict[str, FakeHandler] | None = None,
        *,
        default_handler: FakeHandler | None = None,
        allowed_roles: set[str] | None = None,
        tool_handlers: dict[str, FakeHandler] | None = None,
    ) -> None:
        self.handlers = dict(handlers or {})
        self.default_handler = default_handler
        self.allowed_roles = set(allowed_roles) if allowed_roles is not None else None
        self.tool_handlers = dict(tool_handlers or {})
        self.tool_requests: list[dict[str, Any]] = []
        self.requests: list[JobRequest] = []
        self.started_job_ids: list[str] = []
        self.finished_job_ids: list[str] = []
        self.cancelled_job_ids: list[str] = []
        self.active_jobs = 0
        self.max_active_jobs = 0

    def validate_role(self, role: str) -> None:
        if self.allowed_roles is not None and role not in self.allowed_roles:
            from agent_harness.orchestration.exceptions import UnknownRole
            raise UnknownRole(
                f"unknown role {role!r}; available: {sorted(self.allowed_roles)}",
            )

    async def execute(
        self,
        request: JobRequest,
        emit: WorkerEventEmitter,
    ) -> AgentResult:
        self.requests.append(request)
        self.started_job_ids.append(request.job_id)
        self.active_jobs += 1
        self.max_active_jobs = max(self.max_active_jobs, self.active_jobs)
        await emit("fake_worker_started", {"role": request.role})
        try:
            handler = self.handlers.get(request.role, self.default_handler)
            if handler is None:
                value: Any = request.instruction
            else:
                value = handler(request)
                if inspect.isawaitable(value):
                    value = await value
            return value if isinstance(value, AgentResult) else AgentResult(value=value)
        finally:
            self.active_jobs -= 1
            self.finished_job_ids.append(request.job_id)

    async def cancel(self, job_id: str) -> None:
        if job_id not in self.cancelled_job_ids:
            self.cancelled_job_ids.append(job_id)

    async def execute_tool(
        self, tool_name: str, arguments: dict[str, Any], *, role: str,
    ) -> Any:
        request = {"tool_name": tool_name, "arguments": arguments, "role": role}
        self.tool_requests.append(request)
        try:
            handler = self.tool_handlers[tool_name]
        except KeyError as exc:
            raise ValueError(f"fake tool is not configured: {tool_name}") from exc
        value = handler(request)  # type: ignore[arg-type]
        return await value if inspect.isawaitable(value) else value


__all__ = ["FakeHandler", "FakeWorkerExecutor"]
