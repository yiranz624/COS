"""Structured runtime for trusted, handwritten orchestration scripts."""

from __future__ import annotations

import asyncio
import contextvars
import dataclasses
import hashlib
import inspect
import json
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

from agent_harness.orchestration.context import OrchestrationContext
from agent_harness.orchestration.events import (
    OrchestrationEvent,
    OrchestrationEventType,
)
from agent_harness.orchestration.exceptions import (
    IdempotencyConflict,
    InvalidJobRequest,
    JobLimitExceeded,
    RunTimedOut,
)
from agent_harness.orchestration.executor import WorkerExecutor
from agent_harness.orchestration.handles import JobHandle
from agent_harness.orchestration.store import (
    InMemoryOrchestrationStore,
    JobRecord,
)
from agent_harness.orchestration.types import (
    AgentResult,
    BudgetSnapshot,
    FailurePolicy,
    JobFailure,
    JobOutcome,
    JobRequest,
    JobSnapshot,
    JobStatus,
    RunLimits,
    ScriptRunResult,
)

_CURRENT_TOOL_ACTOR: contextvars.ContextVar = contextvars.ContextVar("current_tool_actor", default=None)

_CURRENT_WORKER: contextvars.ContextVar = contextvars.ContextVar("current_orchestration_worker")

OrchestrationScript = Callable[[OrchestrationContext], Awaitable[Any]]

_CURRENT_BRANCH_PATH: contextvars.ContextVar[tuple[str, ...]] = (
    contextvars.ContextVar(
        "orchestration_branch_path",
        default=(),
    )
)


class ScriptRuntime:
    """Run trusted async orchestration functions under structured concurrency."""

    def __init__(
        self,
        executor: WorkerExecutor,
        *,
        limits: RunLimits | None = None,
        live_trace_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        self._executor = executor
        self._limits = limits or RunLimits()
        self._stores: dict[str, InMemoryOrchestrationStore] = {}
        self._active_run_ids: set[str] = set()
        self._live_trace_path = live_trace_path
        self._checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path is not None else None
        )

    async def run(
        self,
        script: OrchestrationScript,
        *,
        task: str,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ScriptRunResult:
        """Execute one script and retain its in-memory trace."""
        if not isinstance(task, str):
            raise TypeError("orchestration task must be a string")
        resolved_run_id = run_id or uuid.uuid4().hex
        if resolved_run_id in self._stores or resolved_run_id in self._active_run_ids:
            raise ValueError(f"run_id {resolved_run_id!r} already exists")

        store = InMemoryOrchestrationStore(
            resolved_run_id,
            live_trace_path=self._live_trace_path,
        )
        self._stores[resolved_run_id] = store
        self._active_run_ids.add(resolved_run_id)
        state = _RunState(
            run_id=resolved_run_id,
            executor=self._executor,
            limits=self._limits,
            store=store,
            checkpoint_path=self._checkpoint_path,
            metadata=dict(metadata or {}),
            original_task=task,
        )
        store.append_event(
            OrchestrationEventType.RUN_STARTED,
            payload={"metadata": dict(metadata or {})},
        )
        state.load_persisted_checkpoints()

        try:
            if self._limits.wall_time_seconds is None:
                value = await self._execute_script(
                    state, script, task, metadata,
                )
            else:
                try:
                    async with asyncio.timeout(self._limits.wall_time_seconds):
                        value = await self._execute_script(
                            state, script, task, metadata,
                        )
                except TimeoutError as exc:
                    store.append_event(
                        OrchestrationEventType.RUN_TIMED_OUT,
                        payload={
                            "wall_time_seconds": self._limits.wall_time_seconds,
                        },
                    )
                    raise RunTimedOut(
                        f"orchestration run {resolved_run_id!r} exceeded "
                        f"{self._limits.wall_time_seconds}s",
                    ) from exc
        except Exception as exc:
            if not isinstance(exc, RunTimedOut):
                store.append_event(
                    OrchestrationEventType.RUN_FAILED,
                    payload={
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
            raise
        else:
            store.append_event(OrchestrationEventType.RUN_SUCCEEDED)
            return ScriptRunResult(
                run_id=resolved_run_id,
                value=value,
                jobs=store.snapshots(),
                events=store.events,
            )
        finally:
            self._active_run_ids.discard(resolved_run_id)

    async def _execute_script(
        self,
        state: "_RunState",
        script: OrchestrationScript,
        task: str,
        metadata: dict[str, Any] | None,
    ) -> Any:
        value: Any = None
        script_error: BaseException | None = None
        try:
            async with asyncio.TaskGroup() as task_group:
                state.attach_task_group(task_group)
                ctx = OrchestrationContext(
                    state,
                    task=task,
                    metadata=metadata,
                )
                try:
                    value = await script(ctx)
                except BaseException as exc:
                    # Do not let a script-body exception escape through
                    # TaskGroup.__aexit__: Python would wrap it in an
                    # ExceptionGroup even though worker tasks are already
                    # failure-contained. Preserve the script's original error.
                    script_error = exc
                finally:
                    await state.cancel_outstanding(
                        reason="orchestration script exited",
                    )
        finally:
            state.detach_task_group()
        if script_error is not None:
            raise script_error
        return value

    def get_events(self, run_id: str) -> tuple[OrchestrationEvent, ...]:
        """Return the retained event trace for a completed or failed run."""
        return self._stores[run_id].events

    def get_jobs(self, run_id: str) -> tuple[JobSnapshot, ...]:
        """Return current snapshots for every job in a retained run."""
        return self._stores[run_id].snapshots()


class _RunState:
    """Mutable state for one invocation of :class:`ScriptRuntime`."""

    def __init__(
        self,
        *,
        run_id: str,
        executor: WorkerExecutor,
        limits: RunLimits,
        store: InMemoryOrchestrationStore,
        checkpoint_path: Path | None,
        metadata: dict[str, Any],
        original_task: str = "",
    ) -> None:
        self.original_task = original_task
        self.run_id = run_id
        self._executor = executor
        self._limits = limits
        self._store = store
        self._semaphore = asyncio.Semaphore(limits.max_parallel_jobs)
        self._task_group: asyncio.TaskGroup | None = None
        self._job_sequence = 0
        self._branch_sequence = 0
        self._checkpoint_path = checkpoint_path
        self._metadata = metadata
        self._checkpoints: dict[str, Any] = {}
        self._started_at = time.monotonic()
        self._checkpoint_lock = asyncio.Lock()
        self._tool_lock = asyncio.Lock()
        self._direct_tool_calls = 0
        self._evidence_semaphore = asyncio.Semaphore(1)
        self._tool_keys: dict[str, tuple[str, str, Any]] = {}

    def load_persisted_checkpoints(self) -> None:
        if self._checkpoint_path is None or not self._checkpoint_path.exists():
            return
        document = json.loads(self._checkpoint_path.read_text(encoding="utf-8"))
        expected_hash = self._metadata.get("script_source_hash")
        stored_hash = document.get("script_source_hash")
        if expected_hash and stored_hash != expected_hash:
            raise ValueError(
                "checkpoint source hash does not match the selected script"
            )
        checkpoints = document.get("checkpoints", {})
        if not isinstance(checkpoints, dict):
            raise ValueError("checkpoint document has invalid checkpoints")
        self._checkpoints = checkpoints
        self._store.append_event(
            OrchestrationEventType.RUN_RESUMED,
            payload={"checkpoint_path": str(self._checkpoint_path)},
        )

    async def checkpoint(self, name: str, value: Any) -> None:
        if not name or not isinstance(name, str):
            raise ValueError("checkpoint name must be a non-empty string")
        # Validate before mutating memory or disk.
        json.dumps(value, ensure_ascii=False)
        async with self._checkpoint_lock:
            self._checkpoints[name] = value
            if self._checkpoint_path is not None:
                document = {
                    "format": "agent_harness_checkpoint_v1",
                    "run_id": self.run_id,
                    "script_source_hash": self._metadata.get(
                        "script_source_hash"
                    ),
                    "checkpoints": self._checkpoints,
                }
                self._checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = self._checkpoint_path.with_suffix(
                    self._checkpoint_path.suffix + ".tmp"
                )
                temporary.write_text(
                    json.dumps(document, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                temporary.replace(self._checkpoint_path)
        self._store.append_event(
            OrchestrationEventType.CHECKPOINT_SAVED,
            payload={"name": name, "checkpoint_path": (
                str(self._checkpoint_path) if self._checkpoint_path else None
            )},
        )

    def load_checkpoint(self, name: str) -> Any | None:
        return self._checkpoints.get(name)

    def emit_script_event(
        self, event_type: str, payload: dict[str, Any]
    ) -> None:
        self._store.append_event(
            OrchestrationEventType.SCRIPT_EVENT,
            payload={"script_event": event_type, **payload},
        )

    def remaining_budget(self) -> BudgetSnapshot:
        usage = self.usage()
        elapsed = time.monotonic() - self._started_at
        budget = self._metadata.get("budget", {})

        def remaining(name: str, used: int) -> int | None:
            limit = budget.get(name)
            return None if limit is None else max(0, int(limit) - used)

        return BudgetSnapshot(
            usage=usage,
            remaining={
                "tokens": remaining("max_tokens", usage["tokens"]),
                "llm_calls": remaining(
                    "max_llm_calls", usage["llm_calls"]
                ),
                "tool_calls": remaining(
                    "max_tool_calls", usage["tool_calls"]
                ),
                "jobs": max(0, self._limits.max_total_jobs - self._store.job_count),
                "parallel_jobs": self._limits.max_parallel_jobs,
                "wall_time_seconds": (
                    None if self._limits.wall_time_seconds is None else
                    max(0.0, self._limits.wall_time_seconds - elapsed)
                ),
            },
        )

    def attach_task_group(self, task_group: asyncio.TaskGroup) -> None:
        self._task_group = task_group

    def detach_task_group(self) -> None:
        self._task_group = None

    def usage(self) -> dict[str, int]:
        """Aggregate usage; evidence-enabled runs also account for active/failed jobs."""
        if getattr(self._executor, "evidence_scope", None) is not None:
            from agent_harness.orchestration.evidence_store import evidence_usage
            return {**evidence_usage(self._store.events), "jobs": self._store.job_count}
        totals = {
            "tokens": 0,
            "llm_calls": 0,
            "tool_calls": 0,
            "jobs": 0,
        }
        for record in self._store.records():
            outcome = record.outcome
            if outcome is None or not outcome.succeeded or outcome.result is None:
                continue
            usage = outcome.result.usage
            totals["tokens"] += int(
                usage.get("total_tokens", 0)
                or usage.get("tokens", 0)
                or (
                    usage.get("prompt_tokens", usage.get("input_tokens", 0))
                    + usage.get(
                        "completion_tokens", usage.get("output_tokens", 0),
                    )
                )
            )
            metadata = outcome.result.metadata
            totals["llm_calls"] += int(metadata.get("llm_calls", 0))
            totals["tool_calls"] += int(metadata.get("tool_calls", 0))
            totals["jobs"] += 1
        totals["tool_calls"] += self._direct_tool_calls
        return totals

    async def execute_tool(
        self, tool_name: str, arguments: dict[str, Any], *, role: str,
        key: str | None,
    ) -> Any:
        """Run a direct tool call exactly once under permission and budget."""
        if not tool_name.strip() or not role.strip():
            raise ValueError("tool_name and role must be non-empty strings")
        if not isinstance(arguments, dict):
            raise TypeError("tool arguments must be an object")
        digest = hashlib.sha256(json.dumps(
            {"tool": tool_name, "role": role, "arguments": arguments},
            ensure_ascii=False, sort_keys=True,
        ).encode()).hexdigest()
        async with self._tool_lock:
            if key and key in self._tool_keys:
                stored_digest, status, value = self._tool_keys[key]
                if stored_digest != digest:
                    raise ValueError(f"tool key {key!r} reused with different input")
                if status == "succeeded":
                    return value
            limit = self._metadata.get("budget", {}).get("max_tool_calls")
            if limit is not None and self.usage()["tool_calls"] >= int(limit):
                raise JobLimitExceeded("max_tool_calls exhausted before tool call")
            self._direct_tool_calls += 1
        resolved_key = key or f"direct-tool:{self._direct_tool_calls}"
        self._store.append_event(
            OrchestrationEventType.TOOL_CALL_STARTED, key=resolved_key,
            payload={"tool_name": tool_name, "role": role,
                     "arguments": arguments},
        )
        try:
            method = getattr(self._executor, "execute_tool", None)
            if method is None:
                raise RuntimeError("worker executor does not support direct tools")
            actor_token = _CURRENT_TOOL_ACTOR.set((self, resolved_key))
            try:
                value = await method(tool_name, arguments, role=role)
            finally:
                _CURRENT_TOOL_ACTOR.reset(actor_token)
        except Exception as exc:
            self._store.append_event(
                OrchestrationEventType.TOOL_CALL_FAILED, key=resolved_key,
                payload={"tool_name": tool_name,
                         "error_type": type(exc).__name__, "message": str(exc)},
            )
            raise
        if key:
            async with self._tool_lock:
                self._tool_keys[key] = (digest, "succeeded", value)
        self._store.append_event(
            OrchestrationEventType.TOOL_CALL_SUCCEEDED, key=resolved_key,
            payload={"tool_name": tool_name, "result": value},
        )
        return value

    def spawn(
        self,
        *,
        role: str,
        instruction: str,
        key: str | None,
        max_turns: int | None,
        timeout: float | None,
        metadata: dict[str, Any] | None,
        parent_job_id: str | None,
        depth: int,
    ) -> JobHandle:
        """Create or reuse a job without yielding control."""
        if self._task_group is None:
            raise RuntimeError("orchestration runtime is not accepting jobs")
        self._ensure_usage_budget_allows_job()
        if not role or not role.strip():
            raise InvalidJobRequest("role must be a non-empty string")
        if not isinstance(instruction, str):
            raise InvalidJobRequest("instruction must be a string")
        if key is not None and (not isinstance(key, str) or not key.strip()):
            raise InvalidJobRequest("key must be a non-empty string when set")
        if max_turns is not None and (
            not isinstance(max_turns, int)
            or isinstance(max_turns, bool)
            or max_turns <= 0
        ):
            raise InvalidJobRequest(
                "max_turns must be a positive integer when set",
            )
        if timeout is not None and timeout <= 0:
            raise InvalidJobRequest("timeout must be positive when set")
        validator = getattr(self._executor, "validate_role", None)
        if validator is not None:
            validator(role)

        prospective_sequence = self._job_sequence + 1
        resolved_key = key or f"auto:{prospective_sequence}"
        if depth > self._limits.max_depth:
            raise JobLimitExceeded(
                f"job {resolved_key!r} depth {depth} exceeds max_depth "
                f"{self._limits.max_depth}",
            )

        digest = _request_digest(
            role=role,
            instruction=instruction,
            max_turns=max_turns,
            timeout=timeout,
            metadata=metadata,
            parent_job_id=parent_job_id,
        )
        existing = self._store.get_by_key(resolved_key)
        if existing is not None:
            if existing.request.request_digest != digest:
                raise IdempotencyConflict(
                    resolved_key,
                    existing.request.request_digest,
                    digest,
                )
            existing.reuse_count += 1
            self._store.append_event(
                OrchestrationEventType.JOB_REUSED,
                job_id=existing.request.job_id,
                key=resolved_key,
                payload={"reuse_count": existing.reuse_count},
            )
            return JobHandle(self, existing.request.job_id, resolved_key)

        if self._store.job_count >= self._limits.max_total_jobs:
            raise JobLimitExceeded(
                f"max_total_jobs={self._limits.max_total_jobs} exceeded",
            )

        self._job_sequence += 1
        job_id = f"{self.run_id}:job:{self._job_sequence}"
        request = JobRequest(
            job_id=job_id,
            run_id=self.run_id,
            key=resolved_key,
            role=role,
            instruction=instruction,
            request_digest=digest,
            timeout=timeout,
            max_turns=max_turns,
            metadata=dict(metadata or {}),
            parent_job_id=parent_job_id,
            depth=depth,
            branch_path=_CURRENT_BRANCH_PATH.get(),
            original_task=self.original_task,
        )
        record = JobRecord(
            request=request,
            status=JobStatus.PENDING,
            created_at=time.time(),
            completion=asyncio.get_running_loop().create_future(),
        )
        self._store.add(record)
        self._store.append_event(
            OrchestrationEventType.JOB_CREATED,
            job_id=job_id,
            key=resolved_key,
            payload={
                "role": role,
                "instruction": instruction,
                "source_job_id": parent_job_id,
                "target_job_id": job_id,
                "target_role": role,
                "information_transfer": {
                    "from": parent_job_id or "orchestration",
                    "to": job_id,
                    "target_role": role,
                    "content": instruction,
                },
                "depth": depth,
                "max_turns": max_turns,
                "key_was_automatic": key is None,
            },
        )
        record.task = self._task_group.create_task(
            self._execute_job(record),
            name=f"orchestration:{job_id}",
        )
        return JobHandle(self, job_id, resolved_key)

    def _ensure_usage_budget_allows_job(self) -> None:
        """Do not start another Job after a reported global limit is spent."""
        budget = self._metadata.get("budget", {})
        usage = self.usage()
        for usage_name, limit_name in (
            ("tokens", "max_tokens"),
            ("llm_calls", "max_llm_calls"),
            ("tool_calls", "max_tool_calls"),
        ):
            limit = budget.get(limit_name)
            if limit is not None and usage[usage_name] >= int(limit):
                raise JobLimitExceeded(
                    f"{limit_name}={limit} exhausted before starting a new job"
                )

    async def _execute_job(self, record: JobRecord) -> None:
        token = _CURRENT_WORKER.set((self, record.request))
        semaphore = self._semaphore
        if (getattr(self._executor, "evidence_scope", None) is not None
                and record.request.role == "evidence_storesubagent"):
            semaphore = self._evidence_semaphore
        try:
            async with semaphore:
                if record.status.is_terminal:
                    return
                record.status = JobStatus.RUNNING
                record.started_at = time.time()
                self._store.append_event(
                    OrchestrationEventType.JOB_STARTED,
                    job_id=record.request.job_id,
                    key=record.request.key,
                )

                async def emit(name: str, payload: dict) -> None:
                    self._store.append_event(
                        OrchestrationEventType.WORKER_EVENT,
                        job_id=record.request.job_id,
                        key=record.request.key,
                        payload={"name": name, "data": dict(payload)},
                    )

                if record.request.timeout is None:
                    raw_result = await self._executor.execute(
                        record.request, emit,
                    )
                else:
                    try:
                        async with asyncio.timeout(record.request.timeout):
                            raw_result = await self._executor.execute(
                                record.request, emit,
                            )
                    except TimeoutError:
                        await self._request_executor_cancel(record)
                        self._finalize_failure(
                            record,
                            status=JobStatus.TIMED_OUT,
                            error_type="TimeoutError",
                            message=(
                                f"job exceeded timeout of "
                                f"{record.request.timeout}s"
                            ),
                            event_type=OrchestrationEventType.JOB_TIMED_OUT,
                        )
                        return

                result = (
                    raw_result
                    if isinstance(raw_result, AgentResult)
                    else AgentResult(value=raw_result)
                )
                self._finalize_success(record, result)
        except asyncio.CancelledError:
            await self._request_executor_cancel(record)
            self._finalize_failure(
                record,
                status=JobStatus.CANCELLED,
                error_type="CancelledError",
                message="job was cancelled",
                event_type=OrchestrationEventType.JOB_CANCELLED,
            )
        except Exception as exc:
            self._finalize_failure(
                record,
                status=JobStatus.FAILED,
                error_type=type(exc).__name__,
                message=str(exc),
                event_type=OrchestrationEventType.JOB_FAILED,
            )

        finally:
            _CURRENT_WORKER.reset(token)

    def _finalize_success(
        self,
        record: JobRecord,
        result: AgentResult,
    ) -> None:
        if record.status.is_terminal:
            return
        record.status = JobStatus.SUCCEEDED
        record.finished_at = time.time()
        outcome = JobOutcome(
            job_id=record.request.job_id,
            status=JobStatus.SUCCEEDED,
            result=result,
        )
        record.outcome = outcome
        if not record.completion.done():
            record.completion.set_result(outcome)
        self._store.append_event(
            OrchestrationEventType.JOB_SUCCEEDED,
            job_id=record.request.job_id,
            key=record.request.key,
            payload={
                "usage": dict(result.usage),
                "result": result.value,
                "metadata": dict(result.metadata),
            },
        )

    def _finalize_failure(
        self,
        record: JobRecord,
        *,
        status: JobStatus,
        error_type: str,
        message: str,
        event_type: OrchestrationEventType,
    ) -> None:
        if record.status.is_terminal:
            return
        record.status = status
        record.finished_at = time.time()
        failure = JobFailure(
            job_id=record.request.job_id,
            status=status,
            error_type=error_type,
            message=message,
        )
        outcome = JobOutcome(
            job_id=record.request.job_id,
            status=status,
            failure=failure,
        )
        record.outcome = outcome
        if not record.completion.done():
            record.completion.set_result(outcome)
        self._store.append_event(
            event_type,
            job_id=record.request.job_id,
            key=record.request.key,
            payload={"error_type": error_type, "message": message},
        )

    async def wait_for_outcome(self, handle: JobHandle) -> JobOutcome:
        record = self._record_for_handle(handle)
        return await asyncio.shield(record.completion)

    async def gather(
        self,
        handles: Iterable[JobHandle],
        *,
        failure_policy: FailurePolicy,
        cancel_remaining: bool,
    ) -> list[Any] | list[JobOutcome]:
        resolved = tuple(handles)
        for handle in resolved:
            self._record_for_handle(handle)
        if failure_policy == "collect":
            return list(await asyncio.gather(
                *(handle.outcome() for handle in resolved),
            ))
        if failure_policy != "fail_fast":
            raise ValueError(
                "failure_policy must be 'fail_fast' or 'collect'",
            )

        try:
            return list(await asyncio.gather(
                *(handle.result() for handle in resolved),
            ))
        except Exception:
            if cancel_remaining:
                await self.cancel_handles(
                    resolved,
                    reason="gather failed fast",
                )
            raise

    async def parallel(
        self,
        branches: Iterable[Awaitable[Any]],
    ) -> list[Any]:
        """Run orchestration coroutines concurrently and cancel on failure."""
        resolved = tuple(branches)
        if not resolved:
            return []

        branch_limit = asyncio.Semaphore(self._limits.max_parallel_jobs)
        parent_branch_path = _CURRENT_BRANCH_PATH.get()
        branch_specs: list[tuple[str, int, Awaitable[Any]]] = []
        for index, branch in enumerate(resolved):
            self._branch_sequence += 1
            branch_specs.append((
                f"{self.run_id}:branch:{self._branch_sequence}",
                index,
                branch,
            ))

        async def run_branch(
            branch_id: str,
            index: int,
            branch: Awaitable[Any],
        ) -> Any:
            branch_was_awaited = False
            branch_path = parent_branch_path + (branch_id,)
            branch_token = _CURRENT_BRANCH_PATH.set(branch_path)
            try:
                async with branch_limit:
                    self._store.append_event(
                        OrchestrationEventType.BRANCH_STARTED,
                        payload={"branch_id": branch_id, "index": index},
                    )
                    branch_was_awaited = True
                    value = await branch
                    self._store.append_event(
                        OrchestrationEventType.BRANCH_SUCCEEDED,
                        payload={"branch_id": branch_id, "index": index},
                    )
                    return value
            except asyncio.CancelledError:
                self._store.append_event(
                    OrchestrationEventType.BRANCH_CANCELLED,
                    payload={"branch_id": branch_id, "index": index},
                )
                raise
            except Exception as exc:
                self._store.append_event(
                    OrchestrationEventType.BRANCH_FAILED,
                    payload={
                        "branch_id": branch_id,
                        "index": index,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
                raise
            finally:
                # A branch cancelled while queued on ``branch_limit`` never
                # awaits its coroutine object. Close it to avoid leaking an
                # un-awaited generated-script coroutine.
                if not branch_was_awaited and inspect.iscoroutine(branch):
                    branch.close()
                _CURRENT_BRANCH_PATH.reset(branch_token)

        tasks = [
            asyncio.create_task(
                run_branch(branch_id, index, branch),
                name=f"orchestration:{branch_id}",
            )
            for branch_id, index, branch in branch_specs
        ]
        try:
            return list(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:
            owned_paths = {
                parent_branch_path + (branch_id,)
                for branch_id, _index, _branch in branch_specs
            }
            branch_jobs = [
                JobHandle(self, record.request.job_id, record.request.key)
                for record in self._store.records()
                if (
                    any(
                        record.request.branch_path[:len(path)] == path
                        for path in owned_paths
                    )
                    and not record.status.is_terminal
                )
            ]
            if branch_jobs:
                await self.cancel_handles(
                    branch_jobs,
                    reason="parallel branch exited without awaiting its job",
                )

    async def map(
        self,
        function: Callable[[Any], Awaitable[Any]],
        items: Iterable[Any],
    ) -> list[Any]:
        """Apply an async orchestration function with structured cleanup."""

        async def invoke(item: Any) -> Any:
            return await function(item)

        return await self.parallel(invoke(item) for item in items)

    def inspect(self, handle: JobHandle) -> JobSnapshot:
        return self._store.snapshot(self._record_for_handle(handle))

    async def cancel_job(
        self,
        handle: JobHandle,
        *,
        reason: str,
    ) -> None:
        record = self._record_for_handle(handle)
        self._initiate_cancel(record, reason=reason)
        await self._finish_cancel(record)

    def _initiate_cancel(self, record: JobRecord, *, reason: str) -> None:
        """Mark and signal cancellation without yielding to worker tasks."""
        if record.status.is_terminal:
            return
        if not record.cancellation_requested:
            record.cancellation_requested = True
            self._store.append_event(
                OrchestrationEventType.JOB_CANCEL_REQUESTED,
                job_id=record.request.job_id,
                key=record.request.key,
                payload={"reason": reason},
            )

        if record.status == JobStatus.PENDING:
            self._finalize_failure(
                record,
                status=JobStatus.CANCELLED,
                error_type="CancelledError",
                message=reason,
                event_type=OrchestrationEventType.JOB_CANCELLED,
            )
            if record.task is not None:
                record.task.cancel()
        else:
            if record.task is not None:
                record.task.cancel()

    async def _finish_cancel(self, record: JobRecord) -> None:
        if not record.status.is_terminal:
            await self._request_executor_cancel(record)
        await asyncio.shield(record.completion)

    async def cancel_handles(
        self,
        handles: Iterable[JobHandle],
        *,
        reason: str,
    ) -> None:
        # Initiate every cancellation before the first await. Otherwise a
        # fast queued worker can start and finish while cancellation
        # coroutines are merely being scheduled by asyncio.gather.
        records = [
            self._record_for_handle(handle)
            for handle in handles
        ]
        for record in records:
            self._initiate_cancel(record, reason=reason)
        await asyncio.gather(
            *(self._finish_cancel(record) for record in records),
        )

    async def cancel_outstanding(self, *, reason: str) -> None:
        handles = [
            JobHandle(self, record.request.job_id, record.request.key)
            for record in self._store.records()
            if not record.status.is_terminal
        ]
        if handles:
            await self.cancel_handles(handles, reason=reason)

    async def _request_executor_cancel(self, record: JobRecord) -> None:
        if record.executor_cancel_requested:
            return
        record.executor_cancel_requested = True
        try:
            await self._executor.cancel(record.request.job_id)
        except Exception:
            # Cancellation is best-effort. The runtime task is still cancelled
            # and its state transition remains authoritative.
            return

    def _record_for_handle(self, handle: JobHandle) -> JobRecord:
        if handle._state is not self:
            raise ValueError("job handle belongs to a different run")
        return self._store.get(handle.job_id)


def _request_digest(
    *,
    role: str,
    instruction: str,
    max_turns: int | None,
    timeout: float | None,
    metadata: dict[str, Any] | None,
    parent_job_id: str | None,
) -> str:
    body = {
        "role": role,
        "instruction": instruction,
        "max_turns": max_turns,
        "timeout": timeout,
        "metadata": metadata or {},
        "parent_job_id": parent_job_id,
    }
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=repr)
    return {
        "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
        "__repr__": repr(value),
    }


__all__ = ["OrchestrationScript", "ScriptRuntime"]
