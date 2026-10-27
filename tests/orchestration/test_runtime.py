"""Behavioral tests for orchestration-script primitives."""

from __future__ import annotations

import asyncio
import unittest

from agent_harness.orchestration import (
    IdempotencyConflict,
    InvalidJobRequest,
    JobCancelled,
    JobFailed,
    JobLimitExceeded,
    JobStatus,
    RunLimits,
    RunTimedOut,
    ScriptRuntime,
    UnknownRole,
)
from agent_harness.orchestration.testing import FakeWorkerExecutor


class ScriptRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_run_requires_string_task(self) -> None:
        runtime = ScriptRuntime(FakeWorkerExecutor(allowed_roles={"solver"}))

        async def script(ctx):
            return ctx.task

        with self.assertRaisesRegex(TypeError, "task must be a string"):
            await runtime.run(script, task={"question": "not a string"})

    async def test_solver_verifier_motivation_flow(self) -> None:
        async def handler(request):
            if request.role == "solver":
                return f"solution for {request.instruction}"
            if request.role == "verifier":
                return {"accepted": True, "critique": ""}
            if request.role == "synthesizer":
                return f"synthesis of {request.instruction}"
            raise AssertionError(f"unexpected role {request.role}")

        executor = FakeWorkerExecutor(
            default_handler=handler,
            allowed_roles={
                "solver",
                "verifier",
                "synthesizer",
            },
        )
        runtime = ScriptRuntime(
            executor,
            limits=RunLimits(max_parallel_jobs=2),
        )

        async def script(ctx):
            problems = ["subproblem A", "subproblem B"]

            async def solve_one(problem):
                solution = await ctx.call(
                    "solver",
                    f"Solve:\n{problem}",
                )
                verdict = await ctx.call(
                    "verifier",
                    f"Problem:\n{problem}\n\nSolution:\n{solution}",
                )
                self.assertTrue(verdict["accepted"])
                return solution

            solutions = await ctx.map(solve_one, problems)
            return await ctx.call(
                "synthesizer",
                f"Task:\n{ctx.task}\n\nSolutions:\n{solutions}",
            )

        run = await runtime.run(
            script,
            task="Answer the original task",
            run_id="motivation",
        )

        self.assertIn("solution for", run.value)
        self.assertEqual(
            [request.role for request in executor.requests],
            [
                "solver",
                "solver",
                "verifier",
                "verifier",
                "synthesizer",
            ],
        )

    async def test_call_returns_plain_value_and_assigns_automatic_key(
        self,
    ) -> None:
        runtime = ScriptRuntime(
            FakeWorkerExecutor(allowed_roles={"solver"}),
        )

        async def script(ctx):
            return await ctx.call("solver", "Solve this")

        run = await runtime.run(script, task="test", run_id="call")

        self.assertEqual(run.value, "Solve this")
        self.assertEqual(run.jobs[0].key, "auto:1")

    async def test_call_forwards_max_turns_to_job_and_snapshot(self) -> None:
        executor = FakeWorkerExecutor(allowed_roles={"researcher"})
        runtime = ScriptRuntime(executor)

        async def script(ctx):
            return await ctx.call(
                "researcher",
                "Investigate this",
                max_turns=7,
            )

        run = await runtime.run(script, task="test", run_id="call-budget")

        self.assertEqual(run.value, "Investigate this")
        self.assertEqual(executor.requests[0].max_turns, 7)
        self.assertEqual(run.jobs[0].max_turns, 7)
        created = next(
            event
            for event in run.events
            if event.event_type.value == "job_created"
        )
        self.assertEqual(created.payload["max_turns"], 7)

    async def test_max_turns_must_be_a_positive_integer(self) -> None:
        runtime = ScriptRuntime(FakeWorkerExecutor())

        for index, value in enumerate((0, -1, 1.5, True)):
            async def script(ctx, max_turns=value):
                return await ctx.call(
                    "researcher",
                    "Investigate this",
                    max_turns=max_turns,
                )

            with self.subTest(max_turns=value):
                with self.assertRaises(InvalidJobRequest):
                    await runtime.run(
                        script,
                        task="test",
                        run_id=f"invalid-budget-{index}",
                    )

    async def test_call_requires_a_string_instruction(self) -> None:
        runtime = ScriptRuntime(FakeWorkerExecutor())

        async def script(ctx):
            return await ctx.call("solver", {"not": "a string"})

        with self.assertRaises(InvalidJobRequest):
            await runtime.run(script, task="test", run_id="string-only")

    async def test_unknown_role_fails_before_creating_a_job(self) -> None:
        runtime = ScriptRuntime(
            FakeWorkerExecutor(allowed_roles={"solver"}),
        )

        async def script(ctx):
            return await ctx.call("invented", "Do work")

        with self.assertRaises(UnknownRole):
            await runtime.run(script, task="test", run_id="unknown-role")

        self.assertEqual(runtime.get_jobs("unknown-role"), ())

    async def test_spawn_runs_jobs_concurrently_and_gather_preserves_order(
        self,
    ) -> None:
        release = asyncio.Event()

        async def handler(request):
            await release.wait()
            return request.instruction

        executor = FakeWorkerExecutor(default_handler=handler)
        runtime = ScriptRuntime(
            executor,
            limits=RunLimits(max_parallel_jobs=2),
        )

        async def script(ctx):
            slow = ctx.spawn("worker", "first", key="first")
            fast = ctx.spawn("worker", "second", key="second")
            await asyncio.sleep(0)
            self.assertEqual(executor.active_jobs, 2)
            release.set()
            return await ctx.gather(slow, fast)

        run = await runtime.run(script, task="test", run_id="concurrent")

        self.assertEqual(run.value, ["first", "second"])
        self.assertEqual(executor.max_active_jobs, 2)
        self.assertTrue(all(job.status == JobStatus.SUCCEEDED for job in run.jobs))

    async def test_awaiting_failed_job_raises_typed_error(self) -> None:
        async def fail(_request):
            raise ValueError("worker exploded")

        runtime = ScriptRuntime(
            FakeWorkerExecutor({"broken": fail}),
        )

        async def script(ctx):
            handle = ctx.spawn("broken", "fail", key="broken")
            return await handle

        with self.assertRaises(JobFailed) as raised:
            await runtime.run(script, task="test", run_id="failure")

        self.assertEqual(raised.exception.failure.error_type, "ValueError")
        jobs = runtime.get_jobs("failure")
        self.assertEqual(jobs[0].status, JobStatus.FAILED)

    async def test_collect_policy_returns_partial_successes(self) -> None:
        async def handler(request):
            if request.role == "broken":
                raise RuntimeError("no result")
            return request.role

        runtime = ScriptRuntime(
            FakeWorkerExecutor(default_handler=handler),
        )

        async def script(ctx):
            good = ctx.spawn("good", "succeed", key="good")
            bad = ctx.spawn("broken", "fail", key="bad")
            return await ctx.gather(
                good,
                bad,
                failure_policy="collect",
            )

        run = await runtime.run(script, task="test", run_id="collect")

        self.assertTrue(run.value[0].succeeded)
        self.assertEqual(run.value[0].result.value, "good")
        self.assertFalse(run.value[1].succeeded)
        self.assertEqual(run.value[1].failure.error_type, "RuntimeError")

    async def test_fail_fast_cancels_another_running_job(self) -> None:
        slow_started = asyncio.Event()

        async def handler(request):
            if request.role == "slow":
                slow_started.set()
                await asyncio.Event().wait()
            await slow_started.wait()
            raise RuntimeError("fast failure")

        executor = FakeWorkerExecutor(default_handler=handler)
        runtime = ScriptRuntime(
            executor,
            limits=RunLimits(max_parallel_jobs=2),
        )

        async def script(ctx):
            slow = ctx.spawn("slow", "wait", key="slow")
            broken = ctx.spawn("broken", "fail", key="broken")
            return await ctx.gather(slow, broken)

        with self.assertRaises(JobFailed):
            await runtime.run(script, task="test", run_id="fail-fast")

        statuses = {
            job.key: job.status
            for job in runtime.get_jobs("fail-fast")
        }
        self.assertEqual(statuses["slow"], JobStatus.CANCELLED)
        self.assertEqual(statuses["broken"], JobStatus.FAILED)

    async def test_per_job_timeout_is_a_collectable_outcome(self) -> None:
        async def handler(_request):
            await asyncio.Event().wait()

        executor = FakeWorkerExecutor(default_handler=handler)
        runtime = ScriptRuntime(executor)

        async def script(ctx):
            handle = ctx.spawn(
                "worker",
                "wait",
                key="timed",
                timeout=0.01,
            )
            return await handle.outcome()

        run = await runtime.run(script, task="test", run_id="job-timeout")

        self.assertEqual(run.value.status, JobStatus.TIMED_OUT)
        self.assertEqual(run.jobs[0].status, JobStatus.TIMED_OUT)
        self.assertEqual(len(executor.cancelled_job_ids), 1)

    async def test_whole_run_timeout_cleans_up_workers(self) -> None:
        async def handler(_request):
            await asyncio.Event().wait()

        executor = FakeWorkerExecutor(default_handler=handler)
        runtime = ScriptRuntime(
            executor,
            limits=RunLimits(wall_time_seconds=0.01),
        )

        async def script(ctx):
            return await ctx.spawn("worker", "wait", key="long")

        with self.assertRaises(RunTimedOut):
            await runtime.run(script, task="test", run_id="run-timeout")

        self.assertEqual(
            runtime.get_jobs("run-timeout")[0].status,
            JobStatus.CANCELLED,
        )
        self.assertEqual(
            runtime.get_events("run-timeout")[-1].event_type.value,
            "run_timed_out",
        )

    async def test_concurrency_limit_is_enforced(self) -> None:
        async def handler(request):
            await asyncio.sleep(0.01)
            return request.key

        executor = FakeWorkerExecutor(default_handler=handler)
        runtime = ScriptRuntime(
            executor,
            limits=RunLimits(max_parallel_jobs=2, max_total_jobs=10),
        )

        async def script(ctx):
            handles = [
                ctx.spawn("worker", str(index), key=f"job-{index}")
                for index in range(6)
            ]
            return await ctx.gather(*handles)

        await runtime.run(script, task="test", run_id="limit")

        self.assertEqual(executor.max_active_jobs, 2)

    async def test_map_runs_orchestration_branches_and_preserves_order(
        self,
    ) -> None:
        async def handler(request):
            await asyncio.sleep(0.01)
            return request.instruction

        executor = FakeWorkerExecutor(
            default_handler=handler,
            allowed_roles={"solver"},
        )
        runtime = ScriptRuntime(
            executor,
            limits=RunLimits(max_parallel_jobs=2),
        )

        async def script(ctx):
            async def solve_one(problem):
                value = await ctx.call("solver", f"Solve {problem}")
                return value.upper()

            return await ctx.map(solve_one, ["a", "b", "c", "d"])

        run = await runtime.run(script, task="test", run_id="map")

        self.assertEqual(
            run.value,
            ["SOLVE A", "SOLVE B", "SOLVE C", "SOLVE D"],
        )
        self.assertEqual(executor.max_active_jobs, 2)
        event_types = [event.event_type.value for event in run.events]
        self.assertEqual(event_types.count("branch_started"), 4)
        self.assertEqual(event_types.count("branch_succeeded"), 4)

    async def test_parallel_cancels_sibling_branch_on_failure(self) -> None:
        sibling_started = asyncio.Event()
        sibling_cancelled = asyncio.Event()

        async def sibling():
            sibling_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                sibling_cancelled.set()

        async def broken():
            await sibling_started.wait()
            raise RuntimeError("branch failed")

        runtime = ScriptRuntime(FakeWorkerExecutor())

        async def script(ctx):
            return await ctx.parallel(sibling(), broken())

        with self.assertRaises(RuntimeError):
            await runtime.run(script, task="test", run_id="parallel-failure")

        self.assertTrue(sibling_cancelled.is_set())
        event_types = [
            event.event_type.value
            for event in runtime.get_events("parallel-failure")
        ]
        self.assertIn("branch_failed", event_types)
        self.assertIn("branch_cancelled", event_types)

    async def test_nested_maps_keep_branch_job_ownership_isolated(self) -> None:
        async def handler(request):
            await asyncio.sleep(0.005)
            return request.instruction

        runtime = ScriptRuntime(
            FakeWorkerExecutor(default_handler=handler),
            limits=RunLimits(max_parallel_jobs=2),
        )

        async def script(ctx):
            async def outer(prefix):
                async def inner(suffix):
                    return await ctx.call("worker", f"{prefix}{suffix}")

                return await ctx.map(inner, ["1", "2"])

            return await ctx.map(outer, ["a", "b"])

        run = await runtime.run(script, task="test", run_id="nested-map")

        self.assertEqual(run.value, [["a1", "a2"], ["b1", "b2"]])
        self.assertTrue(all(job.branch_path for job in run.jobs))
        self.assertTrue(all(job.status == JobStatus.SUCCEEDED for job in run.jobs))

    async def test_caught_parallel_failure_does_not_leak_worker_jobs(self) -> None:
        worker_started = asyncio.Event()

        async def handler(_request):
            worker_started.set()
            await asyncio.Event().wait()

        executor = FakeWorkerExecutor(default_handler=handler)
        runtime = ScriptRuntime(
            executor,
            limits=RunLimits(max_parallel_jobs=2),
        )

        async def script(ctx):
            async def worker_branch():
                return await ctx.call("worker", "wait")

            async def broken_branch():
                await worker_started.wait()
                raise RuntimeError("stop branches")

            try:
                await ctx.parallel(worker_branch(), broken_branch())
            except RuntimeError:
                return "recovered"

        run = await runtime.run(
            script,
            task="test",
            run_id="parallel-worker-cleanup",
        )

        self.assertEqual(run.value, "recovered")
        self.assertEqual(run.jobs[0].status, JobStatus.CANCELLED)
        self.assertEqual(len(executor.cancelled_job_ids), 1)

    async def test_total_job_limit_is_enforced(self) -> None:
        runtime = ScriptRuntime(
            FakeWorkerExecutor(),
            limits=RunLimits(max_total_jobs=1),
        )

        async def script(ctx):
            ctx.spawn("worker", "first", key="first")
            ctx.spawn("worker", "second", key="second")

        with self.assertRaises(JobLimitExceeded):
            await runtime.run(script, task="test", run_id="job-limit")

        jobs = runtime.get_jobs("job-limit")
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].status, JobStatus.CANCELLED)

    async def test_inspect_is_non_blocking_and_returns_a_snapshot(self) -> None:
        started = asyncio.Event()
        release = asyncio.Event()

        async def handler(_request):
            started.set()
            await release.wait()
            return "done"

        runtime = ScriptRuntime(
            FakeWorkerExecutor(default_handler=handler),
        )

        async def script(ctx):
            handle = ctx.spawn("worker", "observe", key="observed")
            await started.wait()
            snapshot = ctx.inspect(handle)
            self.assertEqual(snapshot.status, JobStatus.RUNNING)
            self.assertEqual(snapshot.key, "observed")
            release.set()
            return await handle

        run = await runtime.run(script, task="test", run_id="inspect")

        self.assertEqual(run.value, "done")
        self.assertEqual(run.jobs[0].status, JobStatus.SUCCEEDED)

    async def test_cancel_queued_job_prevents_worker_execution(self) -> None:
        first_started = asyncio.Event()
        release_first = asyncio.Event()

        async def handler(request):
            if request.key == "first":
                first_started.set()
                await release_first.wait()
            return request.key

        executor = FakeWorkerExecutor(default_handler=handler)
        runtime = ScriptRuntime(
            executor,
            limits=RunLimits(max_parallel_jobs=1),
        )

        async def script(ctx):
            first = ctx.spawn("worker", "first", key="first")
            second = ctx.spawn("worker", "second", key="second")
            await first_started.wait()
            await ctx.cancel(second, reason="not needed")
            release_first.set()
            await first
            with self.assertRaises(JobCancelled):
                await second

        run = await runtime.run(script, task="test", run_id="cancel-queued")

        self.assertEqual([request.key for request in executor.requests], ["first"])
        statuses = {job.key: job.status for job in run.jobs}
        self.assertEqual(statuses["first"], JobStatus.SUCCEEDED)
        self.assertEqual(statuses["second"], JobStatus.CANCELLED)

    async def test_identical_job_key_reuses_one_execution(self) -> None:
        executor = FakeWorkerExecutor()
        runtime = ScriptRuntime(executor)

        async def script(ctx):
            first = ctx.spawn("worker", "same", key="stable")
            second = ctx.spawn("worker", "same", key="stable")
            self.assertEqual(first.job_id, second.job_id)
            return await ctx.gather(first, second)

        run = await runtime.run(script, task="test", run_id="reuse")

        self.assertEqual(len(executor.requests), 1)
        self.assertEqual(len(run.jobs), 1)
        self.assertEqual(run.jobs[0].reuse_count, 1)

    async def test_conflicting_job_key_is_rejected(self) -> None:
        runtime = ScriptRuntime(FakeWorkerExecutor())

        async def script(ctx):
            ctx.spawn("worker", "first", key="stable")
            ctx.spawn("worker", "second", key="stable")

        with self.assertRaises(IdempotencyConflict):
            await runtime.run(script, task="test", run_id="conflict")

        jobs = runtime.get_jobs("conflict")
        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0].status, JobStatus.CANCELLED)

    async def test_job_key_rejects_a_different_max_turns_budget(self) -> None:
        runtime = ScriptRuntime(FakeWorkerExecutor())

        async def script(ctx):
            ctx.spawn(
                "worker",
                "same",
                key="stable",
                max_turns=2,
            )
            ctx.spawn(
                "worker",
                "same",
                key="stable",
                max_turns=3,
            )

        with self.assertRaises(IdempotencyConflict):
            await runtime.run(
                script,
                task="test",
                run_id="budget-conflict",
            )

    async def test_script_exception_cancels_outstanding_jobs(self) -> None:
        started = asyncio.Event()

        async def handler(_request):
            started.set()
            await asyncio.Event().wait()

        executor = FakeWorkerExecutor(default_handler=handler)
        runtime = ScriptRuntime(executor)

        async def script(ctx):
            ctx.spawn("worker", "wait", key="long-running")
            await started.wait()
            raise LookupError("script failed")

        with self.assertRaises(LookupError):
            await runtime.run(script, task="test", run_id="script-error")

        jobs = runtime.get_jobs("script-error")
        self.assertEqual(jobs[0].status, JobStatus.CANCELLED)
        self.assertEqual(len(executor.cancelled_job_ids), 1)

    async def test_lifecycle_events_have_monotonic_sequence_numbers(self) -> None:
        runtime = ScriptRuntime(FakeWorkerExecutor())

        async def script(ctx):
            return await ctx.spawn("worker", "answer", key="answer")

        run = await runtime.run(script, task="test", run_id="events")

        self.assertEqual(
            [event.sequence for event in run.events],
            list(range(1, len(run.events) + 1)),
        )
        event_types = [event.event_type.value for event in run.events]
        self.assertEqual(event_types[0], "run_started")
        self.assertIn("job_created", event_types)
        self.assertIn("job_started", event_types)
        self.assertIn("job_succeeded", event_types)
        self.assertEqual(event_types[-1], "run_succeeded")

    async def test_direct_tool_is_traced_reused_and_budgeted(self) -> None:
        executor = FakeWorkerExecutor(
            tool_handlers={"lookup": lambda request: request["arguments"]["q"]},
        )
        runtime = ScriptRuntime(executor)

        async def script(ctx):
            first = await ctx.tool(
                "lookup", {"q": "answer"}, role="worker", key="lookup:1",
            )
            reused = await ctx.tool(
                "lookup", {"q": "answer"}, role="worker", key="lookup:1",
            )
            with self.assertRaises(JobLimitExceeded):
                await ctx.tool(
                    "lookup", {"q": "second"}, role="worker", key="lookup:2",
                )
            self.assertEqual(ctx.usage()["tool_calls"], 1)
            return first, reused

        result = await runtime.run(
            script, task="test", run_id="direct-tool",
            metadata={"budget": {"max_tool_calls": 1}},
        )
        self.assertEqual(result.value, ("answer", "answer"))
        self.assertEqual(len(executor.tool_requests), 1)
        event_types = [event.event_type.value for event in result.events]
        self.assertEqual(event_types.count("tool_call_started"), 1)
        self.assertEqual(event_types.count("tool_call_succeeded"), 1)


if __name__ == "__main__":
    unittest.main()
