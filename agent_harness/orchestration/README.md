# Orchestration script runtime

This package is an experimental runtime for expressing orchestration policy as
ordinary async Python:

```text
task
  -> orchestration function
  -> role calls and parallel branches
  -> AgentHarness workers
  -> plain Python values
  -> final result and trace
```

The orchestration function decides which roles to call, what work to delegate,
what can run concurrently, how many solver/verifier rounds to perform, and when
to stop. Roles own their prompts, tools, model routing, and output contracts.

The runtime executes imported, trusted async functions. The benchmark's
``generated_code_orchestration`` strategy may additionally compile a per-task
``solve(ctx)`` only after restricted-AST validation and fake-context preflight.
That source receives minimal builtins and is registered only in the current
worker process. If no generated program passes validation and review, or the
program raises during execution, the task is recorded as failed, and no other
script runs in its place. Under ``rolechange_self_evolving_v6``, the same
failures end only the current round, and the next round regenerates the program
until the round cap is reached.
The runtime itself never accepts arbitrary unvalidated Python source.

## Quick start

The smallest orchestration function receives an `OrchestrationContext` and
returns any Python value:

```python
async def orchestrate(ctx):
    return await ctx.call(
        "researcher",
        f"Research this task and cite sources:\n{ctx.task}",
        max_turns=6,
        timeout=300,
    )
```

`ctx.call(...)` suspends this orchestration coroutine until the worker
finishes. It does not block the event loop: other scheduled jobs and branches
continue running.

## Context values

Every orchestration function receives `ctx` with:

- `ctx.task`: the task supplied to `ScriptRuntime.run`
- `ctx.metadata`: run-level metadata supplied to `ScriptRuntime.run`
- `ctx.run_id`: the current run identifier

### `await ctx.tool(...)`

Trusted orchestration scripts may execute a registered tool without placing
tool selection inside an Agent Loop:

```python
results = await ctx.tool(
    "web_search",
    {"q": "Brazilian referee 1990 matches", "num": 5},
    role="solver",
    key="search:referees",
)
```

The runtime checks the named role's permission, enforces `max_tool_calls`,
reuses identical keys within the run, and emits `tool_call_started`,
`tool_call_succeeded`, or `tool_call_failed`. Scripts, not models, own the
decision to call the tool. This is intended for explicit retrieval state
machines; ordinary agent work should continue to use role tools.

The second positional argument to `call` and `spawn` is always a string
instruction. Complex inputs should be rendered into that instruction.

## Primitives

### `await ctx.call(...)`

Run one predefined role, wait for it, and return its plain value.

```python
answer = await ctx.call(
    "solver",
    "Solve 17 * 23 and explain the calculation.",
    key="solve:multiplication",
    max_turns=4,
    timeout=120,
)
```

Arguments:

- `role`: registered role name
- `instruction`: delegated task as a string
- `key`: optional stable key for within-run reuse
- `max_turns`: optional positive integer overriding the role default
- `timeout`: optional wall-clock timeout in seconds

`max_turns` and `timeout` are independent. A turn budget limits agent-loop
steps; a timeout limits elapsed time, including model and tool latency. If
`max_turns` is omitted, the role default is used.

`call` is the simplest primitive and should be preferred when the script does
not need to inspect or cancel the job while it is running.

### `ctx.spawn(...)`

Schedule a worker immediately and return a `JobHandle` without waiting.

```python
facts = ctx.spawn(
    "researcher",
    "Find primary sources for the factual claims.",
    key="research:facts",
    max_turns=6,
)
counterexamples = ctx.spawn(
    "researcher",
    "Try to falsify the leading hypothesis.",
    key="research:counterexamples",
    max_turns=4,
)
```

`spawn` accepts the same policy arguments as `call`, plus optional job
`metadata`. It is the advanced primitive for explicit scheduling,
observation, cancellation, and failure handling.

### `JobHandle`

A handle returned by `spawn` is awaitable:

```python
value = await facts
```

Its public operations are:

- `await handle` or `await handle.result()`: return the plain role value or
  raise a typed job exception
- `await handle.result_details()`: return `AgentResult`, including usage,
  artifacts, and worker metadata
- `await handle.outcome()`: return `JobOutcome` without raising worker
  failures
- `handle.done`: report whether the job is terminal
- `handle.job_id` and `handle.key`: stable job identifiers

### `await ctx.gather(...)`

Wait for jobs that were already scheduled with `spawn`. Results preserve
handle order, not completion order.

```python
facts, counterexamples = await ctx.gather(facts, counterexamples)
```

The default policy is fail-fast:

```python
values = await ctx.gather(
    first,
    second,
    failure_policy="fail_fast",
    cancel_remaining=True,
)
```

When one job fails, the error is raised and remaining jobs are cancelled by
default.

Use collection mode when failure is data the script wants to inspect:

```python
outcomes = await ctx.gather(
    first,
    second,
    failure_policy="collect",
)
for outcome in outcomes:
    if outcome.succeeded:
        print(outcome.result.value)
    else:
        print(outcome.failure.message)
```

### `await ctx.parallel(...)`

Run orchestration branches concurrently and return their values in argument
order:

```python
identity, character = await ctx.parallel(
    ctx.call("researcher", "Identify the work.", max_turns=6),
    ctx.call("researcher", "Identify the character.", max_turns=6),
)
```

A branch is an awaitable orchestration operation. It may contain loops,
conditionals, and multiple nested `call` or `spawn` operations. If one branch
fails, sibling branches and the jobs they own are cancelled.

Use `parallel` for concise concurrent control flow. Use `spawn` plus `gather`
when handles, inspection, partial failure, or explicit cancellation matter.

### `await ctx.map(function, items)`

Apply an async orchestration function to every item concurrently. Result order
matches input order.

```python
async def solve_one(problem):
    return await ctx.call(
        "solver",
        f"Solve this subproblem:\n{problem}",
        max_turns=4,
    )

solutions = await ctx.map(solve_one, subproblems)
```

Each mapped function is an orchestration branch and may schedule multiple
workers.

### `ctx.inspect(handle)`

Return an immutable, non-blocking `JobSnapshot`:

```python
snapshot = ctx.inspect(facts)
print(snapshot.status, snapshot.elapsed_seconds, snapshot.max_turns)
```

Snapshots include status, timing, role, key, request digest, budget, timeout,
metadata, branch path, and reuse count.

### `await ctx.cancel(...)` and `await ctx.cancel_all(...)`

Request cancellation and wait until the runtime records a terminal state:

```python
await ctx.cancel(facts, reason="another branch found decisive evidence")
await ctx.cancel_all(handles, reason="consensus reached")
```

Cancellation is cooperative at the executor boundary. The asyncio worker task
is cancelled immediately, but hard-killing a synchronous SDK call requires
future process isolation.

## Keys and within-run reuse

Explicit keys provide idempotent reuse within one run:

```python
first = ctx.spawn("researcher", "Find source A.", key="source:a")
second = ctx.spawn("researcher", "Find source A.", key="source:a")
assert first.job_id == second.job_id
```

For every `ctx.call` or `ctx.spawn` inside a loop, encode the logical stage,
loop iteration, and stable entity identity in the key, for example
`f"verify:{round_index}:{candidate_id}"`. Different logical invocations must
not share a key. Reuse is reserved for retrying the exact same request, and
`unknown` must not be used alone as an entity identity.

Reusing a key with a different role, instruction, turn budget, timeout,
metadata, or parent produces `IdempotencyConflict`. Omitted keys receive
automatic sequence keys such as `auto:1`.

The current store is in memory, so reuse does not yet cross process restarts or
separate `run` calls.

## Built-in agent roles

`BenchmarkSession` registers these roles for orchestration scripts:

| Role | Purpose | Tools | Plain return value | Default turns |
|---|---|---|---|---:|
| `researcher` | Find and reconcile external evidence with source URLs | Web search, page fetch, Python | `str` | 10 |
| `solver` | Construct a self-contained candidate solution | Python | `str` | 8 |
| `verifier` | Independently audit one candidate against every requirement | Web search, page fetch, Python | `{"accepted": bool, "classification": str, "critique": str, ...}` | 2 |
| `synthesizer` | Combine supplied results into one coherent answer | None | `str` | 6 |
| `general` | Handle delegated work that does not fit another role | Web search, page fetch, Python | `str` | 30 |

Generated scripts do not define output schemas. The predefined role does that.
Each verifier call must contain exactly one candidate:

```python
eligible_candidates = []
rediscovery_needed = []
verdict = await ctx.call(
    "verifier",
    f"Original task:\n{ctx.task}\n\nCandidate to audit:\n{candidate}",
    key=f"verify:{candidate_id}",
)
classification = verdict["classification"]
if classification == "contradicted":
    rediscovery_needed.append(candidate_id)
elif classification in {"verified", "missing_evidence"}:
    eligible_candidates.append((candidate, verdict))
```

The verifier does not discover, name, compare, or rank candidates. Its
`classification` is exactly `verified`, `missing_evidence`, or `contradicted`.
`missing_evidence` sets `accepted` to true but does not create confirmed facts;
only `verified` may populate `confirmed_facts`. The synthesizer receives every
candidate and its independent audit, excludes only `contradicted` candidates,
and compares the remaining candidates itself.

Role tool lists are permissions, not requirements. A worker may finish without
using a tool. Actual tool availability and model routing come from the
AgentHarness runtime services; model selection is intentionally absent from
the orchestration primitive.

The AgentHarness worker adapter reserves the last configured turn for a final
answer. If a worker still exhausts the budget, it performs one bounded,
tool-free finalization over the evidence already collected. Structured roles
also receive at most one tool-free formatting repair if their final value does
not match the predefined JSON contract.

## Runtime and limits

Create a runtime with a `WorkerExecutor` and hard run limits:

```python
runtime = ScriptRuntime(
    executor,
    limits=RunLimits(
        max_parallel_jobs=4,
        max_total_jobs=32,
        max_depth=5,
        wall_time_seconds=1800,
    ),
)
```

- `max_parallel_jobs`: maximum simultaneously executing worker jobs
- `max_total_jobs`: maximum jobs created in one run
- `max_depth`: maximum nested job depth
- `wall_time_seconds`: optional timeout for the whole script; `None` (the default) means unlimited

Run a script:

```python
run = await runtime.run(
    orchestrate,
    task="Investigate the incident",
    run_id="incident-001",
    metadata={"experiment": "solver-verifier-v1"},
)

print(run.value)
print(run.jobs)
print(run.events)
```

`ScriptRunResult` contains the script return value, final job snapshots, and
the append-only event trace. `runtime.get_jobs(run_id)` and
`runtime.get_events(run_id)` also work after successful or failed runs.

The runtime enforces structured cleanup: when the script exits or raises,
outstanding jobs are cancelled before `run` returns.

## Failures

Awaiting a failed handle raises:

- `JobFailed`: worker execution failed
- `JobTimedOut`: per-job wall-clock timeout expired
- `JobCancelled`: the job was cancelled

Scheduling or run-level errors include:

- `InvalidJobRequest`: invalid role instruction or policy value
- `UnknownRole`: the requested role is not registered
- `IdempotencyConflict`: a key was reused for a different request
- `JobLimitExceeded`: a runtime job/depth limit was exceeded
- `RunTimedOut`: the whole-run wall-time limit expired

A role that still violates its predefined output contract after the bounded
repair attempt is recorded as a failed job with error type
`InvalidRoleOutput`; awaiting it raises `JobFailed`.

Use `handle.outcome()` or `gather(..., failure_policy="collect")` when these
states should be handled as data instead of exceptions.

## Complete solver-verifier example

This example assigns three task-specific solution branches, solves them
concurrently, audits each concrete candidate independently, and synthesizes the
eligible results.

```python
from __future__ import annotations

import asyncio

from dotenv import load_dotenv

from agent_harness.orchestration import (
    AgentHarnessWorkerExecutor,
    RunLimits,
    ScriptRuntime,
)
from benchmarks.core.kernel_adapter import BenchmarkSession


async def solve(ctx):
    subproblems = (
        "Discover concrete candidates using the rarest conjunctions of constraints.",
        "Independently identify candidates from structured records and primary sources.",
        "Resolve likely evidence gaps and test every required condition.",
    )

    async def solve_one(item):
        index, problem = item
        candidate = await ctx.call(
            "solver",
            f"Solve this subproblem and preserve evidence:\n{problem}",
            key=f"solve:{index}",
            max_turns=4,
            timeout=180,
        )
        return {"candidate_id": f"candidate-{index}", "answer": candidate}

    candidates = await ctx.map(solve_one, enumerate(subproblems))

    async def audit(candidate):
        return {
            "candidate_id": candidate["candidate_id"],
            "verdict": await ctx.call(
                "verifier",
                (
                    f"Original task:\n{ctx.task}\n\n"
                    "Candidate to audit (audit only this candidate):\n"
                    f"{candidate}"
                ),
                key=f"verify:{candidate['candidate_id']}",
                max_turns=2,
                timeout=300,
            ),
        }

    audits = await ctx.map(audit, candidates)
    return await ctx.call(
        "synthesizer",
        (
            f"Original task:\n{ctx.task}\n\n"
            f"Candidates and independent audits:\n{candidates}\n{audits}\n\n"
            "Exclude only contradicted candidates and compare the remaining "
            "verified or missing_evidence candidates yourself."
        ),
        key="synthesize",
        max_turns=2,
        timeout=180,
    )


async def main():
    load_dotenv()
    async with BenchmarkSession():
        runtime = ScriptRuntime(
            AgentHarnessWorkerExecutor.from_runtime_services(),
            limits=RunLimits(
                max_parallel_jobs=4,
                max_total_jobs=32,
                wall_time_seconds=1800,
            ),
        )
        result = await runtime.run(
            solve,
            task="Explain why the sky is blue at noon but red near sunset.",
            run_id="solver-verifier-example",
        )

    print(result.value)
    for job in result.jobs:
        print(job.key, job.role, job.status.value)


if __name__ == "__main__":
    asyncio.run(main())
```

The repository also contains a real research example at
`examples/real_problem_orchestration.py`.

## Current limitations

- Orchestration functions must currently be imported trusted callables.
- Generated Python source sandboxing is deferred.
- Jobs and traces are retained only in memory.
- Cancellation cannot hard-kill a synchronous SDK call without process
  isolation.
- Role/model routing is configured by AgentHarness runtime services rather
  than by the script.
