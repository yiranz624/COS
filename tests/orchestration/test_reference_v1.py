from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from agent_harness.orchestration.runtime import ScriptRuntime
from agent_harness.orchestration.testing import FakeWorkerExecutor
from agent_harness.orchestration.types import RunLimits
from orchestrations.contracts import SubtaskPlanV1, fallback_plan, validate_plan
from orchestrations.profiles import load_model_profile
from orchestrations.registry import get_orchestration_script


def plan() -> dict:
    def subtask(id_, question, dependencies, difficulty):
        return {"id": id_, "question": question, "dependencies": dependencies,
                "difficulty": difficulty, "success_criteria": ["done"],
                "required_evidence": ["source"], "tools": ["web_search"]}
    return {"original_task": "find answer", "subtasks": [
        subtask("a", "find candidates", [], "easy"),
        subtask("b", "check history", [], "medium"),
        subtask("c", "combine filters", ["a", "b"], "easy"),
    ], "synthesis_requirements": ["short answer"]}


class DagContractTests(unittest.TestCase):
    def test_valid_plan_and_profile(self):
        validate_plan(SubtaskPlanV1.from_dict(plan()))
        validate_plan(fallback_plan("task"))
        routes = load_model_profile("deepseek_v4_flash")
        self.assertIn("decomposer", routes)
        self.assertEqual(set(routes), {
            "decomposer", "researcher", "solver", "verifier",
            "synthesizer", "evolution_judge", "evidence_storesubagent",
        })
        self.assertEqual(set(routes.values()), {"deepseek-v4-flash"})

    def test_invalid_dags_are_rejected(self):
        values = []
        value = plan()
        value["subtasks"][0]["dependencies"] = ["c"]
        values.append(value)
        value = plan()
        value["subtasks"][0]["dependencies"] = ["missing"]
        values.append(value)
        value = plan()
        value["subtasks"][1]["id"] = "a"
        values.append(value)
        value = plan()
        value["subtasks"].append(dict(value["subtasks"][0], id="d"))
        values.append(value)
        for value in values:
            with self.assertRaises(ValueError):
                validate_plan(SubtaskPlanV1.from_dict(value))

    def test_unknown_subtask_tool_is_rejected(self):
        value = plan()
        value["subtasks"][0]["tools"] = ["shell_anything"]
        with self.assertRaisesRegex(ValueError, "unavailable tools"):
            validate_plan(SubtaskPlanV1.from_dict(value))


class ReferenceWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_frontier_rounds_critique_and_trace(self):
        verifier_calls = {}

        async def solver(request):
            await asyncio.sleep(0.01)
            return f"solution {request.key}"

        def verifier(request):
            sid = request.key.split(":")[1]
            verifier_calls[sid] = verifier_calls.get(sid, 0) + 1
            accepted = sid != "b" or verifier_calls[sid] == 2
            return {"accepted": accepted, "critique": "补充历史来源" if not accepted else "",
                    "satisfied_criteria": ["done"] if accepted else [],
                    "failed_criteria": [] if accepted else ["done"],
                    "verified_evidence": ["https://example.test"], "confidence": 0.9}

        executor = FakeWorkerExecutor(handlers={"decomposer": lambda _: plan(),
            "solver": solver, "verifier": verifier, "synthesizer": lambda _: "FINAL"})
        registered = get_orchestration_script("reference_v1")
        with tempfile.TemporaryDirectory() as directory:
            result = await ScriptRuntime(executor, limits=RunLimits(max_parallel_jobs=4, max_total_jobs=20),
                checkpoint_path=Path(directory) / "checkpoint.json").run(
                    registered.solve, task="find answer", run_id="reference-test",
                    metadata={"script_source_hash": registered.manifest.source_hash,
                              "budget": {"max_tokens": 72_000}})
        self.assertEqual(result.value, "FINAL")
        self.assertEqual(verifier_calls, {"a": 1, "b": 2, "c": 1})
        retry = next(r for r in executor.requests if r.key == "reference_v1:b:solver:1")
        self.assertIn("补充历史来源", retry.instruction)
        self.assertEqual(retry.metadata["allowed_tools"], ["web_search"])
        dependent = next(r for r in executor.requests if r.key == "reference_v1:c:solver:0")
        self.assertIn('"a"', dependent.instruction)
        names = [e.payload.get("script_event") for e in result.events if e.event_type.value == "script_event"]
        self.assertIn("verifier_round_completed", names)
        self.assertGreaterEqual(executor.max_active_jobs, 2)

    async def test_checkpoint_resume_and_source_guard(self):
        registered = get_orchestration_script("reference_v1")
        verdict = {"accepted": True, "critique": "", "satisfied_criteria": ["done"],
                   "failed_criteria": [], "verified_evidence": [], "confidence": 1.0}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.json"
            metadata = {"script_source_hash": registered.manifest.source_hash,
                        "budget": {"max_tokens": 72_000}}
            first = FakeWorkerExecutor(handlers={"decomposer": lambda _: fallback_plan("task").to_dict(),
                "solver": lambda _: "solution", "verifier": lambda _: verdict,
                "synthesizer": lambda _: "SAVED"})
            await ScriptRuntime(first, checkpoint_path=path).run(
                registered.solve, task="task", run_id="first", metadata=metadata)
            second = FakeWorkerExecutor(allowed_roles=set())
            resumed = await ScriptRuntime(second, checkpoint_path=path).run(
                registered.solve, task="task", run_id="second", metadata=metadata)
            self.assertEqual(resumed.value, "SAVED")
            self.assertEqual(second.requests, [])
            self.assertIn("run_resumed", [e.event_type.value for e in resumed.events])
            with self.assertRaisesRegex(ValueError, "source hash"):
                await ScriptRuntime(second, checkpoint_path=path).run(
                    registered.solve, task="task", run_id="bad",
                    metadata={"script_source_hash": "different"})

    async def test_budget_reserve_keeps_best_effort(self):
        executor = FakeWorkerExecutor(handlers={
            "decomposer": lambda _: fallback_plan("task").to_dict(),
            "synthesizer": lambda _: "信息未查全"})
        registered = get_orchestration_script("reference_v1")
        result = await ScriptRuntime(executor).run(
            registered.solve, task="task", run_id="budget",
            metadata={"script_source_hash": registered.manifest.source_hash,
                      "budget": {"max_tokens": 10_000}})
        self.assertEqual(result.value, "信息未查全")
        self.assertEqual([r.role for r in executor.requests], ["decomposer", "synthesizer"])

    async def test_solver_failure_is_checkpointed_and_retried(self):
        value = plan()
        value["subtasks"] = [value["subtasks"][1]]
        value["subtasks"][0]["dependencies"] = []
        calls = 0

        def solver(request):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise TimeoutError("temporary provider timeout")
            self.assertIn("Solver失败:", request.instruction)
            self.assertIn("temporary provider timeout", request.instruction)
            return "recovered solution"

        verdict = {"accepted": True, "critique": "",
                   "satisfied_criteria": ["done"], "failed_criteria": [],
                   "verified_evidence": [], "confidence": 1.0}
        executor = FakeWorkerExecutor(handlers={
            "decomposer": lambda _: value,
            "solver": solver,
            "verifier": lambda _: verdict,
            "synthesizer": lambda _: "RECOVERED",
        })
        registered = get_orchestration_script("reference_v1")
        with tempfile.TemporaryDirectory() as directory:
            result = await ScriptRuntime(
                executor,
                checkpoint_path=Path(directory) / "checkpoint.json",
            ).run(
                registered.solve, task="find answer", run_id="solver-retry",
                metadata={"script_source_hash": registered.manifest.source_hash,
                          "budget": {"max_tokens": 72_000}},
            )
        self.assertEqual(result.value, "RECOVERED")
        self.assertEqual(calls, 2)
        self.assertEqual(
            [request.role for request in executor.requests].count("verifier"), 1,
        )
