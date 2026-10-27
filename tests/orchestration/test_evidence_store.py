"""Offline transactional, permission and multi-round evidence integration tests."""

from __future__ import annotations

import asyncio
import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_harness.core.llm import LLMResponse
from agent_harness.orchestration import RunLimits, ScriptRuntime
from agent_harness.orchestration.agent_executor import AgentHarnessWorkerExecutor
from agent_harness.orchestration.evidence_judge import (
    LOCAL_JUDGE_PROMPT,
    REPORT_JUDGE_PROMPT,
    fallback_answer,
    local_judge_role,
    normalize_local_judge,
)
from agent_harness.orchestration.evidence_store import (
    EVIDENCE_ROLE,
    EvidenceScope,
    EvidenceStore,
    query_document,
)
from agent_harness.orchestration.generated_code import GeneratedSolveValidator
from tests.orchestration import test_rolechange_v4 as fusion_fixtures
from tests.orchestration.test_rolechange_v4 import (
    OfflineSession,
    ScriptedLLM,
    SOURCE,
    proposal,
)


def add(claim, **fields):
    return {
        "op": "add",
        "fact_id": "",
        "fact": {
            "claim": claim,
            "trace_refs": ["round:1:trace"],
            **fields,
        },
        "reason": "Observed local evidence",
    }


def commit(store, changes, round_number=1):
    return store._commit(
        changes,
        round_number=round_number,
        job_id="test:maintenance",
        observed_urls={"https://example.test/source"},
        trace_refs={"round:1:trace", "round:2:trace"},
    )


class EvidenceStoreTests(unittest.IsolatedAsyncioTestCase):
    async def test_large_table_filters_versions_and_physical_deletion(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "facts.json")
            self.assertEqual(await store.query(), {"revision": 0, "facts": []})
            commit(store, [add(f"Claim {i}") for i in range(80)])
            first, second = store.snapshot()["facts"][:2]
            commit(store, [
                {"op": "delete", "fact_id": first["id"], "fact": {}, "reason": "Merged into second"},
                {"op": "update", "fact_id": second["id"], "fact": {"trace_refs": ["round:2:trace"]},
                 "reason": "Combined evidence"},
            ], round_number=2)
            found = await store.query(fact_ids=[second["id"]], rounds=[2])
            self.assertEqual(found["facts"][0]["id"], second["id"])
            commit(store, [{"op": "delete", "fact_id": second["id"], "fact": {},
                            "reason": "No longer relevant"}], round_number=2)
            self.assertEqual(len((await store.query())["facts"]), 78)
            self.assertEqual(len((await store.query(statuses=[]))["facts"]), 78)
            for status in ("deleted", "superseded"):
                with self.assertRaises(ValueError):
                    await store.query(statuses=[status])
            document = json.loads(store.path.read_text())
            deletion = document["history"][-1]["changes"][0]
            self.assertEqual(deletion["before"]["status"], "active")
            self.assertIsNone(deletion["after"])
            self.assertEqual(document["revision"], 3)

    async def test_atomic_rejection_and_storage_failure_preserve_entire_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "facts.json")
            commit(store, [add("Original")])
            original = store.path.read_bytes()
            for invalid in [
                add("Bad URL", source_urls=["https://invented.test"]),
                add("Bad reference", trace_refs=["invented"]),
                add("Broken edge", conflicts_with=["unknown"]),
            ]:
                with self.assertRaises(ValueError):
                    commit(store, [add("Must also roll back"), invalid])
                self.assertEqual(store.path.read_bytes(), original)
                self.assertEqual(len(store.snapshot()["facts"]), 1)
            with patch.object(Path, "replace", side_effect=OSError("disk failure")):
                with self.assertRaises(OSError):
                    commit(store, [add("Never persisted")])
            self.assertEqual(store.path.read_bytes(), original)
            self.assertEqual(store.snapshot()["revision"], 1)
            self.assertFalse(list(Path(tmp).glob("*.tmp")))

    async def test_duplicate_merge_frozen_reads_and_attempt_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "a.json")
            other = EvidenceStore(Path(tmp) / "b.json")
            commit(store, [add("Same claim")])
            snapshot = store.snapshot()
            original_id = snapshot["facts"][0]["id"]
            commit(
                store,
                [
                    add(
                        "same   claim",
                        source_urls=["https://example.test/source"],
                    )
                ],
            )
            result = await store.query()
            self.assertEqual(len(result["facts"]), 1)
            self.assertEqual(result["facts"][0]["id"], original_id)
            self.assertEqual(result["facts"][0]["source_urls"], ["https://example.test/source"])
            result["facts"].clear()
            self.assertEqual(len((await store.query())["facts"]), 1)
            self.assertEqual(query_document(snapshot)["revision"], 1)
            self.assertEqual((await other.query())["facts"], [])


class MaintenanceLLM(ScriptedLLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.empty_changes = False
        self.tool_sets = []
        self.summary_only = False
        self.requested = False
        self.worker_requests_maintenance = False
        self.judge_record = None
        self.answer_from_history = None
        self.sequence_probe = False
        self.judge_started = asyncio.Event()
        self.maintenance_started = asyncio.Event()

    async def chat(self, messages, **kwargs):
        system = messages[0]["content"]
        self.tool_sets.append(
            (system, {tool["function"]["name"] for tool in (kwargs.get("tools") or [])})
        )
        if system.startswith(LOCAL_JUDGE_PROMPT):
            self.judge_record = json.loads(messages[-1]["content"].split("policy.\n", 1)[1])
            if self.sequence_probe:
                assert self.maintenance_started.is_set()
                self.judge_started.set()
        if system.startswith(EVIDENCE_ROLE.system_prompt):
            if self.sequence_probe:
                self.maintenance_started.set()
                self.judge_started.clear()
                await asyncio.sleep(0.01)
                assert not self.judge_started.is_set()
            if self.empty_changes or self.summary_only:
                payload = json.loads(messages[-1]["content"].split("Local evidence:\n", 1)[1])
                ids = [fact["id"] for fact in payload["related_existing_facts"]]
                return LLMResponse(
                    content=json.dumps(
                        {"summary": "Summary of " + str(ids), "fact_refs": ids, "changes": [], "submitted_fact_ref": ""}
                    ),
                    model=self.model,
                )
        if (
            system.startswith("Investigate the requested question")
            and self.worker_requests_maintenance
            and not self.requested
        ):
            self.requested = True
            return LLMResponse(
                content="Read and organize facts",
                model=self.model,
                tool_calls=[
                    {
                        "id": "maintenance",
                        "type": "function",
                        "function": {
                            "name": "facts_submit",
                            "arguments": json.dumps({"claim": "Relevant evidence", "summary": "Evidence",
                                "source_urls": [], "evidence_excerpt": "Local supported observation"}),
                        },
                    }
                ],
            )
        response = await super().chat(messages, **kwargs)
        if system.startswith(REPORT_JUDGE_PROMPT) and self.round == 3 and self.answer_from_history:
            value = json.loads(response.content)
            value.update(
                decision="best_effort", final_answer=self.answer_from_history
            )
            return LLMResponse(content=json.dumps(value), model=self.model)
        return response


# Reuse the fixture helper, without inheriting and duplicating its test methods.
class EvidenceIntegrationTests(unittest.IsolatedAsyncioTestCase):
    run_case = fusion_fixtures.FusionIntegrationTests.run_case

    async def test_sequential_terminal_reads_and_forced_planner_reads(self):
        llm = MaintenanceLLM(("evolve", "approve_synthesis"))
        llm.sequence_probe = True
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            _, manifest, _ = await self.run_case(path, llm)
            self.assertTrue(manifest["accepted"])
            self.assertIsNone(manifest["rounds"][0]["evidence_maintenance"]["error"])
            first = json.loads(
                (
                    path / "generated/evolution_round_01/plan/attempts/attempt_01/facts_read.json"
                ).read_text()
            )
            second = json.loads(
                (
                    path / "generated/evolution_round_02/plan/attempts/attempt_01/facts_read.json"
                ).read_text()
            )
            self.assertEqual(first, {"revision": 0, "facts": []})
            self.assertEqual(second["revision"], 1)
            self.assertEqual(second["facts"][0]["claim"], "Relevant evidence")
            self.assertTrue(llm.judge_record["facts"])
            self.assertEqual(manifest["fact_store_revision"], 2)
            for system, tools in llm.tool_sets:
                if system.startswith((EVIDENCE_ROLE.system_prompt, LOCAL_JUDGE_PROMPT, REPORT_JUDGE_PROMPT)):
                    self.assertEqual(tools, {"facts_index", "facts_query"})

    async def test_first_round_legacy_active_facts_are_not_imported_without_model_changes(self):
        llm = MaintenanceLLM()
        llm.empty_changes = True
        llm.source = SOURCE.replace(
            '"facts": []',
            '"facts": [{"claim": "Tagged local fact", "source_urls": [], "status": "active", "evidence_excerpt": "A local observation"}]',
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            _, manifest, _ = await self.run_case(path, llm)
            self.assertTrue(manifest["accepted"])
            self.assertEqual(manifest["fact_memory"], [])
            self.assertEqual(manifest["fact_store_revision"], 0)

    async def test_nested_maintenance_with_one_worker_slot_is_counted_and_does_not_deadlock(self):
        llm = MaintenanceLLM()
        llm.worker_requests_maintenance = True
        llm.summary_only = True
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch(
                "agent_harness.orchestration.evidence_store.EvidenceStore._save",
                autospec=True,
            ),
        ):
            store = EvidenceStore(Path(tmp) / "facts.json")
            async with OfflineSession(llm):
                executor = AgentHarnessWorkerExecutor.from_runtime_services(
                    evidence_scope=EvidenceScope(store, 1)
                )
                runtime = ScriptRuntime(
                    executor, limits=RunLimits(max_parallel_jobs=1, max_total_jobs=2)
                )

                async def solve(ctx):
                    return await ctx.call("researcher", "Read and summarize local facts")

                result = await asyncio.wait_for(runtime.run(solve, task="Task"), 3)
                self.assertEqual(len(result.jobs), 2)
                self.assertEqual(result.jobs[1].parent_job_id, result.jobs[0].job_id)
                self.assertEqual(store.snapshot()["revision"], 0)
                for role in ("evidence_storesubagent", "evolution_judge"):
                    for tool in (
                        "web_search",
                        "web_fetch",
                        "run_python_code",
                        "evidence_store_request",
                    ):
                        with self.assertRaises(ValueError):
                            await executor.execute_tool(tool, {}, role=role)

    async def test_script_query_and_submission_are_runnable(self):
        llm = MaintenanceLLM()
        llm.source = SOURCE.replace("    subtasks =", "    index = await ctx.facts_index()\n"
                                  "    details = await ctx.facts_query(statuses=[])\n    subtasks =")
        with tempfile.TemporaryDirectory() as tmp:
            _, manifest, requests = await self.run_case(Path(tmp), llm)
            self.assertTrue(manifest["accepted"], manifest)
            self.assertEqual(len([r for r in requests if r.role == "evidence_storesubagent"]), 2)
            self.assertEqual(manifest["usage"]["jobs"], 9)
            self.assertEqual(manifest["fact_store_revision"], 1)
        legacy = GeneratedSolveValidator(allowed_roles={"solver"}, max_parallel_jobs=1)
        self.assertFalse(legacy.validate("async def solve(ctx):\n    return await ctx.facts_index()").accepted)


    async def test_third_round_history_selection_and_no_candidate_inference(self):
        for failed_rounds in ((), (1, 2, 3)):
            llm = MaintenanceLLM(("evolve",), generation_failures=failed_rounds)
            llm.answer_from_history = "Most likely answer from accumulated evidence"
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)
                outcome, manifest, requests = await self.run_case(path, llm)
                self.assertEqual(manifest["status"], "best_effort")
                self.assertFalse(manifest["accepted"])
                self.assertTrue(manifest["answer_emitted"])
                self.assertTrue(manifest["fallback_used"])
                self.assertEqual(
                    (path / "agent/final_answer.txt").read_text(), llm.answer_from_history
                )
                self.assertNotIn("prior_round_traces", llm.judge_record)
                self.assertNotIn("prior_round_reports", llm.judge_record)
                self.assertFalse(any(r.key == "terminal:synthesizer" for r in requests))
                self.assertNotIn("_terminal_error", outcome)

    async def test_no_candidate_and_broken_judge_emits_nonempty_failure(self):
        llm = ScriptedLLM(generation_failures=(1, 2, 3))
        llm.invalid_judge = True
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            outcome, manifest, _ = await self.run_case(path, llm)
            self.assertEqual(manifest["status"], "failed")
            self.assertIn("_terminal_error", outcome)
            self.assertTrue((path / "agent/final_answer.txt").read_text().strip())

    async def test_maintenance_failure_preserves_previous_table(self):
        llm = ScriptedLLM(("evolve", "approve_synthesis"))
        with tempfile.TemporaryDirectory() as tmp:
            _, manifest, _ = await self.run_case(
                Path(tmp),
                llm,
                failures={
                    (2, "evidence_storesubagent", "terminal:evidence_store"),
                },
            )
            self.assertTrue(manifest["accepted"])
            self.assertEqual(manifest["fact_store_revision"], 2)
            self.assertTrue(manifest["rounds"][1]["evidence_maintenance"]["error"])
            self.assertEqual(len(manifest["fact_memory"]), 1)

    async def test_final_synthesis_failure_retains_approved_candidate(self):
        llm = ScriptedLLM(("evolve", "evolve", "approve_synthesis"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            _, manifest, _ = await self.run_case(
                path,
                llm,
                failures={
                    (3, "synthesizer", "terminal:synthesizer"),
                },
            )
            self.assertTrue(manifest["accepted"])
            self.assertTrue(manifest["fallback_used"])
            self.assertEqual((path / "agent/final_answer.txt").read_text(), "Candidate A")

    def test_historical_fallback_prefers_authentic_eligible_candidate(self):
        accepted = proposal()
        rejected = {**proposal(), "status": "incomplete", "selected_candidate": "Latest rejected"}
        self.assertEqual(fallback_answer([accepted, rejected]), "Candidate A")
        self.assertEqual(fallback_answer([rejected]), "Latest rejected")
        self.assertEqual(fallback_answer([]), "")
        self.assertEqual(local_judge_role(3).allowed_tools, ["facts_index", "facts_query"])
        self.assertNotIn("confirmed_facts", local_judge_role(1).result_schema["properties"])
        decision = normalize_local_judge(
            {
                "decision": "best_effort",
                "final_answer": "  ",
                "rationale": "",
                "failure_report": None,
            },
            round_number=3,
            proposal=accepted,
            failure_stage="judge",
            summary="broken",
            previous_report=None,
        )
        self.assertEqual(decision["decision"], "evolve")

    async def test_concurrent_maintenance_serializes_snapshot_and_commit(self):
        from tests.orchestration.test_fact_submission import SubmissionLLM
        llm = SubmissionLLM(); llm.per_job_claim = True
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "facts.json")
            async with OfflineSession(llm):
                runtime = ScriptRuntime(AgentHarnessWorkerExecutor.from_runtime_services(
                    evidence_scope=EvidenceScope(store, 1)),
                    limits=RunLimits(max_parallel_jobs=2, max_total_jobs=4))
                async def solve(ctx):
                    return await ctx.gather(ctx.spawn("researcher", "First"), ctx.spawn("researcher", "Second"))
                result = await runtime.run(solve, task="Task")
            self.assertEqual([p["store_revision"] for p in llm.packets], [0, 1])
            self.assertEqual(len(result.jobs), 4)
            self.assertEqual(len(store.snapshot()["facts"]), 2)


    async def test_later_submission_replaces_fact_and_next_planner_reads_resolution(self):
        class ConflictLLM(ScriptedLLM):
            async def chat(self, messages, **kwargs):
                system = messages[0]["content"]
                if system.startswith(EVIDENCE_ROLE.system_prompt):
                    payload = json.loads(messages[-1]["content"].split("Local evidence:\n", 1)[1])
                    if self.round == 2 and payload["mode"] == "round_end":
                        rows = payload["related_existing_facts"]
                        obsolete = next(f for f in rows if f["claim"] == "Relevant evidence")
                        return LLMResponse(content=json.dumps({"summary": "Reconciled correction", "fact_refs": [],
                            "changes": [{"op": "delete", "fact_id": obsolete["id"], "fact": {},
                                         "reason": "Replaced by source-supported correction"}]}), model=self.model)
                response = await super().chat(messages, **kwargs)
                for call in response.tool_calls or []:
                    if self.round >= 2 and call["function"]["name"] == "facts_submit":
                        value = json.loads(call["function"]["arguments"])
                        value.update(claim="Corrected claim", summary="Corrected summary")
                        call["function"]["arguments"] = json.dumps(value)
                return response
        with tempfile.TemporaryDirectory() as tmp:
            _, manifest, _ = await self.run_case(Path(tmp), ConflictLLM(("evolve", "evolve", "approve_synthesis")))
            self.assertEqual([f["claim"] for f in manifest["fact_memory"]], ["Corrected claim"])
            read = json.loads((Path(tmp) / "generated/evolution_round_03/plan/attempts/attempt_01/facts_read.json").read_text())
            self.assertEqual([f["claim"] for f in read["facts"]], ["Corrected claim"])


    async def test_plan_repair_reads_facts_before_each_new_plan(self):
        from agent_harness.orchestration.contracts import BudgetSpec
        from agent_harness.orchestration.generated_code import GeneratedSolvePlanner
        from tests.orchestration.test_rolechange_v4 import request

        class ReplanLLM(ScriptedLLM):
            async def chat(self, messages, **kwargs):
                if (
                    "independent reviewer of a high-level" in messages[0]["content"]
                    and self.round == 1
                ):
                    return LLMResponse(
                        content=json.dumps(
                            {
                                "accepted": False,
                                "critique": "Try another plan",
                                "issues": [],
                                "suggested_changes": [],
                            }
                        ),
                        model=self.model,
                    )
                return await super().chat(messages, **kwargs)

        reads = []

        async def query(**kwargs):
            reads.append(kwargs)
            return {"revision": len(reads), "facts": []}

        with tempfile.TemporaryDirectory() as tmp:
            req = request()
            planner = GeneratedSolvePlanner(
                ReplanLLM(),
                GeneratedSolveValidator(
                    allowed_roles=set(req.environment.roles),
                    max_parallel_jobs=4,
                    require_evolution_proposal=True,
                ),
                model="offline",
                max_repair_attempts=1,
                facts_query=query,
            )
            result = await planner.generate(req, artifact_dir=Path(tmp), budget=BudgetSpec())
            self.assertIsNotNone(result.generated)
            self.assertEqual(len(reads), 2)
            self.assertEqual(result.usage.tool_calls, 2)
            self.assertEqual(
                json.loads((Path(tmp) / "plan/attempts/attempt_02/facts_read.json").read_text())[
                    "revision"
                ],
                2,
            )

    async def test_nested_maintenance_cannot_exceed_job_budget(self):
        llm = MaintenanceLLM()
        llm.worker_requests_maintenance = True
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "facts.json")
            async with OfflineSession(llm):
                runtime = ScriptRuntime(
                    AgentHarnessWorkerExecutor.from_runtime_services(
                        evidence_scope=EvidenceScope(store, 1),
                    ),
                    limits=RunLimits(max_parallel_jobs=1, max_total_jobs=1),
                )

                async def solve(ctx):
                    return await ctx.call("researcher", "Request maintenance")

                result = await runtime.run(solve, task="Task")
                self.assertEqual(len(result.jobs), 1)
                self.assertEqual(store.snapshot()["revision"], 0)

    async def test_solver_candidate_survives_script_failure_and_broken_judge(self):
        llm = ScriptedLLM()
        llm.invalid_judge = True
        failures = {(r, "verifier", "verify") for r in range(1, 4)}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            outcome, manifest, _ = await self.run_case(path, llm, failures=failures)
            self.assertEqual(manifest["status"], "best_effort")
            self.assertNotIn("_terminal_error", outcome)
            self.assertEqual((path / "agent/final_answer.txt").read_text(), "Candidate A")


class TwoStageLLM(ScriptedLLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage_inputs = []
        self.query_results = []
        self.report_outputs = []
        self.break_report = None
        self.repaired_reports = 0
        self.repair_value = None
        self.pending_query = False
        self.queried = set()

    async def chat(self, messages, **kwargs):
        system = messages[0]["content"]
        if system.startswith("Repair only the response format") and self.repair_value is not None:
            self.repaired_reports += 1
            return LLMResponse(content=json.dumps(self.repair_value), model=self.model,
                               usage={"total_tokens": 7})
        stage = ("assessment" if system.startswith(LOCAL_JUDGE_PROMPT) else
                 "report" if system.startswith(REPORT_JUDGE_PROMPT) else None)
        if stage:
            key = self.round, stage
            if key not in self.queried:
                self.queried.add(key)
                record = json.loads(messages[-1]["content"].split("policy.\n", 1)[1])
                self.stage_inputs.append((self.round, stage, record))
                self.pending_query = True
                return LLMResponse(content="Read the bound facts", model=self.model, tool_calls=[{
                    "id": f"{self.round}-{stage}", "type": "function",
                    "function": {"name": "facts_query", "arguments": '{"statuses":["active","disputed"]}'},
                }], usage={"total_tokens": 3})
            if self.pending_query:
                self.query_results.append((self.round, stage, ast.literal_eval(messages[-1]["content"])))
                self.pending_query = False
        response = await super().chat(messages, **kwargs)
        if stage == "report":
            value = json.loads(response.content)
            self.report_outputs.append(value)
            if self.break_report == "missing_instructions":
                self.repair_value = json.loads(response.content)
                value["failure_report"].pop("planner_instructions")
                return LLMResponse(content=json.dumps(value), model=self.model, usage={"total_tokens": 5})
            if self.break_report == "empty_final" and self.round == 3:
                value["final_answer"] = "  "
                self.repair_value = value
                return LLMResponse(content=json.dumps(value), model=self.model, usage={"total_tokens": 5})
        return response


class TwoStageIntegrationTests(unittest.IsolatedAsyncioTestCase):
    run_case = fusion_fixtures.FusionIntegrationTests.run_case

    async def test_two_calls_share_latest_facts_and_report_reaches_planner(self):
        llm = TwoStageLLM(("evolve", "approve_synthesis"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            outcome, manifest, requests = await self.run_case(path, llm)
            self.assertTrue(manifest["accepted"])
            keys = [r.key for r in requests if r.key.startswith("terminal:")]
            self.assertEqual(keys, ["terminal:evidence_store", "terminal:evolution_judge",
                                    "terminal:evolution_report", "terminal:evidence_store",
                                    "terminal:evolution_judge", "terminal:synthesizer"])
            self.assertEqual([(r, stage) for r, stage, _ in llm.stage_inputs],
                             [(1, "assessment"), (1, "report"), (2, "assessment")])
            for (_, stage, record), (_, _, queried) in zip(llm.stage_inputs, llm.query_results):
                self.assertEqual(record["facts"], queried["facts"])
                self.assertEqual(set(record), {"task", "facts", "candidate_answer"}
                                 | ({"assessment"} if stage == "report" else set()))
            for _, _, result in llm.query_results:
                self.assertTrue(result["facts"])
            self.assertEqual([result["revision"] for _, _, result in llm.query_results], [1, 1, 2])
            first_input = llm.stage_inputs[0][2]
            self.assertTrue(first_input["task"])
            self.assertEqual(first_input["candidate_answer"], "Candidate A")
            self.assertEqual(llm.stage_inputs[1][2]["assessment"]["issues"], ["Need stronger support"])
            plan = next(prompt for r, stage, prompt in llm.prompts if r == 2 and stage == "plan")
            self.assertIn("Do not treat repeated assertions", plan)
            self.assertIn("acceptance_conditions", plan)
            self.assertEqual(manifest["max_rounds"], 3)
            self.assertEqual(outcome["usage"]["jobs"], 18)
            self.assertEqual(manifest["rounds"][0]["usage"]["jobs"], 9)
            self.assertTrue((path / "generated/evolution_round_01/terminal_review/report.json").exists())
            self.assertFalse((path / "generated/evolution_round_02/terminal_review/report.json").exists())

    async def test_trace_markers_excluded_from_all_rounds_and_repairs(self):
        from agent_harness.orchestration import evolution_judge, evidence_judge
        original_trace = evolution_judge.build_structured_round_trace
        original_report = evidence_judge.local_failure_report
        markers = ("PRIVATE_TRACE_MARKER", "PRIVATE_VERIFIER_MARKER", "PRIVATE_HISTORY_MARKER")
        captured = []

        def traced(*args, **kwargs):
            record = original_trace(*args, **kwargs)
            record["private_trace"] = markers[0]
            record["private_verifier_commentary"] = markers[1]
            return record

        def reported(*args, **kwargs):
            report = original_report(*args, **kwargs)
            report["summary"] += markers[2]
            return report

        class RecordingLLM(TwoStageLLM):
            async def chat(self, messages, **kwargs):
                system = messages[0]["content"]
                if system.startswith((LOCAL_JUDGE_PROMPT, REPORT_JUDGE_PROMPT,
                                      "Repair only the response format")):
                    captured.append(json.dumps(messages))
                return await super().chat(messages, **kwargs)

        llm = RecordingLLM(("approve_synthesis",), verifier_accept=False)
        llm.break_report = "missing_instructions"
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(evolution_judge, "build_structured_round_trace", side_effect=traced), \
                 patch.object(evidence_judge, "local_failure_report", side_effect=reported):
                _, manifest, _ = await self.run_case(Path(tmp), llm)
            self.assertEqual(manifest["completed_rounds"], 3)
            self.assertFalse(manifest["accepted"])
            self.assertEqual(llm.repaired_reports, 3)
            self.assertEqual(len(llm.stage_inputs), 6)
            for _, stage, record in llm.stage_inputs:
                self.assertEqual(set(record), {"task", "facts", "candidate_answer"}
                                 | ({"assessment"} if stage == "report" else set()))
                if stage == "report":
                    self.assertEqual(record["assessment"]["rationale"], "offline terminal audit")
                    self.assertNotIn("eligible_before_judge", record["assessment"])
            for prompt in captured:
                for marker in markers:
                    self.assertNotIn(marker, prompt)
            trace = (Path(tmp) / "generated/evolution_round_03/terminal_review/structured_trace.json").read_text()
            self.assertIn(markers[0], trace)
            self.assertIn(markers[1], trace)
            self.assertIn(markers[2], json.dumps(manifest))

    async def test_disputed_facts_preserved_in_inputs_and_snapshot_queries(self):
        class DisputedLLM(TwoStageLLM):
            async def chat(self, messages, **kwargs):
                if messages[0]["content"].startswith(EVIDENCE_ROLE.system_prompt):
                    payload = json.loads(messages[-1]["content"].split("Local evidence:\n", 1)[1])
                    if payload["mode"] == "round_end":
                        return LLMResponse(content=json.dumps({"summary": "Reviewed conflicts", "fact_refs": [], "changes": []}), model=self.model)
                    refs = payload["new_facts"][0]["trace_refs"]
                    value = {"summary": "Unresolved evidence conflict", "fact_refs": [], "submitted_fact_ref": "new:a", "changes": [
                        {**add("Relevant evidence", trace_refs=refs, status="disputed", conflicts_with=["new:b"]), "local_ref": "new:a", "summary": "Evidence"},
                        {**add("Contrary evidence", trace_refs=refs, status="disputed", conflicts_with=["new:a"]), "local_ref": "new:b", "summary": "Contrary evidence"},
                    ]}
                    return LLMResponse(content=json.dumps(value), model=self.model)
                return await super().chat(messages, **kwargs)

        llm = DisputedLLM(("evolve", "approve_synthesis"))
        with tempfile.TemporaryDirectory() as tmp:
            await self.run_case(Path(tmp), llm)
            for (_, _, record), (_, _, queried) in zip(llm.stage_inputs, llm.query_results):
                self.assertTrue(record["facts"])
                self.assertTrue(all(f["status"] == "disputed" for f in record["facts"]))
                self.assertEqual(record["facts"], queried["facts"])

        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "facts.json")
            commit(store, [
                {**add("Original disputed claim", status="disputed", conflicts_with=["new:b"]), "local_ref": "new:a"},
                {**add("Contrary claim", status="disputed", conflicts_with=["new:a"]), "local_ref": "new:b"},
            ])
            frozen = store.snapshot()
            scope = EvidenceScope(store, 1, read_snapshot=frozen)
            commit(store, [add("Later claim")], round_number=2)
            self.assertEqual(await scope.query(statuses=[]), {
                "revision": frozen["revision"], "facts": frozen["facts"],
            })
            self.assertEqual([tool.name for tool in scope.tools("evolution_judge")], ["facts_index", "facts_query"])
            self.assertEqual(scope.trace, {})

    async def test_missing_planner_instructions_uses_model_repair(self):
        llm = TwoStageLLM(("evolve", "approve_synthesis"))
        llm.break_report = "missing_instructions"
        with tempfile.TemporaryDirectory() as tmp:
            _, manifest, _ = await self.run_case(Path(tmp), llm)
            self.assertTrue(manifest["accepted"])
            self.assertEqual(llm.repaired_reports, 1)
            self.assertIn("planner_instructions", manifest["rounds"][0]["failure_report"])
            self.assertIsNone(manifest["rounds"][0]["evolution_judge"]["report_error"])

    async def test_report_service_failure_preserves_actionable_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, manifest, _ = await self.run_case(
                Path(tmp), ScriptedLLM(("evolve", "approve_synthesis")),
                failures={(1, "evolution_judge", "terminal:evolution_report")},
            )
            report = manifest["rounds"][0]["failure_report"]
            self.assertTrue(manifest["accepted"])
            self.assertIn("offline injected", report["report_error"])
            self.assertIn("Need stronger support", report["failed_routes"])
            self.assertTrue(all(report["planner_instructions"].values()))

    async def test_final_empty_model_answer_repairs_then_uses_history(self):
        llm = TwoStageLLM(("evolve",))
        llm.break_report = "empty_final"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            _, manifest, _ = await self.run_case(path, llm)
            self.assertEqual(manifest["completed_rounds"], 3)
            self.assertEqual(manifest["status"], "best_effort")
            self.assertEqual((path / "agent/final_answer.txt").read_text(), "Candidate A")
            self.assertTrue(manifest["rounds"][-1]["evolution_judge"]["report_error"])
            self.assertFalse(manifest["rounds"][-1]["failure_report"]["continue_evolution"])
            self.assertFalse((path / "generated/evolution_round_04").exists())

    async def test_current_generation_fails_final_report_gets_only_accumulated_facts(self):
        llm = TwoStageLLM(("evolve",), generation_failures=(3,))
        with tempfile.TemporaryDirectory() as tmp:
            _, manifest, requests = await self.run_case(Path(tmp), llm)
            self.assertEqual(manifest["status"], "best_effort")
            record = llm.stage_inputs[-1][2]
            self.assertEqual(record["candidate_answer"], "")
            self.assertEqual(set(record), {"task", "facts", "candidate_answer", "assessment"})
            self.assertTrue(record["facts"])
            self.assertEqual(requests[-1].key, "terminal:evolution_report")

    async def test_maintenance_failure_keeps_last_facts_without_disclosing_trace(self):
        llm = TwoStageLLM(("evolve", "evolve", "approve_synthesis"))
        with tempfile.TemporaryDirectory() as tmp:
            _, manifest, _ = await self.run_case(
                Path(tmp), llm, failures={(2, "evidence_storesubagent", "terminal:evidence_store")},
            )
            self.assertTrue(manifest["accepted"])
            for round_number, _, record in llm.stage_inputs:
                if round_number == 2:
                    self.assertEqual([f["id"] for f in record["facts"]], [f["id"] for f in llm.stage_inputs[0][2]["facts"]])
                    self.assertEqual(record["facts"][0]["updated_round"], 2)
                    self.assertNotIn("offline injected", json.dumps(record))

    def test_report_contract_and_legacy_round_limit(self):
        from agent_harness.orchestration.agent_executor import parse_role_output
        from agent_harness.orchestration.exceptions import InvalidRoleOutput
        from agent_harness.orchestration.self_evolving import MAX_EVOLUTION_ROUNDS
        self.assertEqual(MAX_EVOLUTION_ROUNDS, 5)
        self.assertEqual(local_judge_role(3, stage="report").allowed_tools, ["facts_index", "facts_query"])
        self.assertNotIn("rounds 3-5", local_judge_role(3).system_prompt)
        self.assertNotIn("failure_report", local_judge_role(3).system_prompt)
        with self.assertRaises(InvalidRoleOutput):
            parse_role_output('{"decision":"evolve","rationale":"gap","failure_report":{}}',
                              local_judge_role(1, stage="report").result_schema, role="evolution_judge")

    async def test_empty_final_synthesis_retains_approved_candidate(self):
        class EmptySynthesisLLM(ScriptedLLM):
            async def chat(self, messages, **kwargs):
                if self.round == 3 and messages[0]["content"].startswith("Combine the supplied intermediate results"):
                    return LLMResponse(content=" ", model=self.model)
                return await super().chat(messages, **kwargs)

        llm = EmptySynthesisLLM(("evolve", "evolve", "approve_synthesis"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            _, manifest, _ = await self.run_case(path, llm)
            self.assertEqual(manifest["status"], "approved")
            self.assertTrue(manifest["fallback_used"])
            self.assertEqual((path / "agent/final_answer.txt").read_text(), "Candidate A")

    async def test_minimum_job_budget_and_four_reserved_jobs(self):
        llm = ScriptedLLM(("evolve",))
        llm.source = '''async def solve(ctx):
    verdict = await ctx.call("verifier", "Check the available candidate", key="verify")
    return {"status": "incomplete", "selected_candidate": "Candidate A", "selected_verifier": verdict,
            "facts": [], "issues": ["Needs verification"], "synthesis_context": {}}
'''
        with tempfile.TemporaryDirectory() as tmp:
            _, manifest, _ = await self.run_case(Path(tmp), llm, max_jobs=5)
            self.assertEqual(manifest["status"], "best_effort")
            for record in manifest["rounds"]:
                self.assertEqual(record["usage"]["jobs"], 4)
                self.assertFalse(record["budget_violations"])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "max_jobs >= 5"):
                await self.run_case(Path(tmp), ScriptedLLM(), max_jobs=4)


class FactReferenceFixTests(unittest.IsolatedAsyncioTestCase):
    def prepare(self, store, value):
        return store._prepare_result(value, round_number=1, job_id="test:prepare",
                                     observed_urls={"https://example.test/source"},
                                     trace_refs={"round:1:trace"})

    def test_forward_aliases_dedup_and_summary_are_resolved_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "facts.json")
            changes = [
                {**add("First", status="disputed", conflicts_with=["new:b"]), "local_ref": "new:a"},
                {**add("Second", status="disputed", conflicts_with=["new:a"]), "local_ref": "new:b"},
                {**add("  FIRST  "), "local_ref": "new:duplicate"},
            ]
            document, result = self.prepare(store, {
                "summary": "Compare [new:a] with [new:b]", "fact_refs": ["new:a", "new:b"],
                "changes": changes, "resolved_refs": {"new:a": "forged"},
            })
            self.assertEqual(store.snapshot()["revision"], 0)
            self.assertEqual(json.loads(store.path.read_text())["facts"], [])
            refs = result["resolved_refs"]
            self.assertEqual(refs["new:a"], refs["new:duplicate"])
            self.assertNotEqual(refs["new:a"], "forged")
            self.assertEqual(len(document["facts"]), 2)
            self.assertEqual(document["facts"][0]["conflicts_with"], [refs["new:b"]])
            self.assertEqual(document["facts"][1]["conflicts_with"], [refs["new:a"]])
            self.assertNotIn("new:", result["summary"])
            self.assertEqual(result["changes"][0]["fact_id"], refs["new:a"])
            store._publish(document)
            self.assertEqual(json.loads(store.path.read_text()), document)

    def test_unknown_ambiguous_and_duplicate_references_do_not_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "facts.json")
            commit(store, [add("First"), add("Second")])
            ids = [f["id"] for f in store.snapshot()["facts"]]
            before = store.path.read_bytes()
            cases = [
                {"summary": "", "fact_refs": ["new:missing"], "changes": [add("New")]},
                {"summary": "[fact-fabricated]", "fact_refs": [], "changes": [add("New")]},
                {"summary": "", "fact_refs": ["First"], "changes": [
                    {"op": "update", "fact_id": ids[1], "fact": {"claim": "FIRST"}, "reason": "conflict"}]},
                {"summary": "", "fact_refs": [], "changes": [
                    {**add("New"), "local_ref": "new:duplicate"},
                    {**add("Other"), "local_ref": "new:duplicate"}]},
                {"summary": "", "fact_refs": [], "changes": [
                    {**add("New", conflicts_with=["new:a"]), "local_ref": "new:a"}]},
                {"summary": "", "fact_refs": [], "changes": [add("New", source_urls=["https://invented.test/source"])]},
            ]
            for value in cases:
                with self.subTest(value=value), self.assertRaises(ValueError):
                    self.prepare(store, value)
                self.assertEqual(store.path.read_bytes(), before)
                self.assertEqual(store.snapshot()["revision"], 1)

    def test_prose_references_cannot_create_unsubmitted_facts(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "facts.json")
            with self.assertRaisesRegex(ValueError, "fact_refs\\[0\\]"):
                self.prepare(store, {"summary": "", "fact_refs": ["Claim not submitted"], "changes": []})
            self.assertEqual(store.snapshot()["facts"], [])

    def test_local_url_extraction_preserves_parentheses_and_json_boundaries(self):
        from agent_harness.orchestration.evidence_store import evidence_urls
        url = "https://en.wikipedia.org/wiki/Ang_Huling_El_Bimbo_(musical)"
        other = "https://example.test/path?q=x%26y&n=2"
        value = {"markdown": f"[source]({url}).\nNext", "tool": json.dumps({"url": other}),
                 "prose": f"See ({url}), then stop.", "raw": url}
        found = evidence_urls(value)
        self.assertEqual(found, {url, other})
        self.assertNotIn("https://en.wikipedia.org/wiki/Ang_Huling_El_Bimbo_", found)

    def test_business_validation_error_contains_invalid_source_and_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "facts.json")
            with self.assertRaisesRegex(ValueError, r"changes\[0\].*unknown source_urls=.*fabricated"):
                self.prepare(store, {"summary": "", "fact_refs": [], "changes": [
                    add("Claim", source_urls=["https://example.test/fabricated"])]})


class EvidenceRepairLLM(ScriptedLLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.invalid_mode = "unknown_ref"
        self.repair_calls = 0
        self.maintenance_calls = 0
        self.repair_inputs = []
        self.repair_tools = []
        self.original_fixture = None
        self.corrected_value = None
        self.fail_repair = False
        self.large_marker = False

    async def chat(self, messages, **kwargs):
        system = messages[0]["content"]
        if system.startswith("Repair the evidence maintenance response"):
            self.repair_calls += 1
            self.repair_inputs.append(messages[-1]["content"])
            self.repair_tools.append(kwargs.get("tools") or [])
            value = {**self.corrected_value, "fact_refs": ["new:missing"]} if self.fail_repair else self.corrected_value
            return LLMResponse(content=json.dumps(value), model=self.model, usage={"completion_tokens": 13})
        if system.startswith(EVIDENCE_ROLE.system_prompt):
            payload = json.loads(messages[-1]["content"].split("Local evidence:\n", 1)[1])
            if payload["mode"] == "round_end" and self.original_fixture is None:
                return await super().chat(messages, **kwargs)
            self.maintenance_calls += 1
            if self.original_fixture is not None:
                self.corrected_value = json.loads(json.dumps(self.original_fixture))
                for i, change in enumerate(self.corrected_value["changes"]):
                    change["local_ref"] = f"new:{i}"
                    change["summary"] = change["fact"]["claim"]
                self.corrected_value["fact_refs"] = [c["local_ref"] for c in self.corrected_value["changes"]]
                return LLMResponse(content=json.dumps(self.original_fixture), model=self.model,
                                   usage={"completion_tokens": 11})
            self.corrected_value = {"summary": "Supported claim [new:a]", "fact_refs": ["new:a"],
                                    "submitted_fact_ref": "new:a", "changes": [{
                                        **add("Supported local claim", trace_refs=payload["new_facts"][0]["trace_refs"]),
                                        "summary": "Supported claim", "local_ref": "new:a"}]}
            value = json.loads(json.dumps(self.corrected_value))
            if self.invalid_mode == "unknown_ref":
                value["fact_refs"] = ["new:missing"]
            elif self.invalid_mode == "unknown_source":
                value["changes"][0]["fact"]["source_urls"] = ["https://fabricated.test/never-seen"]
            elif self.invalid_mode == "invalid_status":
                value["changes"][0]["fact"]["status"] = "confident"
            elif self.invalid_mode == "invalid_relation":
                value["changes"][0]["fact"].update(status="disputed", conflicts_with=["fact-made-up"])
            if self.large_marker:
                value["summary"] = "x" * 21000 + " END_OF_ORIGINAL_OUTPUT"
            return LLMResponse(content="{broken" if self.invalid_mode == "json" else json.dumps(value),
                               model=self.model, usage={"completion_tokens": 11})
        return await super().chat(messages, **kwargs)


class EvidenceRepairIntegrationTests(unittest.IsolatedAsyncioTestCase):
    run_case = fusion_fixtures.FusionIntegrationTests.run_case

    async def test_business_and_json_errors_share_one_tool_free_repair(self):
        for mode in ("unknown_ref", "unknown_source", "invalid_status", "invalid_relation", "json"):
            llm = EvidenceRepairLLM()
            llm.invalid_mode = mode
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp)
                _, manifest, _ = await self.run_case(path, llm)
                self.assertTrue(manifest["accepted"])
                self.assertEqual(llm.maintenance_calls, 1)
                self.assertEqual(llm.repair_calls, 1)
                self.assertEqual(llm.repair_tools, [[]])
                self.assertEqual(manifest["fact_store_revision"], 1)
                self.assertEqual(manifest["fact_memory"][0]["claim"], "Supported local claim")
                assessment = json.loads((path / "generated/evolution_round_01/terminal_review/assessment_input.json").read_text())
                self.assertEqual(assessment["facts"][0]["claim"], "Supported local claim")
                self.assertEqual(manifest["usage"]["jobs"], 9)
                self.assertIn("Validation error:", llm.repair_inputs[0])
                self.assertIn("available_source_urls", llm.repair_inputs[0])
                live = (path / "generated/evolution_round_01/evidence_maintenance/live.jsonl").read_text()
                self.assertIn("evidence_output_received", live)
                self.assertIn("resolved_refs", live)

    async def test_invalid_repair_does_not_retry_or_commit_partial_data(self):
        llm = EvidenceRepairLLM()
        llm.fail_repair = True
        with tempfile.TemporaryDirectory() as tmp:
            _, manifest, _ = await self.run_case(Path(tmp), llm)
            self.assertEqual(llm.maintenance_calls, 1)
            self.assertEqual(llm.repair_calls, 1)
            self.assertIsNone(manifest["rounds"][0]["evidence_maintenance"]["error"])  # Empty table is skipped.
            self.assertEqual(manifest["fact_store_revision"], 0)
            self.assertEqual(manifest["fact_memory"], [])

    async def test_repair_prompt_keeps_original_output_beyond_twenty_thousand_characters(self):
        llm = EvidenceRepairLLM()
        llm.large_marker = True
        with tempfile.TemporaryDirectory() as tmp:
            await self.run_case(Path(tmp), llm)
            self.assertIn("END_OF_ORIGINAL_OUTPUT", llm.repair_inputs[0])

    async def test_synthetic_response_retains_all_34_changes_after_reference_repair(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/evidence_13_round_01.json").read_text())
        llm = EvidenceRepairLLM()
        llm.original_fixture = fixture["response"]
        for change in llm.original_fixture["changes"]:
            change["summary"] = change["fact"]["claim"]
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "facts.json")
            trace = {"sources": fixture["observed_urls"], "candidate_proposal": {"facts": []}}
            refs = {r for c in fixture["response"]["changes"] for r in c["fact"].get("trace_refs", [])}
            store._commit([add("Previously committed evidence", source_urls=fixture["observed_urls"], trace_refs=list(refs))],
                round_number=1, job_id="seed", observed_urls=set(fixture["observed_urls"]), trace_refs=refs)
            async with OfflineSession(llm):
                runtime = ScriptRuntime(
                    AgentHarnessWorkerExecutor.from_runtime_services(
                        evidence_scope=EvidenceScope(store, 1, trace=trace, terminal_maintenance=True), capture_evidence=True),
                    limits=RunLimits(max_parallel_jobs=1, max_total_jobs=1),
                )
                async def solve(ctx):
                    return await ctx.call("evidence_storesubagent", "Organize supplied facts", key="maintain")
                result = await runtime.run(solve, task="Offline synthetic response replay")
            self.assertEqual(llm.repair_calls, 1)
            self.assertIn("fact_refs[0]", llm.repair_inputs[0])
            self.assertEqual(len(result.value["changes"]), 34)
            self.assertEqual(len(store.snapshot()["facts"]), 35)
            self.assertEqual(store.snapshot()["revision"], 2)
            self.assertTrue(all(fid.startswith("fact-") for fid in result.value["fact_refs"]))
            self.assertEqual(len(store.snapshot()["history"]), 2)
            from agent_harness.orchestration.evidence_store import usage_from_evidence_run
            usage = usage_from_evidence_run(result)
            self.assertEqual(usage.jobs, 1)
            self.assertEqual(usage.llm_calls, 2)
            self.assertEqual(usage.tokens, 24)

    async def test_failed_second_round_repair_keeps_previous_facts_and_planner_read(self):
        class FailLaterLLM(EvidenceRepairLLM):
            async def chat(self, messages, **kwargs):
                if self.round == 2:
                    self.fail_repair = True
                return await super().chat(messages, **kwargs)
        llm = FailLaterLLM(("evolve", "approve_synthesis"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            _, manifest, _ = await self.run_case(path, llm)
            self.assertTrue(manifest["accepted"])
            self.assertEqual(manifest["fact_store_revision"], 1)
            self.assertEqual(len(manifest["fact_memory"]), 1)
            self.assertEqual(llm.repair_calls, 2)
            self.assertIsNone(manifest["rounds"][1]["evidence_maintenance"]["error"])
            read = json.loads((path / "generated/evolution_round_02/plan/attempts/attempt_01/facts_read.json").read_text())
            self.assertEqual(read["revision"], 1)
            self.assertEqual(read["facts"][0]["claim"], "Supported local claim")
