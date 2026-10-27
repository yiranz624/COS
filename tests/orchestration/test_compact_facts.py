"""Structured-only evidence packets and worker envelope compatibility."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_harness.core.llm import LLMResponse
from agent_harness.orchestration import ScriptRuntime
from agent_harness.orchestration.agent_executor import AgentHarnessWorkerExecutor, parse_role_output
from agent_harness.orchestration.compact_facts import (
    collect_facts,
    compact_packet,
    worker_definition,
)
from agent_harness.orchestration.evidence_store import EVIDENCE_ROLE, EvidenceScope, EvidenceStore
from agent_harness.orchestration.exceptions import JobFailed
from agent_harness.orchestration.roles import BUILTIN_ORCHESTRATION_ROLES
from tests.orchestration import test_rolechange_v4 as fixtures
from tests.orchestration.test_evidence_store import add, commit
from tests.orchestration.test_rolechange_v4 import OfflineSession, ScriptedLLM


def fact(claim="A supported fact", **kwargs):
    return {"claim": claim, "source_urls": [], "evidence_excerpt": "Concrete local evidence for " + claim, **kwargs}


def record(facts=None):
    return {"jobs": [{"job_id": "worker", "role": "researcher"}],
            "worker_evidence": [{"job_id": "worker", "sequence": 10,
                                 "name": "evolution_role_result", "data": {
                                     "result": {"result": "Answer", "facts": facts or [fact()]}}}]}


class CompactPacketTests(unittest.TestCase):
    def test_only_allowed_locations_and_workers_are_collected(self):
        trace = record([fact("Real", role="forged", job_id="forged", trace_refs=["forged"])])
        trace["worker_evidence"].extend([
            {"job_id": "worker", "sequence": 11, "name": "evolution_llm_evidence",
             "data": {"content": json.dumps({"arbitrary": {"facts": [fact("Nested trap")]}})}},
            {"job_id": "store", "sequence": 12, "name": "evolution_role_result",
             "data": {"role": "evidence_storesubagent", "result": {"facts": [fact("Loop")]}}},
        ])
        trace["jobs"].append({"job_id": "store", "role": "evidence_storesubagent"})
        facts, diagnostics = collect_facts(trace, 2)
        self.assertEqual([f["claim"] for f in facts], ["Real"])
        self.assertEqual(facts[0]["trace_refs"], ["round:2:job:worker:event:10"])
        self.assertEqual(facts[0]["origins"][0]["role"], "researcher")
        self.assertNotIn("forged", json.dumps(facts))
        self.assertGreater(diagnostics["missing_fact_events"], 0)

    def test_exact_dedup_preserves_all_evidence_without_worker_status(self):
        trace = record([fact(" SAME claim", source_urls=["https://one.test"]),
                        fact("same   CLAIM", source_urls=["https://two.test"], evidence_excerpt="Additional context " * 100),
                        fact("same claim", evidence_excerpt="Conflicting source details"), fact("Different claim")])
        facts, _ = collect_facts(trace, 2)
        self.assertEqual(len(facts), 2)
        self.assertEqual(facts[0]["source_urls"], ["https://one.test", "https://two.test"])
        self.assertIn("Additional context " * 100, facts[0]["evidence_excerpt"])
        self.assertIn("Conflicting source details", facts[0]["evidence_excerpt"])
        self.assertNotIn("status", facts[0])

    def test_unrelated_trace_and_old_facts_do_not_change_packet(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "facts.json")
            commit(store, [add("Unrelated"),
                           add("Different claim"),
                           add("A supported fact"), add("Direct ID")])
            ids = [f["id"] for f in store.snapshot()["facts"]]
            trace = record([fact(fact_ids=[ids[-1]], evidence_excerpt="short quote")])
            small = compact_packet(trace, store.snapshot(), 2)
            trace["natural_language_plan"] = "DO_NOT_SEND_PLAN" * 10000
            trace["source"] = "DO_NOT_SEND_CODE" * 10000
            trace["worker_evidence"].append({"job_id": "worker", "name": "evolution_tool_evidence",
                                             "data": {"result": "DO_NOT_SEND_WEBPAGE" * 100000}})
            trace["history"] = {"facts": [fact("Nested trap")]}
            large = compact_packet(trace, store.snapshot(), 2)
            self.assertEqual(small, large)
            self.assertEqual({f["id"] for f in large["related_existing_facts"]}, set(ids[2:]))
            self.assertNotIn("Unrelated", json.dumps(large))

    def test_legacy_labels_never_reach_maintenance_or_create_active_tags(self):
        trace = record([fact(status="active", candidate_ids=["legacy"])])
        trace["candidate_proposal"] = {"facts": [fact("Legacy proposal", status="active")]}
        facts, _ = collect_facts(trace, 1)
        self.assertEqual(len(facts), 2)
        for item in facts:
            self.assertNotIn("status", item)
            self.assertNotIn("candidate_ids", item)
            self.assertNotIn("tagged_active", item)
        self.assertEqual(facts[1]["trace_refs"], ["round:1:proposal:facts"])

    def test_excerpt_required_and_never_truncated(self):
        from agent_harness.orchestration.agent_executor import parse_role_output
        from agent_harness.orchestration.exceptions import InvalidRoleOutput
        schema = worker_definition(next(r for r in BUILTIN_ORCHESTRATION_ROLES if r.role_id == "researcher")).result_schema
        for excerpt in (None, "", "  "):
            item = fact(evidence_excerpt=excerpt)
            if excerpt is None:
                item.pop("evidence_excerpt")
            with self.subTest(excerpt=excerpt), self.assertRaises(InvalidRoleOutput):
                parse_role_output(json.dumps({"result": "Answer", "facts": [item]}), schema)
            collected, diagnostics = collect_facts(record([item]), 1)
            self.assertEqual(collected, [])
            self.assertEqual(diagnostics["invalid_facts"], 1)
        excerpt = "Concrete evidence with dates and qualifications. " * 100
        facts, _ = collect_facts(record([fact(evidence_excerpt=excerpt)]), 1)
        self.assertEqual(facts[0]["evidence_excerpt"], excerpt)

    def test_more_than_64_facts_are_not_truncated(self):
        values, _ = collect_facts(record([fact(str(i)) for i in range(100)]), 1)
        self.assertEqual(len(values), 100)

    def test_worker_envelopes_keep_role_result_schema(self):
        expected = {"decomposer": ["Check constraint"], "researcher": "Evidence", "solver": "Answer",
                    "verifier": {"accepted": True, "critique": "Matches"}, "synthesizer": "Answer"}
        for role in BUILTIN_ORCHESTRATION_ROLES:
            if role.role_id not in expected:
                continue
            wrapped = worker_definition(role)
            value = parse_role_output(json.dumps({"result": expected[role.role_id], "facts": [fact()]}),
                                      wrapped.result_schema, role=role.role_id)
            self.assertEqual(value["result"], expected[role.role_id])
            self.assertNotIn("x-fusion-facts", role.result_schema or {})


class ExplicitSubmissionIntegrationTests(unittest.IsolatedAsyncioTestCase):
    run_case = fixtures.FusionIntegrationTests.run_case

    async def test_no_submission_skips_empty_terminal_table(self):
        class NoFactsLLM(ScriptedLLM):
            async def chat(self, messages, **kwargs):
                if messages[0]["content"].startswith(EVIDENCE_ROLE.system_prompt):
                    raise AssertionError("Empty table must skip maintenance")
                return await super().chat(messages, **kwargs)
        llm = NoFactsLLM(); llm.submit_facts = False
        with tempfile.TemporaryDirectory() as tmp:
            _, manifest, _ = await self.run_case(Path(tmp), llm)
            self.assertTrue(manifest["accepted"])
            self.assertEqual(manifest["fact_store_revision"], 0)
            self.assertTrue(manifest["rounds"][0]["evidence_maintenance"]["result"]["skipped"])

    async def test_prose_and_proposal_facts_are_not_imported(self):
        llm = ScriptedLLM(); llm.submit_facts = False
        llm.source = llm.source.replace('"facts": []',
            '\"facts\": [{\"claim\": \"Unsubmitted assertion\", \"source_urls\": [], \"evidence_excerpt\": \"text\"}]')
        with tempfile.TemporaryDirectory() as tmp:
            _, manifest, _ = await self.run_case(Path(tmp), llm)
            self.assertTrue(manifest["accepted"])
            self.assertEqual(manifest["fact_memory"], [])

    async def test_committed_fact_survives_later_worker_failure(self):
        from tests.orchestration.test_fact_submission import SubmissionLLM
        class FailingLLM(SubmissionLLM):
            async def chat(self, messages, **kwargs):
                if (messages[0]["content"].startswith("Investigate")
                        and any(m.get("role") == "tool" for m in messages)):
                    raise RuntimeError("Worker failed after commit")
                return await super().chat(messages, **kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "facts.json")
            async with OfflineSession(FailingLLM()):
                runtime = ScriptRuntime(AgentHarnessWorkerExecutor.from_runtime_services(
                    evidence_scope=EvidenceScope(store, 1)))
                async def solve(ctx):
                    return await ctx.call("researcher", "Task")
                with self.assertRaises(JobFailed):
                    await runtime.run(solve, task="Task")
            self.assertEqual(len(store.snapshot()["facts"]), 1)
            self.assertEqual(store.snapshot()["fact_index"][0]["summary"], "Worker summary")

    async def test_business_output_repair_uses_original_verifier_schema(self):
        class RepairLLM(ScriptedLLM):
            async def chat(self, messages, **kwargs):
                system = messages[0]["content"]
                if system.startswith("Independently verify"):
                    return LLMResponse(content='{\"accepted\": \"invalid\"}', model=self.model)
                if system.startswith("Repair only the response format"):
                    self.repair_schema = messages[-1]["content"]
                    return LLMResponse(content='{\"accepted\": true, \"critique\": \"Supported\"}', model=self.model)
                return await super().chat(messages, **kwargs)
        llm = RepairLLM()
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "facts.json")
            async with OfflineSession(llm):
                runtime = ScriptRuntime(AgentHarnessWorkerExecutor.from_runtime_services(
                    evidence_scope=EvidenceScope(store, 1)))
                async def solve(ctx):
                    return await ctx.call("verifier", "Check candidate")
                result = await runtime.run(solve, task="Task")
            self.assertEqual(result.value, {"accepted": True, "critique": "Supported"})
            self.assertNotIn("x-fusion-facts", llm.repair_schema)
            self.assertEqual(store.snapshot()["facts"], [])

    async def test_forced_finalization_returns_plain_business_result(self):
        class ForcedLLM(ScriptedLLM):
            async def chat(self, messages, **kwargs):
                if "tool budget is exhausted" in messages[-1]["content"]:
                    return LLMResponse(content="Forced answer", model=self.model)
                return LLMResponse(content="Search", model=self.model, tool_calls=[{
                    "id": "search", "type": "function", "function": {
                        "name": "web_search", "arguments": '{\"q\": \"evidence\"}'}}])
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "facts.json")
            async with OfflineSession(ForcedLLM()):
                runtime = ScriptRuntime(AgentHarnessWorkerExecutor.from_runtime_services(
                    evidence_scope=EvidenceScope(store, 1)))
                async def solve(ctx):
                    return await ctx.call("researcher", "Task", max_turns=1)
                result = await runtime.run(solve, task="Task")
            self.assertEqual(result.value, "Forced answer")
            self.assertEqual(store.snapshot()["facts"], [])
