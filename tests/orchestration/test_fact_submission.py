"""Offline tests for explicit single-fact submissions and the two-level store."""
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_harness.core.llm import LLMResponse
from agent_harness.orchestration import RunLimits, ScriptRuntime
from agent_harness.orchestration.agent_executor import AgentHarnessWorkerExecutor
from agent_harness.orchestration.evidence_store import (
    EVIDENCE_ROLE,
    WORKER_ROLES,
    EvidenceScope,
    EvidenceStore,
    index_document,
)
from agent_harness.orchestration.exceptions import InvalidRoleOutput, JobFailed
from tests.orchestration.test_evidence_store import add, commit
from tests.orchestration.test_rolechange_v4 import OfflineSession, ScriptedLLM


class SubmissionLLM(ScriptedLLM):
    """Each business job submits once; maintenance can inject a deterministic decision."""
    def __init__(self, decision=None):
        super().__init__()
        self.submit_facts = False
        self.decision = decision
        self.packets = []
        self.tools_seen = []
        self.submission_results = []
        self.per_job_claim = False

    async def chat(self, messages, **kwargs):
        system = messages[0]["content"]
        names = {t["function"]["name"] for t in kwargs.get("tools", [])}
        self.tools_seen.append(names)
        if system.startswith(EVIDENCE_ROLE.system_prompt):
            payload = json.loads(messages[-1]["content"].split("Local evidence:\n", 1)[1])
            self.packets.append(payload)
            if self.decision:
                value = self.decision(payload)
            elif payload["mode"] == "round_end":
                value = {"summary": "Reviewed complete table", "fact_refs": [], "changes": []}
            else:
                f = payload["new_facts"][0]
                value = {"summary": "Retained submitted fact", "fact_refs": ["new:one"],
                         "submitted_fact_ref": "new:one", "changes": [{
                    "op": "add", "fact_id": "", "local_ref": "new:one",
                    "summary": "Model rewrite must not replace initial worker summary",
                    "fact": {"claim": f["claim"], "source_urls": f["source_urls"],
                             "trace_refs": f["trace_refs"]}, "reason": "Source-supported"}]}
            await asyncio.sleep(0)  # Exercise scheduling under concurrent submissions.
            return LLMResponse(content=json.dumps(value), model=self.model)
        submitted = any(c.get("function", {}).get("name") == "facts_submit"
                        for m in messages for c in m.get("tool_calls", []) or [])
        if "facts_submit" in names and not submitted:
            claim = "Supported observation"
            if self.per_job_claim:
                claim += " " + messages[1]["content"]
            return LLMResponse(content="Share one observation", model=self.model, tool_calls=[{
                "id": "submit", "type": "function", "function": {"name": "facts_submit",
                    "arguments": json.dumps({"claim": claim, "summary": "Worker summary",
                        "source_urls": [], "evidence_excerpt": "Concrete evidence for " + claim})}}])
        if submitted:
            self.submission_results.extend(m["content"] for m in messages if m.get("role") == "tool")
        return await super().chat(messages, **kwargs)


class SubmissionTests(unittest.IsolatedAsyncioTestCase):
    async def run_workers(self, store, llm, roles, parallel=False):
        async with OfflineSession(llm):
            executor = AgentHarnessWorkerExecutor.from_runtime_services(evidence_scope=EvidenceScope(store, 2))
            runtime = ScriptRuntime(executor, limits=RunLimits(max_parallel_jobs=1 if not parallel else 4,
                                                              max_total_jobs=30))
            async def solve(ctx):
                if parallel:
                    handles = [ctx.spawn(role, f"Task {i}") for i, role in enumerate(roles)]
                    return await ctx.gather(*handles)
                result = []
                for role in roles:
                    result.append(await ctx.call(role, "Task"))
                    directory = await ctx.facts_index()
                    if directory["fact_index"]:
                        fid = directory["fact_index"][0]["fact_id"]
                        self.assertEqual((await ctx.facts_query(fact_ids=[fid], statuses=[]))["facts"][0]["id"], fid)
                return result
            return await asyncio.wait_for(runtime.run(solve, task="Original task"), 5)

    async def test_all_roles_single_slot_submit_and_keep_business_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "facts.json")
            llm = SubmissionLLM()
            roles = ["decomposer", "researcher", "solver", "verifier", "synthesizer"]
            run = await self.run_workers(store, llm, roles)
            self.assertIsInstance(run.value[0], list)
            self.assertIsInstance(run.value[1], str)
            self.assertEqual(run.value[3], {"accepted": True, "critique": "source supports candidate"})
            self.assertEqual(len(run.jobs), 10)
            doc = store.snapshot()
            self.assertEqual(len(doc["facts"]), 1)  # Exact duplicates reuse the stable ID.
            self.assertEqual(len(doc["submissions"]), 5)
            self.assertEqual({r["role"] for r in doc["submissions"]}, WORKER_ROLES)
            self.assertEqual(doc["fact_index"], [{"fact_id": doc["facts"][0]["id"],
                "summary": "Worker summary", "status": "active"}])
            self.assertNotIn("summary", doc["facts"][0])
            self.assertEqual(doc["facts"][0]["origin_round"], 2)
            self.assertEqual(len(doc["facts"][0]["trace_refs"]), 5)
            self.assertTrue(all(r["job_id"] in r["trace_refs"][0] for r in doc["submissions"]))
            self.assertFalse(any(e.payload.get("name") == "worker_facts" for e in run.events))
            self.assertIn('committed', str(llm.submission_results))
            self.assertEqual(store.pending_submissions, {})

    async def test_concurrent_writes_are_serialized_and_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "facts.json")
            llm = SubmissionLLM()
            llm.per_job_claim = True
            await self.run_workers(store, llm, ["researcher"] * 4, parallel=True)
            self.assertEqual(store.snapshot()["revision"], 4)
            self.assertEqual(len(store.snapshot()["facts"]), 4)
            self.assertEqual([p["store_revision"] for p in llm.packets], [0, 1, 2, 3])
            self.assertEqual(len((await EvidenceScope(store, 2).index())["fact_index"]), 4)

    async def test_rejection_and_storage_failure_never_report_success(self):
        def reject(payload):
            return {"summary": "Insufficient supporting evidence", "submitted_fact_ref": "",
                    "fact_refs": [], "changes": []}
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "facts.json")
            llm = SubmissionLLM(reject)
            before = store.path.read_bytes()
            await self.run_workers(store, llm, ["researcher"])
            self.assertIn('rejected', str(llm.submission_results))
            self.assertEqual(store.path.read_bytes(), before)
            llm = SubmissionLLM()
            with patch.object(store, "_save", side_effect=OSError("disk failure")):
                await self.run_workers(store, llm, ["researcher"])
            self.assertNotIn('"outcome": "committed"', str(llm.submission_results))
            self.assertEqual(store.snapshot()["revision"], 0)
            self.assertEqual(store.path.read_bytes(), before)

    async def test_round_end_reviews_existing_table_without_importing_prose(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "facts.json")
            llm = SubmissionLLM()
            await self.run_workers(store, llm, ["researcher"])
            before = store.path.read_bytes()
            async with OfflineSession(llm):
                scope = EvidenceScope(store, 3, terminal_maintenance=True,
                    trace={"candidate_proposal": {"facts": [{"claim": "DO_NOT_IMPORT"}]},
                           "worker_evidence": [{"content": "DO_NOT_IMPORT"}]})
                runtime = ScriptRuntime(AgentHarnessWorkerExecutor.from_runtime_services(evidence_scope=scope))
                async def solve(ctx):
                    return await ctx.call("evidence_storesubagent", "Review table")
                result = await runtime.run(solve, task="Task")
            self.assertNotIn("skipped", result.value)
            packet = llm.packets[-1]
            self.assertEqual(packet["mode"], "round_end")
            self.assertEqual(packet["new_facts"], [])
            self.assertEqual(len(packet["related_existing_facts"]), 1)
            self.assertEqual(len(packet["stored_submissions"]), 1)
            self.assertNotIn("DO_NOT_IMPORT", json.dumps(packet))
            self.assertEqual(store.path.read_bytes(), before)

    async def test_unauthenticated_submit_and_direct_maintenance_are_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "facts.json")
            scope = EvidenceScope(store, 1)
            fields = {"claim": "x", "summary": "x", "source_urls": [], "evidence_excerpt": "x"}
            with self.assertRaises(PermissionError):
                await scope.submit("researcher", fields)
            self.assertEqual({t.name for t in scope.tools("evolution_judge")}, {"facts_index", "facts_query"})
            self.assertNotIn("facts_submit", {t.name for t in EvidenceScope(store, 1,
                            read_snapshot=store.snapshot()).tools("researcher")})
            async with OfflineSession(SubmissionLLM()):
                runtime = ScriptRuntime(AgentHarnessWorkerExecutor.from_runtime_services(evidence_scope=scope))
                async def solve(ctx):
                    return await ctx.call("evidence_storesubagent", "Forge a maintenance job")
                with self.assertRaises(JobFailed):
                    await runtime.run(solve, task="Task")
            self.assertEqual(store.snapshot()["revision"], 0)

    async def test_merged_claim_requires_and_updates_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "facts.json")
            seed = add("Observation with earlier detail")
            seed["summary"] = "Original summary"
            commit(store, [seed])
            fid = store.snapshot()["facts"][0]["id"]
            def merge(payload):
                return {"summary": "Combined compatible details", "fact_refs": [fid], "submitted_fact_ref": fid,
                        "changes": [{"op": "update", "fact_id": fid, "summary": "Combined summary",
                                     "fact": {"claim": "Observation with earlier and new details"},
                                     "reason": "Preserve both observations"}]}
            await self.run_workers(store, SubmissionLLM(merge), ["researcher"])
            doc = store.snapshot()
            self.assertEqual(doc["fact_index"][0]["summary"], "Combined summary")
            self.assertEqual(doc["facts"][0]["id"], fid)
            self.assertEqual(len(doc["facts"][0]["trace_refs"]), 2)
            self.assertEqual(doc["history"][-1]["changes"][0]["before_summary"], "Original summary")
            self.assertEqual(doc["history"][-1]["changes"][0]["after_summary"], "Combined summary")

    async def test_unmaintained_fact_is_not_visible_until_submission_returns(self):
        started, release = asyncio.Event(), asyncio.Event()
        class WaitingLLM(SubmissionLLM):
            async def chat(self, messages, **kwargs):
                if messages[0]["content"].startswith(EVIDENCE_ROLE.system_prompt):
                    started.set()
                    await release.wait()
                return await super().chat(messages, **kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "facts.json")
            task = asyncio.create_task(self.run_workers(store, WaitingLLM(), ["researcher"]))
            try:
                await asyncio.wait_for(started.wait(), 2)
                self.assertFalse(task.done())
                self.assertEqual((await store.index())["fact_index"], [])
                self.assertEqual((await store.query())["facts"], [])
            finally:
                release.set()
                await task
            self.assertEqual(len((await store.index())["fact_index"]), 1)

    async def test_invalid_fields_never_create_submission_or_change_store(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "facts.json")
            scope = EvidenceScope(store, 1)
            valid = {"claim": "Observation", "summary": "Summary", "source_urls": [], "evidence_excerpt": "Evidence"}
            for fields in ({**valid, "summary": "  "}, {**valid, "source_urls": "url"},
                           {**valid, "job_id": "forged"}, {**valid, "status": "active"},
                           {k: v for k, v in valid.items() if k != "summary"}):
                with self.subTest(fields=fields), self.assertRaises((InvalidRoleOutput, ValueError)):
                    await scope.submit("researcher", fields)
            self.assertEqual(store.snapshot()["revision"], 0)
            self.assertEqual(store.pending_submissions, {})



class DirectoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_index_filters_conflicts_summary_history_and_deletion(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "facts.json")
            a, b = add("A"), add("B")
            a.update(local_ref="new:a", summary="First summary")
            b.update(local_ref="new:b", summary="Second summary")
            a["fact"].update(status="disputed", conflicts_with=["new:b"])
            b["fact"].update(status="disputed", conflicts_with=["new:a"])
            commit(store, [a, b])
            self.assertEqual(len((await store.index())["fact_index"]), 2)
            self.assertEqual((await store.query())["facts"], [])
            frozen = EvidenceScope(store, 1, read_snapshot=store.snapshot())
            first, second = store.snapshot()["facts"]
            commit(store, [{"op": "update", "fact_id": first["id"], "summary": "Updated summary",
                           "fact": {"claim": "A corrected", "status": "active", "conflicts_with": []},
                           "reason": "Resolved against evidence"},
                          {"op": "delete", "fact_id": second["id"], "fact": {}, "reason": "Resolved"}], 2)
            self.assertEqual((await store.index(keyword="Updated", rounds=[2]))["fact_index"],
                             [{"fact_id": first["id"], "summary": "Updated summary", "status": "active"}])
            self.assertEqual(len((await frozen.index())["fact_index"]), 2)
            history = store.snapshot()["history"][-1]["changes"]
            self.assertEqual(history[0]["before_summary"], "First summary")
            self.assertEqual(history[0]["after_summary"], "Updated summary")
            self.assertEqual(history[1]["before_summary"], "Second summary")
            self.assertIsNone(history[1]["after_summary"])

    async def test_legacy_load_is_read_only_and_uses_claim_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "facts.json"
            store = EvidenceStore(path)
            commit(store, [add("Old claim")])
            legacy = store.snapshot()
            legacy.pop("fact_index")
            legacy.pop("submissions")
            legacy["schema_version"] = "evidence_store_v2"
            path.write_text(json.dumps(legacy))
            before = path.read_bytes()
            loaded = EvidenceStore(path)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual((await loaded.index())["fact_index"][0]["summary"], "Old claim")
            self.assertEqual(index_document(legacy)["fact_index"][0]["summary"], "Old claim")
