"""Flow observation follows real authenticated worker and transaction boundaries."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent_harness.orchestration import AgentHarnessWorkerExecutor, RunLimits, ScriptRuntime
from agent_harness.orchestration.evidence_store import EvidenceScope, EvidenceStore
from agent_harness.orchestration.fact_flow import FactFlowLedger
from tests.orchestration.test_fact_submission import SubmissionLLM
from tests.orchestration.test_rolechange_v4 import OfflineSession


class FlowTests(unittest.IsolatedAsyncioTestCase):
    async def execute(self, root, llm):
        store = EvidenceStore(root / "facts.json")
        ledger = FactFlowLedger(root, "attempt")
        scope = EvidenceScope(store, 1, flow_ledger=ledger)
        async with OfflineSession(llm):
            executor = AgentHarnessWorkerExecutor.from_runtime_services(evidence_scope=scope)
            runtime = ScriptRuntime(executor, limits=RunLimits(max_parallel_jobs=4, max_total_jobs=30))
            async def solve(ctx):
                await ctx.gather(ctx.spawn("researcher", "one"), ctx.spawn("researcher", "two"))
                directory = await ctx.facts_index()
                ids = [f["fact_id"] for f in directory["fact_index"]]
                await ctx.facts_query(fact_ids=ids, statuses=[])
                instruction = "Use " + " ".join(ids)
                await ctx.call("solver", instruction, key="same")
                await ctx.call("solver", instruction, key="same")
                return ids
            result = await runtime.run(solve, task="t", run_id="flow-test")
        return store, ledger, result

    async def test_authenticated_concurrent_lifecycle_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, ledger, _result = await self.execute(Path(tmp), SubmissionLLM())
            events = ledger.events
            committed = [e for e in events if e["event_type"] == "COMMITTED"]
            self.assertEqual(len(committed), 3)
            self.assertEqual(len({e["source_job_id"] for e in committed}), 3)
            self.assertEqual(len(store.snapshot()["facts"]), 1)
            transfers = [e for e in events if e["event_type"] == "TRANSFERRED" and e["role"] == "solver"]
            self.assertEqual(len(transfers), 1)
            reads = [e for e in events if e["event_type"] in {"READ_FULL", "READ_SUMMARY"}]
            self.assertEqual(len(reads), 2)
            self.assertTrue(all(e["role"] == "orchestration" and e["target_job_id"] is None for e in reads))
            self.assertTrue(all(e["revision"] == 2 for e in reads))
            self.assertEqual(len(FactFlowLedger(Path(tmp), "attempt").events), len(events))

    async def test_rejection_is_not_commit(self):
        def reject(packet):
            return {"summary": "Rejected", "submitted_fact_ref": "", "fact_refs": [], "changes": []}
        with tempfile.TemporaryDirectory() as tmp:
            _, ledger, _ = await self.execute(Path(tmp), SubmissionLLM(reject))
            self.assertEqual(sum(e["event_type"] == "REJECTED_SUBMISSION" for e in ledger.events), 3)
            self.assertFalse(any(e["event_type"] == "COMMITTED" for e in ledger.events))

    async def test_exact_tokens_freeze_and_observation_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = FactFlowLedger(Path(tmp), "a")
            req = SimpleNamespace(instruction="fact-abc-long xfact-abc fact-abc", run_id="r",
                                  job_id="j", parent_job_id=None, role="solver")
            ledger.transfer(req, {"revision": 1, "facts": [{"id": "fact-abc"}]}, 1, "execution")
            ledger.transfer(req, {"revision": 1, "facts": [{"id": "fact-abc"}]}, 1, "execution")
            self.assertEqual(len(ledger.events), 1)
            frozen = ledger.freeze(1)
            with patch.object(Path, "open", side_effect=OSError("disk full")):
                ledger.record("READ_FULL", round_number=2, fact_id="fact-abc")
            self.assertEqual(len(frozen["events"]), 1)
            self.assertEqual(len(ledger.freeze(1)["events"]), 1)
            self.assertTrue(ledger.errors)

    async def test_publish_failure_cannot_emit_committed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(EvidenceStore, "_publish", side_effect=OSError("transaction failed")):
                await self.execute(Path(tmp), SubmissionLLM())
            ledger = FactFlowLedger(Path(tmp), "attempt")
            self.assertTrue(any(e["event_type"] == "SUBMITTED" for e in ledger.events))
            self.assertFalse(any(e["event_type"] in {"COMMITTED", "REJECTED_SUBMISSION"} for e in ledger.events))

    async def test_attempt_mismatch_never_mix_or_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = FactFlowLedger(Path(tmp), "first")
            first.record("SUBMITTED", round_number=1)
            path = Path(tmp) / "fact_flow/events.jsonl"
            original = path.read_bytes()
            second = FactFlowLedger(Path(tmp), "second")
            self.assertEqual(second.events, [])
            self.assertTrue(second.errors)
            second.record("READ_FULL", round_number=1)
            self.assertEqual(original, path.read_bytes())

    async def test_worker_reads_keep_identity_and_filter_revision(self):
        from agent_harness.orchestration.types import AgentResult
        from tests.orchestration.test_evidence_store import add, commit
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "facts.json")
            commit(store, [add("supported")])
            fid = store.snapshot()["facts"][0]["id"]
            ledger = FactFlowLedger(Path(tmp), "a")
            scope = EvidenceScope(store, 1, flow_ledger=ledger)
            class Reader:
                evidence_scope = scope
                def validate_role(self, role):
                    pass
                async def execute(self, request, emit):
                    await scope.query(fact_ids=[fid, "fact-missing"])
                    return AgentResult(value=request.job_id)
            runtime = ScriptRuntime(Reader())
            async def solve(ctx):
                return await ctx.gather(ctx.spawn("solver", "s"), ctx.spawn("verifier", "v"))
            result = await runtime.run(solve, task="t")
            reads = ledger.events
            self.assertEqual({e["target_job_id"] for e in reads}, set(result.value))
            self.assertEqual({e["role"] for e in reads}, {"solver", "verifier"})
            self.assertEqual({e["fact_id"] for e in reads}, {fid})
            self.assertEqual({e["revision"] for e in reads}, {1})

    async def test_dispute_resolution_and_deletion_after_real_maintenance(self):
        from tests.orchestration.test_evidence_store import add, commit
        with tempfile.TemporaryDirectory() as tmp:
            store = EvidenceStore(Path(tmp) / "facts.json")
            commit(store, [add("one"), add("two")])
            ids = [f["id"] for f in store.snapshot()["facts"]]
            ledger = FactFlowLedger(Path(tmp), "a")
            async def maintenance(changes, number):
                def decision(packet):
                    return {"summary": "Reconciled", "fact_refs": [], "changes": changes}
                scope = EvidenceScope(store, number, terminal_maintenance=True, flow_ledger=ledger)
                async with OfflineSession(SubmissionLLM(decision)):
                    executor = AgentHarnessWorkerExecutor.from_runtime_services(evidence_scope=scope)
                    runtime = ScriptRuntime(executor)
                    async def solve(ctx):
                        return await ctx.call("evidence_storesubagent", "Maintain")
                    await runtime.run(solve, task="t")
            await maintenance([{"op": "update", "fact_id": fid, "fact": {
                "status": "disputed", "conflicts_with": [ids[1-i]]}, "reason": "Conflict"}
                for i,fid in enumerate(ids)], 1)
            self.assertEqual(sum(e["event_type"] == "DISPUTED" for e in ledger.events), 2)
            await maintenance([{"op": "update", "fact_id": fid, "fact": {
                "status": "active", "conflicts_with": []}, "reason": "Resolved"} for fid in ids], 2)
            self.assertEqual(sum(e["event_type"] == "UPDATED" for e in ledger.events), 2)
            await maintenance([{"op": "delete", "fact_id": ids[0], "fact": {}, "reason": "Merged"}], 3)
            deleted = [e for e in ledger.events if e["event_type"] == "DELETED"]
            self.assertEqual(len(deleted), 1)
            self.assertEqual(deleted[0]["revision"], store.snapshot()["revision"])
            self.assertEqual(deleted[0]["details"]["before"]["claim"], "one")
