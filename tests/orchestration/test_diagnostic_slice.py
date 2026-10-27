import tempfile
import unittest
from pathlib import Path

from agent_harness.orchestration.diagnostic_slice import build_diagnostic_slice
from agent_harness.orchestration.fact_flow import FactFlowLedger
from agent_harness.orchestration.trace_store import TraceStore


class SliceTests(unittest.IsolatedAsyncioTestCase):
    def fixture(self, root):
        store = TraceStore(root)
        ledger = FactFlowLedger(root, store.attempt_id)
        for job, role, number in [("producer", "researcher", 1), ("solver", "solver", 2),
                                   ("verifier", "verifier", 2), ("unrelated", "researcher", 2)]:
            store.add(round_number=number, stage="execution", runtime="r", sequence=job,
                      content={"payload": {"role": role}}, job_id=job, role=role, event_type="job_created")
        ledger.record("COMMITTED", round_number=1, fact_id="fact-a", source_job_id="producer")
        ledger.record("READ_FULL", round_number=2, fact_id="fact-a", target_job_id="solver")
        store.ingest_flow(ledger.freeze(2))
        return store, ledger

    def build(self, store, ledger, **kw):
        return build_diagnostic_slice(round_number=2,
            assessment={"constraints": [{"constraint": "answer", "status": "unsatisfied", "fact_ids": ["fact-a"]}]},
            gate={}, proposal={}, frozen_facts={"facts": [{"id": "fact-a", "conflicts_with": []}]},
            fact_flow=ledger.freeze(2), trace_store=store, **kw)

    async def test_dependency_slice_is_stable_and_avoids_unrelated_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, ledger = self.fixture(Path(tmp))
            result = self.build(store, ledger)
            self.assertEqual(result, self.build(store, ledger))
            self.assertIn("producer", result["job_ids"])
            self.assertIn("solver", result["job_ids"])
            self.assertNotIn("unrelated", result["job_ids"])
            self.assertTrue(set(result["trace_refs"]) <= store.records.keys())
            bounded = self.build(store, ledger, limits={"fact_nodes": 0, "trace_refs": 1})
            self.assertTrue(bounded["truncation"]["nodes"])
            self.assertLessEqual(len(bounded["trace_refs"]), 1)

    async def test_expansion_budgets_legacy_and_direct_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, ledger = self.fixture(Path(tmp))
            result = self.build(store, ledger)
            scope = store.freeze(2, preferred_trace_ids=result["trace_refs"])
            inside = await scope.index()
            self.assertTrue({r["trace_id"] for r in inside["items"]} <= scope.slice_ids)
            self.assertEqual((await scope.index(scope="run"))["status"], "invalid_request")
            self.assertEqual((await scope.index(scope="run", expansion_reason="find earlier origin"))["status"], "ok")
            self.assertEqual((await scope.index(rounds=[]))["status"], "ok")
            outside = next(t for t,r in store.records.items() if r["job_id"] == "unrelated")
            self.assertEqual((await scope.query(trace_ids=[outside]))["status"], "invalid_request")
            self.assertEqual((await scope.query(trace_ids=[outside], expansion_reason="test competing hypothesis"))["status"], "ok")
            for _ in range(15):
                await scope.index()
            self.assertEqual(scope.calls, 12)
            self.assertTrue(scope.usage()["round_expansions"])

    async def test_conflict_timeout_and_unlinked_constraint(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, ledger = self.fixture(Path(tmp))
            failure = store.add(round_number=2, stage="execution", runtime="r", sequence="timeout",
                                job_id="timed", role="researcher", event_type="job_timed_out", content={})
            result = build_diagnostic_slice(round_number=2,
                assessment={"constraints": [{"constraint": "unknown clue", "status": "unknown", "fact_ids": []},
                                             {"constraint": "conflict", "status": "unsatisfied", "fact_ids": ["fact-a"]}]},
                gate={}, proposal={}, frozen_facts={"facts": [
                    {"id": "fact-a", "conflicts_with": ["fact-b"]}, {"id": "fact-b", "conflicts_with": ["fact-a"]}]},
                fact_flow=ledger.freeze(2), trace_store=store)
            self.assertIn("fact-b", result["fact_ids"])
            self.assertIn("timed", result["job_ids"])
            self.assertIn(failure, result["trace_refs"])
            self.assertEqual(len(result["unresolved_seeds"]), 1)

    async def test_no_read_is_not_reported_as_information_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, ledger = self.fixture(Path(tmp))
            ledger.events = [e for e in ledger.events if e["event_type"] != "READ_FULL"]
            result = self.build(store, ledger)
            self.assertFalse(any(e["type"] == "READ_FULL" for e in result["edges"]))
            self.assertNotIn("root_cause", result)
            self.assertIn("solver", result["job_ids"])  # Still inspect ordinary input/output.

    async def test_sanitized_agent_trace_identity_and_reingest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            live = source / "generated/evolution_round_01/orchestration_live.jsonl"
            live.parent.mkdir(parents=True)
            import json
            live.write_text(json.dumps({"sequence": 1, "job_id": "run:job:1", "event_type": "job_created",
                                        "payload": {"role": "solver"}}) + "\n")
            agent = source / "agent_traces/evolution_round_01/run_job_1_solver.json"
            agent.parent.mkdir(parents=True)
            agent.write_text('{}')
            store = TraceStore(root / "output")
            store.ingest(source, 1, {}, source_root=source / "generated")
            ids = set(store.records)
            store.ingest(source, 1, {}, source_root=source / "generated")
            self.assertEqual(ids, set(store.records))
            row = next(r for r in store.records.values() if r["stage"] == "agent_trace")
            self.assertEqual(row["job_id"], "run:job:1")
            self.assertEqual(row["role"], "solver")
