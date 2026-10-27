"""Current-table invariants and reference validation across replacement transactions."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_harness.orchestration.agent_executor import parse_role_output
from agent_harness.orchestration.compact_facts import worker_definition
from agent_harness.orchestration.evidence_store import EVIDENCE_ROLE, EvidenceStore
from agent_harness.orchestration.exceptions import InvalidRoleOutput
from agent_harness.orchestration.roles import BUILTIN_ORCHESTRATION_ROLES
from tests.orchestration.test_evidence_store import add, commit


def update(fid, **fields):
    return {"op": "update", "fact_id": fid, "fact": fields, "reason": "Reconciled evidence"}


def delete(fid):
    return {"op": "delete", "fact_id": fid, "fact": {}, "reason": "Replaced by supported evidence"}


class FactTransactionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = EvidenceStore(Path(self.tmp.name) / "facts.json")

    def prepare(self, **value):
        return self.store._prepare_result(
            {"summary": "", "fact_refs": [], "changes": [], **value},
            round_number=2, job_id="maintenance", observed_urls=set(),
            trace_refs={"round:1:trace", "round:2:trace"},
        )

    def disputed_pair(self):
        commit(self.store, [
            {**add("First account", status="disputed", conflicts_with=["new:b"]), "local_ref": "new:a"},
            {**add("Second account", status="disputed", conflicts_with=["new:a"]), "local_ref": "new:b"},
        ])
        return [f["id"] for f in self.store.snapshot()["facts"]]

    def test_delete_requires_resolution_of_surviving_conflicts_and_is_atomic(self):
        first, second = self.disputed_pair()
        before = self.store.snapshot()
        persisted = self.store.path.read_bytes()
        for changes in ([delete(first)], [delete(first), update(second, conflicts_with=[])],
                        [update(first, status="active", conflicts_with=[])]):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.prepare(changes=changes)
            self.assertEqual(self.store.snapshot(), before)
            self.assertEqual(self.store.path.read_bytes(), persisted)
        document, result = self.prepare(
            summary=f"[{first}] was disproven; retain [{second}]",
            fact_refs=[second], changes=[delete(first), update(second, status="active", conflicts_with=[])],
        )
        self.store._publish(document)
        self.assertEqual([f["id"] for f in document["facts"]], [second])
        self.assertEqual(result["fact_refs"], [second])
        self.assertNotIn("facts", result)
        self.assertIn(first, result["summary"])
        history = document["history"][-1]["changes"]
        self.assertEqual(history[0]["before"], before["facts"][0])
        self.assertIsNone(history[0]["after"])
        self.assertEqual(history[1]["after"]["status"], "active")

    def test_replacement_and_merge_remove_old_records_and_return_only_live_refs(self):
        commit(self.store, [add("Old one"), add("Old two")])
        old = [f["id"] for f in self.store.snapshot()["facts"]]
        document, result = self.prepare(
            summary=f"Merged [{old[0]}] and [{old[1]}] into [new:merged]",
            fact_refs=["new:merged"],
            changes=[{**add("Combined supported claim"), "local_ref": "new:merged"}, *map(delete, old)],
        )
        self.assertEqual(len(document["facts"]), 1)
        self.assertEqual(result["fact_refs"], [document["facts"][0]["id"]])
        self.assertEqual(len(result["changes"]), 3)
        self.assertTrue(all(e["after"] is None for e in document["history"][-1]["changes"][1:]))
        self.store._publish(document)
        self.assertTrue(all(f["id"] not in old for f in self.store.snapshot()["facts"]))

    def test_deleted_and_prose_refs_cannot_be_returned_even_when_claim_matches(self):
        commit(self.store, [add("An exact claim")])
        fid = self.store.snapshot()["facts"][0]["id"]
        before = self.store.path.read_bytes()
        for refs, changes in [([fid], [delete(fid)]), (["An exact claim"], []),
                              (["new:temporary"], [
                                  {**add("An exact claim"), "local_ref": "new:temporary"}, delete(fid)])]:
            with self.subTest(refs=refs), self.assertRaisesRegex(ValueError, r"fact_refs\[0\]"):
                self.prepare(fact_refs=refs, changes=changes)
            self.assertEqual(self.store.path.read_bytes(), before)
        document, result = self.prepare(summary=f"Removed [{fid}]", changes=[delete(fid)])
        self.assertEqual(result["fact_refs"], [])
        self.assertEqual(document["facts"], [])
        self.store._publish(document)
        with self.assertRaisesRegex(ValueError, "summary"):
            self.prepare(summary=f"Old deletion [{fid}]")

    def test_invalid_states_and_relations_roll_back(self):
        for fields in [dict(status="superseded"), dict(status="deleted"), dict(supersedes=[]),
                       dict(candidate_ids=[]), dict(status="disputed"),
                       dict(status="active", conflicts_with=["new:b"])]:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                commit(self.store, [add("Uncommitted"), add("Invalid", **fields)])
            self.assertEqual(self.store.snapshot()["revision"], 0)
        with self.assertRaisesRegex(ValueError, "reciprocal"):
            commit(self.store, [
                {**add("A", status="disputed", conflicts_with=["new:b"]), "local_ref": "new:a"},
                {**add("B", status="disputed", conflicts_with=["new:c"]), "local_ref": "new:b"},
                {**add("C", status="disputed", conflicts_with=["new:b"]), "local_ref": "new:c"},
            ])
        self.assertEqual(self.store.snapshot()["facts"], [])

    def test_duplicate_updates_then_delete_do_not_resurrect_a_record(self):
        commit(self.store, [add("Repeated claim")])
        fid = self.store.snapshot()["facts"][0]["id"]
        document, result = self.prepare(changes=[
            {**add("repeated CLAIM"), "local_ref": "new:duplicate"},
            update(fid, claim="Updated claim"), delete(fid),
        ])
        self.assertEqual(document["facts"], [])
        self.assertEqual(result["fact_refs"], [])
        self.assertEqual(document["history"][-1]["changes"][-1]["before"]["claim"], "Updated claim")

    def test_worker_and_maintenance_schemas_reject_removed_fields(self):
        worker = worker_definition(next(r for r in BUILTIN_ORCHESTRATION_ROLES if r.role_id == "researcher"))
        for field, value in (("status", "active"), ("candidate_ids", []), ("conflicts_with", [])):
            fact = {"claim": "Observed claim", "source_urls": [], "evidence_excerpt": "Concrete context", field: value}
            with self.subTest(field=field), self.assertRaises(InvalidRoleOutput):
                parse_role_output(json.dumps({"result": "Answer", "facts": [fact]}), worker.result_schema)
        with self.assertRaises(InvalidRoleOutput):
            parse_role_output('{"summary":"", "facts":[], "changes":[]}', EVIDENCE_ROLE.result_schema)
