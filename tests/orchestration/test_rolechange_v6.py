import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.orchestration.test_rolechange_v5 import ReflectingLLM, run_case


class V6Tests(unittest.IsolatedAsyncioTestCase):
    async def test_modes_and_roundtrip(self):
        outputs = []
        for mode in ("off", "flow_only", "slice"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"ROLECHANGE_V6_MODE": mode}):
                root = Path(tmp)
                llm = ReflectingLLM(decisions=("evolve", "approve_synthesis"))
                result, manifest, requests = await run_case(root, llm, strategy="rolechange_self_evolving_v6")
                self.assertEqual(manifest["fact_flow_mode"], mode)
                self.assertEqual(manifest["schema_version"], "self_evolving_orchestration_v6_flow_slice_v1")
                generated = root / "generated"
                self.assertEqual((generated / "fact_flow/events.jsonl").exists(), mode != "off")
                review = generated / "evolution_round_01/terminal_review"
                self.assertEqual((review / "diagnostic_slice.json").exists(), mode == "slice")
                report = json.loads((review / "report.json").read_text())
                self.assertIsNone(report["error"])
                self.assertFalse(any(r.role == "evolution_judge" and r.key == "terminal:assessment" and
                                     "diagnostic_slice" in r.instruction for r in requests))
                outputs.append((manifest["rounds"][-1]["status"], result.get("answer")))
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[1], outputs[2])

    async def test_slice_failure_falls_back_without_breaking_report(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"ROLECHANGE_V6_MODE": "slice"}):
            with patch("agent_harness.orchestration.diagnostic_slice.build_diagnostic_slice", side_effect=ValueError("injected slice failure")):
                await run_case(Path(tmp), ReflectingLLM(decisions=("evolve", "approve_synthesis")),
                               strategy="rolechange_self_evolving_v6")
            review = Path(tmp) / "generated/evolution_round_01/terminal_review"
            bundle = json.loads((review / "diagnostic_bundle.json").read_text())
            self.assertIn("injected slice failure", bundle["diagnostic_slice_summary"]["fallback_reason"])
            self.assertIsNone(json.loads((review / "report.json").read_text())["error"])

    async def test_final_round_stop_and_scope_isolation(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"ROLECHANGE_V6_MODE": "slice"}):
            root = Path(tmp)
            llm = ReflectingLLM(decisions=("evolve", "evolve", "evolve"))
            _, manifest, _requests = await run_case(root, llm, strategy="rolechange_self_evolving_v6")
            report = json.loads((root / "generated/evolution_round_03/failure_report.json").read_text())
            self.assertEqual(report["evolution_mode"], "STOP")
            self.assertFalse(report["continue_evolution"])
            index = json.loads((root / "generated/trace_store/index.json").read_text())
            self.assertFalse(any("/report/" in r["source"] or "report_live" in r["source"] for r in index))
            for rd in (root / "generated").glob("evolution_round_*"):
                stage1 = json.loads((rd / "terminal_review/assessment_input.json").read_text())
                self.assertNotIn("diagnostic_slice", str(stage1))
                self.assertNotIn("fact_flow", str(stage1))
            self.assertEqual(len(manifest["rounds"]), 3)
