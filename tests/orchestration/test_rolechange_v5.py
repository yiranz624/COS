"""V5 mechanism tests exercise real runtime loops with deterministic local models."""
from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent_harness.core.llm import LLMResponse
from agent_harness.orchestration import AgentHarnessWorkerExecutor, BudgetSpec
from agent_harness.orchestration.evidence_judge import REPORT_JUDGE_PROMPT, local_judge_role
from agent_harness.orchestration.reflection import (
    REFLECTOR_PROMPT,
    build_diagnostic_bundle,
    normalize_reflection,
)
from agent_harness.orchestration.trace_store import TraceStore
from benchmarks.runner.rolechange_evolving_runner import run_self_evolving_strategy
from tests.orchestration.test_rolechange_v4 import OfflineSession, ScriptedLLM


class ReflectingLLM(ScriptedLLM):
    def __init__(self, *args, malformed_report=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.retrievals = []
        self.malformed_report = malformed_report
        self.report_inputs = []

    async def chat(self, messages, tools=None, **kwargs):
        system = str(messages[0]["content"])
        if not system.startswith(REFLECTOR_PROMPT):
            return await super().chat(messages, tools=tools, **kwargs)
        self.assert_tools = {t["function"]["name"] for t in tools}
        user = next(m["content"] for m in messages if m["role"] == "user")
        if not self.report_inputs or self.report_inputs[-1] != user:
            self.report_inputs.append(user)
        calls = [c for m in messages for c in m.get("tool_calls", []) or []]
        if not calls:
            name, args = "trace_index", {"stages": ["diagnostics"]}
        elif len(calls) == 1:
            result = ast.literal_eval(next(m["content"] for m in reversed(messages) if m["role"] == "tool"))
            name, args = "trace_query", {"trace_ids": [result["items"][0]["trace_id"]]}
        else:
            if self.malformed_report:
                return LLMResponse(content='{"decision":"evolve"}', model=self.model)
            # Reuse the V4 fixture's ordinary feedback, then add the V5 evidence contract.
            baseline_messages = [{**messages[0], "content": REPORT_JUDGE_PROMPT}, *messages[1:]]
            response = await super().chat(baseline_messages, tools=tools, **kwargs)
            value = json.loads(response.content)
            result = ast.literal_eval(next(m["content"] for m in reversed(messages) if m["role"] == "tool"))
            ref = result["items"][0]["trace_id"]
            value["failure_report"].update(
                root_cause="Candidate failed the recorded evidence gate.", failure_target="VERIFICATION",
                evolution_mode="STOP" if self.round == 3 else "REPLAN", fact_refs=[], trace_refs=[ref],
                diagnostic_hypotheses=[{"hypothesis": "The evidence gate rejected the candidate.",
                                        "verdict": "supported", "fact_refs": [], "trace_refs": [ref]}])
            return LLMResponse(content=json.dumps(value), model=self.model, usage={"total_tokens": 10})
        self.retrievals.append(name)
        return LLMResponse(content="Inspect recorded evidence.", model=self.model, tool_calls=[{
            "id": f"reflection-{self.round}-{len(calls)}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}], usage={"total_tokens": 10})


async def run_case(path, llm, *, strategy="rolechange_self_evolving_v5", failures=()):
    config = SimpleNamespace(planner_model=None, llm_provider="openai", openai_model=llm.model,
                             openai_base_url="https://example.test/v1", planner_max_tokens=8192)
    config.model_copy = lambda **kw: config
    original = AgentHarnessWorkerExecutor.execute
    requests = []

    async def execute(executor, req, emit):
        requests.append(req)
        if (req.role, req.key) in failures:
            raise RuntimeError("Injected execution failure")
        return await original(executor, req, emit)

    with patch("benchmarks.core.kernel_adapter.BenchmarkSession", side_effect=lambda **kw: OfflineSession(llm)), \
         patch("agent_harness.infra.config.get_config", return_value=config), \
         patch("agent_harness.infra.llm_adapter.create_llm", return_value=llm), \
         patch.object(AgentHarnessWorkerExecutor, "execute", execute):
        result = await run_self_evolving_strategy(
            "Identify the requested answer.", strategy=strategy, trial_dir=path, question_id="offline-v5",
            orchestration_profile="deepseek_v4_flash", planner_model=None, planner_max_repair_attempts=0,
            skip_generated_code_review=False, planner_only=False, spec=BudgetSpec(max_jobs=12))
    manifest = json.loads((path / "generated/evolution_manifest.json").read_text())
    return result, manifest, requests


class TraceTests(unittest.IsolatedAsyncioTestCase):
    async def test_identity_freeze_and_long_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = TraceStore(Path(tmp))
            a = store.add(round_number=1, stage="execution", runtime="one", sequence=1, content="a" * 24000)
            b = store.add(round_number=2, stage="execution", runtime="one", sequence=1, content="b")
            c = store.add(round_number=1, stage="execution", runtime="two", sequence=1, content="c")
            self.assertEqual(len({a, b, c}), 3)
            scope = store.freeze(1)
            store.add(round_number=1, stage="report", runtime="report", sequence=1, content="private")
            self.assertEqual(len(scope.records), 3)
            rebuilt = TraceStore(Path(tmp))
            self.assertEqual(a, rebuilt.add(round_number=1, stage="execution", runtime="one", sequence=1, content="a" * 24000))
            parts = []
            offset = 0
            for _ in range(3):
                result = await scope.query(trace_ids=[a], offset=offset)
                item = result["items"][0]
                parts.append(item["content"])
                offset = item["next_offset"]
            self.assertEqual("".join(parts), "a" * 24000)
            self.assertIsNone(offset)
            self.assertEqual((await scope.query(trace_ids=["missing"]))["items"][0]["status"], "missing")

    async def test_budgets_and_filters(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = TraceStore(Path(tmp))
            ids = [store.add(round_number=1, stage="execution", runtime="r", sequence=i,
                             content="x" * 9000, role="solver") for i in range(40)]
            scope = store.freeze(1)
            result = await scope.index(roles=["solver"])
            self.assertEqual(len(result["items"]), 30)
            self.assertEqual(result["next_offset"], 30)
            self.assertEqual((await scope.query(trace_ids=ids[:6]))["status"], "invalid_request")
            result = await scope.query(trace_ids=ids[:5])
            self.assertLessEqual(sum(len(x["content"]) for x in result["items"]), 20000)
            for _ in range(20):
                await scope.query(trace_ids=[ids[0]])
            self.assertLessEqual(scope.queries, 8)
            self.assertLessEqual(scope.returned_chars, 60000)
            self.assertEqual((await scope.query(trace_ids=[ids[0]]))["status"], "budget_exhausted")
            scope = store.freeze(1)
            for _ in range(15):
                await scope.index(keyword="absent")
            self.assertEqual(scope.calls, 12)

    async def test_fact_aliases_and_report_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            trial = Path(tmp)
            root = trial / "generated"
            rd = root / "evolution_round_01"
            rd.mkdir(parents=True)
            event = {"sequence": 1, "job_id": "job", "run_id": "runtime", "事件类型": "job_created",
                     "详细信息": {"role": "researcher", "instruction": "actual input"}}
            (rd / "orchestration_live.jsonl").write_text(json.dumps(event) + '\n')
            terminal = rd / "terminal_review"
            terminal.mkdir()
            (terminal / "report_live.jsonl").write_text(json.dumps({**event, "secret": "own report"}))
            ref = "round:1:job:job:submission-1"
            snapshot = {"facts": [{"id": "F1", "trace_refs": [ref]}], "submissions": [{
                "submission_id": "submission-1", "round": 1, "job_id": "job", "trace_refs": [ref],
                "claim": "Fact claim", "evidence_excerpt": "local evidence"}]}
            store = TraceStore(root)
            store.ingest(trial, 1, snapshot)
            scope = store.freeze(1)
            result = await scope.query(trace_ids=[ref])
            self.assertIn("Fact claim", result["items"][0]["content"])
            self.assertTrue(result["items"][0]["related_ids"])
            self.assertNotIn("own report", json.dumps(scope.records))
            self.assertEqual(len((await scope.index(roles=["researcher"]))["items"]), 1)

    async def test_executor_denies_stage1_trace(self):
        executor = AgentHarnessWorkerExecutor(None, None,
            role_overrides={"evolution_judge": local_judge_role(1)})
        with self.assertRaises(PermissionError):
            await executor.execute_tool("trace_index", {}, role="evolution_judge")

    def test_diagnostic_delta_and_invalid_reference(self):
        bundle = build_diagnostic_bundle(task="t", round_number=2, proposal={}, assessment={}, gate={}, trace={},
            frozen_facts={"facts": [{"id": "a", "claim": "new"}, {"id": "c"}], "revision": 2},
            previous_snapshot={"facts": [{"id": "a", "claim": "old"}, {"id": "b"}]},
            previous_report=None, usage={}, synthesis_error="error", trace_limits={})
        self.assertEqual(bundle["fact_delta"], {"added": ["c"], "removed": ["b"], "updated": ["a"]})
        report = normalize_reflection({}, {"failure_target": "SEARCH", "trace_refs": ["invented"]}, round_number=3)
        self.assertEqual(report["failure_target"], "UNKNOWN")
        self.assertEqual(report["evolution_mode"], "STOP")
        self.assertFalse(report["continue_evolution"])


class V5IntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_reflection_reaches_next_planner(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            llm = ReflectingLLM(decisions=("evolve", "approve_synthesis"))
            _, manifest, requests = await run_case(path, llm)
            self.assertTrue(manifest["accepted"])
            self.assertEqual(manifest["completed_rounds"], 2)
            self.assertEqual(llm.retrievals, ["trace_index", "trace_query"])
            self.assertEqual(len([r for r in requests if r.key == "terminal:evolution_report"]), 1)
            self.assertEqual(llm.assert_tools, {"facts_index", "facts_query", "trace_index", "trace_query"})
            report = json.loads((path / "generated/evolution_round_01/failure_report.json").read_text())
            self.assertEqual(report["evolution_mode"], "REPLAN")
            self.assertTrue(report["trace_refs"])
            prompt = next(text for round_num, kind, text in llm.prompts if round_num == 2 and kind == "plan")
            self.assertIn('"evolution_mode": "REPLAN"', prompt)
            self.assertNotIn('"详细信息"', prompt)
            stage1 = json.loads((path / "generated/evolution_round_01/terminal_review/assessment_input.json").read_text())
            self.assertEqual(set(stage1), {"task", "facts", "candidate_answer"})

    async def test_three_round_stop_and_generation_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = ReflectingLLM(decisions=("evolve",), generation_failures=(1,))
            _, manifest, _ = await run_case(Path(tmp), llm)
            self.assertEqual(manifest["completed_rounds"], 3)
            self.assertEqual(manifest["status"], "best_effort")
            self.assertEqual(manifest["last_failure_report"]["evolution_mode"], "STOP")
            self.assertFalse(manifest["last_failure_report"]["continue_evolution"])
            self.assertIn("generation_status", llm.report_inputs[0])

    async def test_synthesis_failure_is_diagnosable(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = ReflectingLLM()
            _, manifest, _ = await run_case(Path(tmp), llm,
                failures={("synthesizer", "terminal:synthesizer")})
            self.assertEqual(manifest["completed_rounds"], 3)
            bundle = json.loads((Path(tmp) / "generated/evolution_round_01/terminal_review/diagnostic_bundle.json").read_text())
            self.assertIn("Injected", bundle["synthesis_error"])

    async def test_worker_failure_is_diagnosable(self):
        with tempfile.TemporaryDirectory() as tmp:
            llm = ReflectingLLM(decisions=("evolve",))
            _, manifest, _ = await run_case(Path(tmp), llm,
                failures={("researcher", "research")})
            self.assertEqual(manifest["completed_rounds"], 3)
            bundle = json.loads((Path(tmp) / "generated/evolution_round_01/terminal_review/diagnostic_bundle.json").read_text())
            self.assertIn("Injected", bundle["execution_error"])

    def test_report_error_fallback(self):
        report = normalize_reflection({}, {"failure_target": "SEARCH", "evolution_mode": "EXPLORE"},
                                      round_number=1, error="Malformed report")
        self.assertEqual(report["failure_target"], "UNKNOWN")
        self.assertEqual(report["evolution_mode"], "REFINE")
        self.assertEqual(report["report_error"], "Malformed report")
