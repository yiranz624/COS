"""Offline fusion tests use real planning, execution and outer-loop code."""
from __future__ import annotations

import ast
import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agent_harness.core.llm import LLMResponse
from agent_harness.core.tool import Tool
from agent_harness.core.loop_types import LoopConfig, ToolResult
from agent_harness.core.runtime import registry
from agent_harness.core.runtime.loop.compact import DefaultCompactionPolicy, DefaultMessageCompactor
from agent_harness.core.runtime.registries.agents import AgentRegistry
from agent_harness.core.runtime.resources.manager import ResourceManager
from agent_harness.orchestration.agent_executor import AgentHarnessWorkerExecutor
from agent_harness.orchestration.contracts import BudgetSpec
from agent_harness.orchestration.evolution_judge import normalize_judge_decision, normalize_round_proposal
from agent_harness.orchestration.evolution_roles import EVOLUTION_JUDGE_ROLE
from agent_harness.orchestration.evidence_judge import LOCAL_JUDGE_PROMPT, REPORT_JUDGE_PROMPT, local_judge_role
from agent_harness.orchestration.evidence_store import EVIDENCE_ROLE
from agent_harness.orchestration.generated_code import (
    GeneratedSolveValidator, build_environment_manifest, build_generation_request, preflight_solve,
    _render_generation_prompt, _render_natural_language_plan_prompt,
)
from agent_harness.orchestration.rolechange_evolution import (
    ROLECHANGE_STRATEGY, SCRIPT_ROLES, EvidenceObserver, attach_worker_evidence,
    bind_executed_verifier, validate_proposal_shape,
)
from agent_harness.orchestration.roles import BUILTIN_ORCHESTRATION_ROLES
from agent_harness.orchestration.self_evolving import evolution_context
from benchmarks.runner.rolechange_evolving_runner import run_self_evolving_strategy


SOURCE = '''async def solve(ctx):
    subtasks = await ctx.call("decomposer", "Decompose " + ctx.task, key="decompose")
    evidence = await ctx.call("researcher", "Research " + ctx.task + str(subtasks), key="research")
    candidate = await ctx.call("solver", "Form an answer for " + ctx.task + evidence, key="solve")
    combined = await ctx.call("synthesizer", "Combine supplied evidence: " + evidence, key="combine")
    verdict = await ctx.call("verifier", "Original task: " + ctx.task + " Candidate to verify: " + str(candidate) + " Evidence: " + evidence, key="verify")
    status = "incomplete"
    if verdict.get("accepted", False):
        status = "ready"
    return {"status": status, "selected_candidate": candidate, "selected_verifier": verdict, "facts": [], "issues": [], "synthesis_context": {"evidence": evidence, "intermediate": combined}}
'''


def request():
    definitions = {r.role_id: r for r in (*BUILTIN_ORCHESTRATION_ROLES, EVIDENCE_ROLE) if r.role_id in SCRIPT_ROLES}
    environment = build_environment_manifest(roles=definitions, tools={}, budget=BudgetSpec())
    return build_generation_request("Identify the requested answer.", {
        "prompt_profile": ROLECHANGE_STRATEGY,
        "evolution_context": evolution_context([], None, round_number=1),
    }, environment)


def proposal():
    return {
        "status": "ready", "selected_candidate": "Candidate A",
        "selected_verifier": {"accepted": True, "critique": "source supports candidate"},
        "facts": [], "issues": [], "synthesis_context": {},
    }


def event(kind, payload, job_id="verify", sequence=1):
    return SimpleNamespace(event_type=kind, payload=payload, job_id=job_id, sequence=sequence)


class ScriptedLLM:
    model = "offline-fusion-model"

    def __init__(self, decisions=("approve_synthesis",), generation_failures=(), verifier_accept=True):
        self.round = 0
        self.decisions = decisions
        self.generation_failures = generation_failures
        self.verifier_accept = verifier_accept
        self.prompts = []
        self.judge_prompts = []
        self.research_tool_turns = 0
        self.research_turn = 0
        self.research_inputs = []
        self.source = SOURCE
        self.invalid_judge = False
        self.submit_facts = True

    async def chat(self, messages, **kwargs):
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if "high-level coordinator for a team" in system:
            self.round += 1
            self.prompts.append((self.round, "plan", user))
            content = (
                "First ask the decomposer to identify independent requirements. "
                "Ask the researcher to collect supporting evidence, then the solver to form "
                "the requested answer. Ask the synthesizer to combine evidence, then the "
                "verifier to check the selected candidate. Submit that candidate and its "
                "actual verification for external review."
            )
        elif "independent reviewer of a high-level" in system or "independent compatibility reviewer" in system:
            content = json.dumps({"accepted": True, "critique": "", "issues": [], "suggested_changes": []})
        elif "You are the Orchestrator and orchestration-program author" in system:
            self.prompts.append((self.round, "code", user))
            content = "invalid source" if self.round in self.generation_failures else self.source
        elif system.startswith(LOCAL_JUDGE_PROMPT):
            self.judge_prompts.append(system)
            if self.invalid_judge:
                return LLMResponse(content='{"decision":"approve_synthesis"}', model=self.model)
            decision = self.decisions[min(self.round - 1, len(self.decisions) - 1)]
            content = json.dumps({
                "decision": decision, "rationale": "offline terminal audit",
                "constraints": [{"constraint": "Identify the requested answer", "status": "satisfied",
                                 "fact_ids": [], "explanation": "The supplied evidence identifies Candidate A"}],
                "issues": [] if decision == "approve_synthesis" else ["Need stronger support"],
            })
        elif system.startswith(REPORT_JUDGE_PROMPT):
            if self.invalid_judge:
                return LLMResponse(content='{"decision":"evolve"}', model=self.model)
            content = json.dumps({
                "decision": "best_effort" if self.round == 3 else "evolve",
                **({"final_answer": "Candidate A"} if self.round == 3 else {}),
                "rationale": "offline failure analysis",
                "failure_report": {
                    "failure_stage": "verification", "summary": "Need stronger support",
                    "failed_routes": [], "rejected_candidates": [],
                    "unverified_identity_conditions": [], "evidence_conflicts": [],
                    "verifier_critique": "gap", "next_round_priorities": ["Find a primary source"],
                    "continue_evolution": self.round < 3,
                    "planner_instructions": {
                        "priority_checks": ["Verify the candidate identity against primary evidence"],
                        "actions": ["Find a primary source linking Candidate A to the task conditions"],
                        "avoid_repeating": ["Do not treat repeated assertions as independent support"],
                        "acceptance_conditions": ["Primary evidence supports every task condition"],
                    },
                },
            })
        elif system.startswith(EVIDENCE_ROLE.system_prompt):
            payload = json.loads(user.split("Local evidence:\n", 1)[1])
            if payload["mode"] == "round_end":
                content = json.dumps({"summary": "Full table reviewed", "fact_refs": [], "changes": []})
            else:
                incoming = payload["new_facts"][0]
                content = json.dumps({"summary": "Submitted fact retained", "fact_refs": ["new:submitted"],
                    "submitted_fact_ref": "new:submitted", "changes": [{
                        "op": "add", "fact_id": "", "local_ref": "new:submitted",
                        "summary": incoming["summary"], "fact": {
                            "claim": incoming["claim"], "source_urls": incoming["source_urls"],
                            "trace_refs": incoming["trace_refs"],
                        }, "reason": "Evidence from explicit submission"}]})
        elif system.startswith("Break the user's task"):
            content = '["Identify the answer"]'
        elif system.startswith("Investigate the requested question"):
            self.research_turn += 1
            self.research_inputs.append(json.loads(json.dumps(messages)))
            if self.research_turn <= self.research_tool_turns:
                return LLMResponse(content="Research evidence", model=self.model, tool_calls=[{
                    "id": f"search-{self.research_turn}", "type": "function",
                    "function": {"name": "web_search", "arguments": json.dumps({"q": f"evidence-{self.research_turn}"})},
                }], usage={"total_tokens": 10})
            submitted = any(c.get("function", {}).get("name") == "facts_submit"
                            for m in messages for c in m.get("tool_calls", []) or [])
            if self.submit_facts and not submitted:
                return LLMResponse(content="Share supported evidence", model=self.model,
                    tool_calls=[{"id": "submit", "type": "function", "function": {
                        "name": "facts_submit", "arguments": json.dumps({
                            "claim": "Relevant evidence", "summary": "Candidate evidence",
                            "source_urls": [], "evidence_excerpt": "Local source supports Candidate A."})}}],
                    usage={"total_tokens": 10})
            content = "Source https://example.test/evidence supports Candidate A."
        elif system.startswith("Construct the strongest possible solution"):
            content = "Candidate A"
        elif system.startswith("Independently verify the candidate"):
            content = json.dumps({"accepted": self.verifier_accept, "critique": "source supports candidate"})
        elif system.startswith("Combine the supplied intermediate results"):
            content = "Candidate A"
        elif "repair" in system.lower() and self.invalid_judge:
            content = '{"decision":"approve_synthesis"}'
        else:
            raise AssertionError("Unexpected model prompt: " + system[:160])
        if "FUSION FACT ENVELOPE:" in system:
            result = json.loads(content) if system.startswith(("Break the user's task", "Independently verify")) else content
            content = json.dumps({"result": result, "facts": [{"claim": "Relevant evidence",
                                  "source_urls": [], "evidence_excerpt": "Local source supports Candidate A."}]})
        return LLMResponse(content=content, model=self.model, usage={"total_tokens": 10})


class OfflineSession:
    def __init__(self, llm):
        self.llm = llm

    async def __aenter__(self):
        self.snapshot = registry.snapshot()
        self.evidence_retries = EVIDENCE_ROLE.metadata.get("max_llm_retries")
        EVIDENCE_ROLE.metadata["max_llm_retries"] = 1
        agents = AgentRegistry()
        for role in (*BUILTIN_ORCHESTRATION_ROLES, EVOLUTION_JUDGE_ROLE):
            definition = role.model_copy(deep=True)
            definition.metadata["max_llm_retries"] = 1
            agents.register(definition)
        registry.register(AgentRegistry, agents)
        async def search(q):
            return "Evidence https://example.test/" + q + " " + ("details " * 200)
        tools = {"web_search": Tool("web_search", "Offline source lookup", {
            "type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"],
        }, search)}
        registry.register(ResourceManager, ResourceManager(llm=self.llm, tools=tools))
        return self

    async def __aexit__(self, *args):
        registry.restore(self.snapshot)
        if self.evidence_retries is None:
            EVIDENCE_ROLE.metadata.pop("max_llm_retries", None)
        else:
            EVIDENCE_ROLE.metadata["max_llm_retries"] = self.evidence_retries


class FusionContractTests(unittest.TestCase):
    def test_source_snapshots_match_pinned_versions(self):
        fixture = Path(__file__).with_name("rolechange_v4_snapshots.json")
        expected = json.loads(fixture.read_text())
        root = Path(__file__).resolve().parents[2]
        for path, blocks in expected.items():
            source = (root / path).read_text()
            tree = ast.parse(source)
            found = {}
            for node in tree.body:
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                    name = node.targets[0].id
                elif isinstance(node, ast.FunctionDef):
                    name = node.name
                else:
                    continue
                if name in blocks:
                    found[name] = hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest()
            self.assertEqual(found, blocks, path)

    def test_five_roles_and_fresh_context(self):
        req = request()
        self.assertEqual(set(req.environment.roles), SCRIPT_ROLES)
        self.assertNotIn("evidence_storesubagent", req.environment.roles)
        self.assertTrue(all(r["default_max_turns"] == 25 for name, r in req.environment.roles.items() if name != "evidence_storesubagent"))
        metadata = dict(req.metadata)
        metadata["evolution_context"] = {"round": 2, "previous_source": "OLD_SOURCE", "raw_trace": "OLD_TRACE"}
        req = replace(req, metadata=metadata)
        for prompt in (_render_natural_language_plan_prompt(req), _render_generation_prompt(req, "New plan")):
            self.assertIn('"round": 2', prompt)
            self.assertNotIn("OLD_SOURCE", prompt)
            self.assertNotIn("OLD_TRACE", prompt)

    def test_validator_accepts_five_roles_and_blocks_others(self):
        validator = GeneratedSolveValidator(allowed_roles=set(SCRIPT_ROLES), max_parallel_jobs=4, require_evolution_proposal=True)
        result = validator.validate(SOURCE)
        self.assertTrue(result.accepted, result.to_dict())
        for role in ("general", "evolution_judge"):
            self.assertFalse(validator.validate(SOURCE.replace('"decomposer"', repr(role))).accepted)
        self.assertFalse(validator.validate(SOURCE.replace('"verifier"', '"solver"')).accepted)

    def test_verifier_gate_cannot_be_fabricated_or_reused_for_other_candidate(self):
        jobs = [SimpleNamespace(job_id="verify", role="verifier")]
        p = proposal()
        create = event("job_created", {"instruction": "Verify Candidate A with sources"})
        success = event("job_succeeded", {"result": p["selected_verifier"]})
        original = event("worker_event", {"name": "evolution_role_result", "data": {"result": p["selected_verifier"]}})
        cases = [
            ([], jobs, ""),
            ([create, success], [], ""),
            ([event("job_created", {"instruction": "Verify Candidate B"}), success], jobs, ""),
            ([create, original, success], jobs, "runtime failed"),
            ([create, event("job_failed", {})], jobs, ""),
            ([create, event("job_succeeded", {"result": {"accepted": False, "critique": "rejected"}})], jobs, ""),
            ([create, original, success, event("worker_event", {"name": "evolution_role_result", "data": {"result": {"accepted": False, "critique": "later rejection"}}}), success], jobs, ""),
            ([create, event("job_succeeded", {"result": {"accepted": True, "critique": "different actual verdict"}})], jobs, ""),
        ]
        for events, actual_jobs, error in cases:
            with self.subTest(events=events):
                p = proposal()
                bind_executed_verifier(p, events, actual_jobs, error)
                decision = normalize_judge_decision({"decision": "approve_synthesis"}, proposal=p, failure_stage="verification", summary="", previous_report=None, observed_urls=set())
                self.assertEqual(decision["decision"], "evolve")
        p = proposal()
        bind_executed_verifier(p, [create, original, success], jobs)
        self.assertEqual(normalize_round_proposal(p)["status"], "ready")
        self.assertNotIn("classification", p["selected_verifier"])

    def test_periodic_compaction(self):
        cfg = LoopConfig()
        policy = DefaultCompactionPolicy(cfg.compact_after_turns, cfg.context_token_limit)
        self.assertEqual([t for t in range(71) if policy.should_compact(t, [], 0)], list(range(5, 71, 5)))
        self.assertTrue(policy.should_compact(3, [], 120001))
        self.assertFalse(policy.should_compact(3, [], 120000))


class FusionIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def run_case(self, directory, llm, *, failures=None, planner_only=False, max_jobs=12):
        config = SimpleNamespace(planner_model=None, llm_provider="openai", openai_model=llm.model, openai_base_url="https://example.test/v1", planner_max_tokens=8192)
        config.model_copy = lambda **kwargs: config
        actual_execute = AgentHarnessWorkerExecutor.execute
        requests = []

        async def execute(executor, req, emit):
            requests.append(req)
            round_number = int(req.run_id.split("round-")[1][:2])
            failure_key = (round_number, req.role, req.key)
            if failures and failure_key in failures:
                raise RuntimeError("offline injected worker failure")
            return await actual_execute(executor, req, emit=emit)

        with patch("benchmarks.core.kernel_adapter.BenchmarkSession", side_effect=lambda **kw: OfflineSession(llm)), patch("agent_harness.infra.config.get_config", return_value=config), patch("agent_harness.infra.llm_adapter.create_llm", return_value=llm), patch.object(AgentHarnessWorkerExecutor, "execute", execute):
            outcome = await run_self_evolving_strategy(
                "Identify the requested answer.", trial_dir=directory, question_id="offline",
                orchestration_profile="deepseek_v4_flash", planner_model=None,
                planner_max_repair_attempts=0, skip_generated_code_review=False,
                planner_only=planner_only, spec=BudgetSpec(max_jobs=max_jobs),
            )
        manifest = json.loads((directory / "generated/evolution_manifest.json").read_text())
        return outcome, manifest, requests

    async def test_first_round_approval_and_terminal_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            outcome, manifest, requests = await self.run_case(path, ScriptedLLM())
            self.assertTrue(manifest["accepted"], manifest["last_failure_report"])
            self.assertEqual(manifest["completed_rounds"], 1)
            self.assertEqual(requests[-1].role, "synthesizer")
            self.assertEqual([r.role for r in requests[-3:-1]], ["evidence_storesubagent", "evolution_judge"])
            self.assertFalse(any(r.key == "terminal:evolution_report" for r in requests))
            self.assertIsNone(requests[-1].max_turns)
            self.assertEqual(requests[-2].max_turns, 25)
            self.assertEqual((path / "agent/final_answer.txt").read_text(), "Candidate A")
            self.assertNotIn("_terminal_error", outcome)
            self.assertEqual(outcome["usage"]["jobs"], 9)
            trace = json.loads((path / "generated/evolution_round_01/terminal_review/structured_trace.json").read_text())
            self.assertTrue(trace["worker_evidence"])

    async def test_fresh_generation_same_source_allowed_and_round_three_policy(self):
        llm = ScriptedLLM(("evolve", "evolve", "approve_synthesis"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            _, manifest, _ = await self.run_case(path, llm)
            self.assertTrue(manifest["accepted"], manifest["last_failure_report"])
            self.assertEqual(manifest["completed_rounds"], 3)
            self.assertEqual(len({r["generation_id"] for r in manifest["rounds"]}), 3)
            self.assertEqual(len({r["source_hash"] for r in manifest["rounds"]}), 1)
            self.assertEqual(len(llm.prompts), 6)
            for round_number, stage, prompt in llm.prompts:
                if stage == "plan" and round_number > 1:
                    self.assertIn("Find a primary source", prompt)
                    self.assertIn("Relevant evidence", prompt)
                    self.assertNotIn(SOURCE, prompt)
                    self.assertNotIn("First ask the decomposer", prompt)
            for round_number, prompt in enumerate(llm.judge_prompts, 1):
                self.assertTrue(prompt.startswith(local_judge_role(round_number).system_prompt))
            self.assertNotIn("V4 persistent relaxed", llm.judge_prompts[1])
            self.assertIn("V4 persistent relaxed", llm.judge_prompts[2])
            self.assertFalse(list(path.glob("generated/evolution_round_*/checkpoint.json")))

    async def test_generation_failure_does_not_reset_round_three_policy(self):
        llm = ScriptedLLM(("evolve", "evolve", "approve_synthesis"), generation_failures=(1,))
        with tempfile.TemporaryDirectory() as tmp:
            _, manifest, _ = await self.run_case(Path(tmp), llm)
            self.assertEqual(manifest["rounds"][0]["status"], "generation_failed")
            self.assertTrue(manifest["accepted"], manifest["last_failure_report"])
            self.assertEqual(manifest["completed_rounds"], 3)
            self.assertIn("V4 persistent relaxed", llm.judge_prompts[-1])

    async def test_execution_judge_and_synthesis_failures_start_fresh_rounds(self):
        for role, key in (("researcher", "research"), ("evolution_judge", "terminal:evolution_judge"), ("synthesizer", "terminal:synthesizer")):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as tmp:
                _, manifest, _ = await self.run_case(Path(tmp), ScriptedLLM(), failures={(1, role, key)})
                self.assertTrue(manifest["accepted"], manifest["last_failure_report"])
                self.assertEqual(manifest["completed_rounds"], 2)
                self.assertNotEqual(manifest["rounds"][0]["status"], "approved")

    async def test_verifier_rejection_cannot_be_overridden_and_exhausts_three_rounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            outcome, manifest, requests = await self.run_case(path, ScriptedLLM(verifier_accept=False))
            self.assertFalse(manifest["accepted"])
            self.assertEqual(manifest["completed_rounds"], 3)
            self.assertNotIn("_terminal_error", outcome)
            self.assertEqual(manifest["status"], "best_effort")
            self.assertTrue((path / "agent/final_answer.txt").read_text().strip())
            self.assertFalse(any(r.key == "terminal:synthesizer" for r in requests))

    async def test_retry_archives_previous_code_and_planner_only_skips_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            _, first, _ = await self.run_case(path, ScriptedLLM())
            llm = ScriptedLLM()
            _, second, requests = await self.run_case(path, llm, planner_only=True)
            self.assertEqual(requests, [])
            self.assertEqual(llm.round, 1)
            self.assertNotEqual(first["rounds"][0]["generation_id"], second["rounds"][0]["generation_id"])
            self.assertTrue(list(path.glob("previous_attempts/*/generated/evolution_manifest.json")))
            self.assertFalse((path / "agent/final_answer.txt").exists())

    async def test_script_mutating_a_rejected_verdict_cannot_change_original_evidence(self):
        llm = ScriptedLLM(verifier_accept=False)
        llm.source = SOURCE.replace('    status = "incomplete"', '    verdict["accepted"] = True\n    status = "incomplete"')
        with tempfile.TemporaryDirectory() as tmp:
            _, manifest, _ = await self.run_case(Path(tmp), llm)
            self.assertFalse(manifest["accepted"])
            self.assertEqual(manifest["completed_rounds"], 3)

    async def test_malformed_judge_output_never_authorizes_synthesis(self):
        llm = ScriptedLLM()
        llm.invalid_judge = True
        with tempfile.TemporaryDirectory() as tmp:
            _, manifest, requests = await self.run_case(Path(tmp), llm)
            self.assertFalse(manifest["accepted"])
            self.assertFalse(any(r.key == "terminal:synthesizer" for r in requests))

    async def test_real_tool_loop_keeps_full_evidence_after_multiple_compactions(self):
        llm = ScriptedLLM()
        llm.research_tool_turns = 20
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            _, manifest, _ = await self.run_case(path, llm)
            self.assertTrue(manifest["accepted"], manifest["last_failure_report"])
            self.assertIn("[Compacted", json.dumps(llm.research_inputs[10]))
            trace = json.loads((path / "generated/evolution_round_01/terminal_review/structured_trace.json").read_text())
            tools = [e["data"] for e in trace["worker_evidence"] if e["name"] == "evolution_tool_evidence"]
            self.assertEqual(len(tools), 21)  # Twenty searches plus the explicit fact submission.
            self.assertIn("https://example.test/evidence-1 ", tools[0]["result"])
            self.assertGreater(len(tools[0]["result"]), 1000)

    async def test_preflight_checks_proposal_shape(self):
        validator = GeneratedSolveValidator(allowed_roles=set(SCRIPT_ROLES), max_parallel_jobs=4, require_evolution_proposal=True)
        await preflight_solve(validator.compile(SOURCE), request().environment.roles, require_evolution_proposal=True)
        with self.assertRaises(ValueError):
            validate_proposal_shape("final answer")

    async def test_evidence_survives_context_compaction(self):
        events = []
        async def emit(name, data):
            events.append(event("worker_event", {"name": name, "data": data}, sequence=len(events) + 1))
        observer = EvidenceObserver(emit)
        result = ToolResult("web_fetch", {}, "early source https://example.test/early", 1, "tool1", False)
        await observer.on_tool_result(SimpleNamespace(turn=1), result)
        messages = [{"role": "system", "content": "Task"}] + [
            {"role": "user", "content": str(i)} for i in range(40)
        ]
        DefaultMessageCompactor().compact(messages, 16)
        trace = {}
        attach_worker_evidence(trace, events)
        self.assertIn("https://example.test/early", json.dumps(trace))


if __name__ == "__main__":
    unittest.main()
