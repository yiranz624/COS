from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from agent_harness.core.llm import LLMResponse
from agent_harness.orchestration.context import OrchestrationContext
from agent_harness.orchestration.contracts import BudgetSpec
from agent_harness.orchestration.generated_code import (
    _PreflightContext,
    _ctx_api_reference,
    _natural_language_planner_system_prompt,
    _render_code_review_prompt,
    _render_natural_language_plan_prompt,
    _render_plan_review_prompt,
    _render_generation_prompt,
    _render_roles_markdown,
    EnvironmentManifestV1,
    GeneratedSolvePlanner,
    GeneratedSolveValidator,
    PreflightSchemaError,
    SolveGenerationRequestV1,
    build_environment_manifest,
    build_generation_request,
    extract_source,
    preflight_solve,
    write_generation_research_report,
)
from agent_harness.orchestration.runtime import ScriptRuntime
from agent_harness.orchestration.roles import BUILTIN_ORCHESTRATION_ROLES
from agent_harness.orchestration.testing import FakeWorkerExecutor

ROLES = {"researcher", "solver", "verifier", "synthesizer"}
NATURAL_LANGUAGE_PLAN = """# Orchestration Plan

1. At the start, call researcher agents to investigate independent evidence routes in parallel.
2. Once research is available, call the solver agent to propose the strongest candidate.
3. When a candidate exists, call the verifier agent to check it against the task.
4. After verification succeeds, call the synthesizer agent to produce the final answer."""
ACCEPT_REVIEW = (
    '{"accepted": true, "critique": "", "issues": [], "suggested_changes": []}'
)


def _validator(**kwargs):
    return GeneratedSolveValidator(
        allowed_roles=ROLES, max_parallel_jobs=2, **kwargs,
    )


def _request(task="# Task\n\nquestion"):
    roles = {role: {"result_schema": {}} for role in ROLES}
    roles["researcher"] = {
        "planner_usage": "Perform constraint-by-constraint verification.",
        "result_guidance": "Return concise evidence text with source URLs.",
        "result_schema": {},
    }
    return SolveGenerationRequestV1(
        task=task, metadata={},
        environment=EnvironmentManifestV1(
            runtime_version="1.1", dsl_version="1",
            roles=roles, tools={},
            ctx_api=("call",), budget={"max_tokens": 10000},
        ),
        templates=(), failure_lessons=(), generation_constraints=(),
    )


class FakePlannerLLM:
    model = "fake-planner"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    async def chat(self, messages, **kwargs):
        value = self.responses[self.calls]
        self.calls += 1
        if isinstance(value, BaseException):
            raise value
        return LLMResponse(
            content=value, model=self.model,
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )


class GeneratedSolveValidatorTests(unittest.TestCase):
    def test_default_budget_has_no_token_cap_and_uses_expanded_limits(self):
        budget = BudgetSpec()
        self.assertIsNone(budget.max_tokens)
        self.assertEqual(budget.max_jobs, 128)
        self.assertIsNone(budget.wall_time_seconds)

    def test_environment_explains_role_responsibilities_and_planner_usage(self):
        environment = build_environment_manifest(
            roles={"solver": SimpleNamespace(
                description="Constructs candidate solutions.",
                system_prompt="Construct a supported solution and preserve evidence.",
                allowed_tools=["web_search"],
                result_schema={},
                metadata={"max_turns": 4},
            )},
            tools={},
            budget=BudgetSpec(max_tokens=10000),
        )
        role = environment.roles["solver"]
        self.assertIn("preserve evidence", role["responsibilities"])
        self.assertIn("stated responsibilities", role["planner_usage"])
        self.assertIn("web_search", role["planner_usage"])
        self.assertIn("Returns plain text", role["result_guidance"])

    def test_environment_explains_researcher_to_planner(self):
        environment = build_environment_manifest(
            roles={"researcher": SimpleNamespace(
                description="Finds and evaluates external evidence.",
                system_prompt="Investigate with reliable evidence and source URLs.",
                allowed_tools=["web_search", "web_fetch"],
                result_schema=None,
                metadata={"max_turns": 10},
            )},
            tools={},
            budget=BudgetSpec(max_tokens=10000),
        )
        role = environment.roles["researcher"]
        self.assertIn("reliable evidence", role["responsibilities"])
        self.assertIn("web_search, web_fetch", role["planner_usage"])
        self.assertIn("Returns plain text", role["result_guidance"])
        self.assertIn("do not read dictionary fields", role["result_guidance"])

    def test_environment_uses_builtin_agent_definitions_as_source_of_truth(self):
        definitions = {
            definition.role_id: definition
            for definition in BUILTIN_ORCHESTRATION_ROLES
        }
        environment = build_environment_manifest(
            roles=definitions,
            tools={},
            budget=BudgetSpec(max_tokens=10000),
        )

        self.assertEqual(set(environment.roles), set(definitions))
        for role_id, definition in definitions.items():
            rendered = environment.roles[role_id]
            self.assertEqual(rendered["responsibilities"], definition.system_prompt)
            self.assertEqual(rendered["allowed_tools"], definition.allowed_tools)
            self.assertEqual(
                rendered["result_schema"], definition.result_schema or {},
            )
            self.assertEqual(
                rendered["default_max_turns"],
                definition.metadata["max_turns"],
            )

    def test_planner_prompt_includes_description_and_planner_usage(self):
        request = SolveGenerationRequestV1(
            task="question",
            metadata={},
            environment=EnvironmentManifestV1(
                runtime_version="1.1",
                dsl_version="1",
                roles={
                    "researcher": {
                        "description": "Finds and evaluates external evidence.",
                        "planner_usage": (
                            "Use when claims require source-backed verification."
                        ),
                    },
                },
                tools={},
                ctx_api=("call",),
                budget={"max_tokens": 10000},
            ),
            templates=(),
            failure_lessons=(),
            generation_constraints=(),
        )

        prompt = _render_natural_language_plan_prompt(request)

        self.assertIn(
            "## Available agents\n\n"
            "Available roles are optional rather than mandatory.\n\n"
            "- `researcher`\n"
            "  - Capability: Finds and evaluates external evidence.\n"
            "  - Instructions: Follow the delegated instruction carefully.\n"
            "  - Best used when: Use when claims require source-backed verification.",
            prompt,
        )
        planner_instruction = (
            "The Planner itself is responsible for decomposing the original task "
            "into stages and assigning work to agents."
        )
        self.assertIn(planner_instruction, prompt)
        self.assertNotIn("decomposer", prompt.casefold())

    def test_active_plan_and_code_prompts_exclude_decomposer(self):
        request = _request()
        plan = NATURAL_LANGUAGE_PLAN
        prompts = (
            _natural_language_planner_system_prompt(),
            _render_natural_language_plan_prompt(request),
            _render_plan_review_prompt(request, plan),
            _render_generation_prompt(request, plan),
            _render_code_review_prompt(
                request, plan, "async def solve(ctx):\n    return 'answer'",
            ),
        )

        for prompt in prompts:
            with self.subTest(prompt=prompt[:40]):
                self.assertNotRegex(prompt, r"(?i)decomposer")
                self.assertNotIn("DECOMPOSER_ROLE", prompt)

    def test_coder_role_prompt_includes_normative_result_schema(self):
        prompt = _render_roles_markdown({
            "researcher": {
                "description": "Finds evidence.",
                "result_guidance": "Preserve candidates.",
                "result_schema": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "candidate_table": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["status", "candidate_table"],
                },
            },
            "solver": {
                "description": "Returns text.",
                "result_guidance": "Use the text directly.",
                "result_schema": {},
            },
        })

        self.assertIn("**Normative Result Schema**", prompt)
        self.assertIn('"candidate_table"', prompt)
        self.assertIn('"items": {\n        "type": "string"', prompt)
        self.assertIn("Use the text directly.", prompt)
        self.assertEqual(prompt.count("**Normative Result Schema**"), 1)

    def test_ctx_api_reference_describes_interfaces_without_examples(self):
        reference = _ctx_api_reference((
            "call", "spawn", "gather", "parallel", "checkpoint",
            "load_checkpoint", "remaining_budget", "emit",
        ))
        self.assertNotIn("### API micro-patterns", reference)
        self.assertNotIn("```python", reference)
        self.assertNotIn("**Sequential role call**", reference)
        self.assertIn("### Parallel and await rules", reference)
        self.assertIn("`ctx.call`, `ctx.gather`, `ctx.parallel`", reference)
        self.assertIn("must be awaited before their results are consumed", reference)
        self.assertIn("`ctx.spawn`, `ctx.load_checkpoint`, `ctx.remaining_budget`, `ctx.emit`", reference)
        self.assertIn("must never be awaited", reference)
        self.assertIn("Do not pass `ctx.call` awaitables to `ctx.gather`", reference)
        self.assertIn("do not pass `ctx.spawn` handles to `ctx.parallel`", reference)
        self.assertIn("### `await ctx.call", reference)
        self.assertIn("### `ctx.spawn", reference)
        self.assertIn("### `await ctx.gather", reference)
        self.assertIn("### `await ctx.parallel", reference)
        self.assertIn("failure_policy=", reference)
        self.assertIn("all other methods are synchronous and must not be awaited", reference)
        self.assertIn("These are plain role results, not `JobOutcome` objects", reference)
        self.assertIn("Never use `unknown` alone", reference)

        call_only = _ctx_api_reference(("call",))
        self.assertIn("### `await ctx.call", call_only)
        self.assertNotIn("### `ctx.emit", call_only)
        self.assertNotIn("### `ctx.spawn", call_only)

    def test_generation_request_has_no_complete_solve_templates(self):
        environment = EnvironmentManifestV1(
            runtime_version="1.1", dsl_version="1",
            roles={role: {} for role in ROLES}, tools={},
            ctx_api=("call", "parallel", "checkpoint", "remaining_budget"),
            budget={"max_tokens": 10000},
        )
        request = build_generation_request("question", {}, environment)
        self.assertEqual(request.templates, ())

    def test_accepts_bounded_control_flow(self):
        source = '''async def solve(ctx):
    answers = []
    for index in range(2):
        budget = ctx.remaining_budget()
        if budget.remaining.get("jobs", 0) < 1:
            break
        try:
            value = await ctx.call(
                "solver", ctx.task, key=f"solve:{index}:task",
            )
            answers.append(value)
        except Exception:
            answers.append("Insufficient information")
    return "\\n".join(answers)
'''
        self.assertTrue(_validator().validate(source).accepted)

    def test_loop_job_key_requires_stage_iteration_and_entity(self):
        invalid_sources = [
            '''async def solve(ctx):
    for iteration in range(2):
        await ctx.call("solver", ctx.task)
    return "done"
''',
            '''async def solve(ctx):
    for iteration in range(2):
        await ctx.call("solver", ctx.task, key="solve")
    return "done"
''',
            '''async def solve(ctx):
    for iteration in range(2):
        await ctx.call("solver", ctx.task, key=f"solve:{iteration}")
    return "done"
''',
            '''async def solve(ctx):
    for iteration in range(2):
        await ctx.call(
            "solver", ctx.task, key=f"solve:{iteration}:unknown",
        )
    return "done"
''',
            '''async def solve(ctx):
    for iteration in range(2):
        ctx.spawn(
            "researcher", ctx.task, key=f"research:{iteration}",
        )
    return "done"
''',
            '''async def solve(ctx):
    for iteration in range(2):
        entity_id = "unknown"
        await ctx.call(
            "solver", ctx.task,
            key=f"solve:{iteration}:{entity_id}",
        )
    return "done"
''',
        ]
        for source in invalid_sources:
            with self.subTest(source=source):
                report = _validator().validate(source)
                self.assertFalse(report.accepted)
                self.assertIn(
                    "loop_job_key", {issue.code for issue in report.issues},
                )

        valid_source = '''async def solve(ctx):
    for iteration in range(2):
        entity_id = "candidate"
        await ctx.call(
            "solver", ctx.task,
            key=f"solve:{iteration}:{entity_id}",
        )
    return "done"
'''
        self.assertTrue(_validator().validate(valid_source).accepted)

        local_key_source = '''async def solve(ctx):
    for iteration in range(2):
        job_key = f"solve:{iteration}:candidate"
        await ctx.call("solver", ctx.task, key=job_key)
    return "done"
'''
        self.assertTrue(_validator().validate(local_key_source).accepted)

    def test_collection_loop_job_key_must_identify_current_item(self):
        invalid_sources = [
            '''async def solve(ctx):
    for item in ["left", "right"]:
        await ctx.call("researcher", "research", key="research:0:source")
    return "done"
''',
            '''async def solve(ctx):
    index = 0
    for item in ["left", "right"]:
        await ctx.call(
            "researcher", "research", key=f"research:{index}:source",
        )
        index += 1
    return "done"
''',
            '''async def solve(ctx):
    for index, item in enumerate(["left", "right"]):
        await ctx.call(
            "researcher", "research", key=f"research:{index}:source",
        )
    return "done"
''',
        ]
        for source in invalid_sources:
            with self.subTest(source=source):
                report = _validator().validate(source)
                self.assertFalse(report.accepted)
                self.assertIn(
                    "loop_job_key", {issue.code for issue in report.issues},
                )

        valid_source = '''async def solve(ctx):
    for item in ["left", "right"]:
        await ctx.call(
            "researcher", "research", key=f"research:{item}:source",
        )
    return "done"
'''
        self.assertTrue(_validator().validate(valid_source).accepted)

        enumerate_source = '''async def solve(ctx):
    for index, item in enumerate(["left", "right"]):
        await ctx.call(
            "researcher", "research", key=f"research:{index}:{item}",
        )
    return "done"
'''
        self.assertTrue(_validator().validate(enumerate_source).accepted)

        alias_source = '''async def solve(ctx):
    for index, item in enumerate(["left", "right"]):
        entity_id = item
        await ctx.call(
            "researcher", "research", key=f"research:{index}:{entity_id}",
        )
    return "done"
'''
        self.assertTrue(_validator().validate(alias_source).accepted)

    def test_different_loop_invocations_cannot_reuse_dynamic_key(self):
        source = '''async def solve(ctx):
    for index, item in enumerate(["left", "right"]):
        await ctx.call(
            "researcher", "research", key=f"research:{index}:{item}",
        )
        await ctx.call(
            "solver", "solve", key=f"research:{index}:{item}",
        )
    return "done"
'''
        report = _validator().validate(source)
        self.assertFalse(report.accepted)
        self.assertIn(
            "duplicate_job_key", {issue.code for issue in report.issues},
        )

    def test_different_invocations_cannot_reuse_one_explicit_key(self):
        source = '''async def solve(ctx):
    first = await ctx.call(
        "solver", "first instruction", key="solve:initial:task",
    )
    second = await ctx.call(
        "solver", "second instruction", key="solve:initial:task",
    )
    return str([first, second])
'''
        report = _validator().validate(source)
        self.assertFalse(report.accepted)
        self.assertIn(
            "duplicate_job_key", {issue.code for issue in report.issues},
        )

    def test_exact_retry_may_reuse_one_explicit_key(self):
        source = '''async def solve(ctx):
    first = await ctx.call(
        "solver", "same instruction", key="solve:initial:task",
    )
    second = await ctx.call(
        "solver", "same instruction", key="solve:initial:task",
    )
    return str([first, second])
'''
        self.assertTrue(_validator().validate(source).accepted)

    def test_allows_best_effort_checkpoint_failure(self):
        sources = [
            '''async def solve(ctx):
    try:
        await ctx.checkpoint("state", {"answer": "candidate"})
    except Exception:
        pass
    return "candidate"
''',
            '''async def solve(ctx):
    try:
        await ctx.checkpoint("state", {"answer": "candidate"})
    except Exception as exc:
        ctx.emit("checkpoint_failed", detail=str(exc))
    return "candidate"
''',
        ]
        for source in sources:
            with self.subTest(source=source):
                self.assertTrue(_validator().validate(source).accepted)

    def test_rejects_unsafe_sources(self):
        cases = [
            ("import os\nasync def solve(ctx):\n return 'x'\n", "top_level"),
            ("async def solve(ctx):\n return open('/tmp/x')\n", "forbidden_name"),
            ("async def solve(ctx):\n return eval('1')\n", "forbidden_name"),
            ("async def solve(ctx):\n while True:\n  pass\n", "forbidden_syntax"),
            ("async def solve(ctx):\n return ctx.__class__\n", "private_attribute"),
            ("async def solve(ctx):\n return await ctx.call('ghost', 'x')\n", "unknown_role"),
            ("async def solve(ctx):\n return await ctx.call('decomposer', 'x')\n", "unknown_role"),
            ("async def solve(ctx):\n return await ctx.call(ctx.metadata['role'], 'x')\n", "dynamic_role"),
            ("async def solve(ctx):\n return await ctx.parallel(1, 2, 3)\n", "parallel_limit"),
            ("async def solve(ctx):\n value = await ctx.load_checkpoint('x')\n return str(value)\n", "await_sync_ctx_api"),
            ("async def solve(ctx):\n return await ctx.gather([ctx.call('solver', 'x')])\n", "gather_requires_handles"),
            ("async def solve(ctx):\n return await ctx.map('solver', ['x'])\n", "map_requires_local_async_function"),
        ]
        for source, code in cases:
            with self.subTest(code=code):
                report = _validator().validate(source)
                self.assertFalse(report.accepted)
                self.assertIn(code, {issue.code for issue in report.issues})

    def test_robustness_findings_are_warnings_and_do_not_block(self):
        cases = [
            ("async def solve(ctx):\n for x in range(100):\n  pass\n return 'x'\n", "loop_bound"),
            ("async def solve(ctx):\n return await ctx.call('verifier', 'x')\n", "unprotected_critical_call"),
            ("async def solve(ctx):\n try:\n  return 'x'\n except Exception:\n  pass\n", "empty_failure_degradation"),
            ("async def solve(ctx):\n for x in range(2):\n  await ctx.call('solver', 'x', key=f'solve:{x}:task')\n return 'x'\n", "missing_loop_budget_check"),
        ]
        for source, code in cases:
            with self.subTest(code=code):
                report = _validator().validate(source)
                self.assertTrue(report.accepted)
                self.assertEqual(report.issues, ())
                self.assertIn(code, {warning.code for warning in report.warnings})
                self.assertTrue(report.source_hash)
                self.assertIn("warnings", report.to_dict())
                self.assertTrue(callable(_validator().compile(source)))

    def test_generation_request_does_not_require_robustness_repairs(self):
        environment = EnvironmentManifestV1(
            runtime_version="1.1", dsl_version="1",
            roles={role: {} for role in ROLES}, tools={},
            ctx_api=("call", "remaining_budget"), budget={"max_jobs": 10},
        )
        request = build_generation_request("question", {}, environment)
        constraints = "\n".join(request.generation_constraints)
        lesson_ids = {item["lesson_id"] for item in request.failure_lessons}
        self.assertIn("Robustness is not an acceptance requirement", constraints)
        self.assertIn("no role is mandatory for every task", constraints)
        self.assertIn("declared contract", constraints)
        self.assertIn("roles without a schema return text", constraints)
        self.assertIn("Do not read fields absent", constraints)
        self.assertIn(
            "logical stage, loop iteration, and stable entity identity",
            constraints,
        )
        self.assertIn("Never use 'unknown' alone", constraints)
        self.assertIn("do not need a static <=8 bound", constraints)
        self.assertNotIn("must inspect ctx.remaining_budget", constraints)
        self.assertNotIn("bounded_agent_loops_v1", lesson_ids)
        self.assertNotIn("preserve_candidate_on_verifier_failure_v1", lesson_ids)

    def test_nested_loop_inherits_outer_budget_gate(self):
        source = '''async def solve(ctx):
    answers = []
    for route in range(2):
        budget = ctx.remaining_budget()
        if budget.remaining.get("jobs", 0) < 1:
            break
        for attempt in range(2):
            try:
                answer = await ctx.call(
                    "solver", str(route),
                    key=f"solve:{route}:{attempt}:route-{route}",
                )
                answers.append(answer)
            except Exception:
                answers.append("Insufficient information")
    return "\\n".join(answers)
'''
        self.assertTrue(_validator().validate(source).accepted)

    def test_allows_safe_setdefault(self):
        source = '''async def solve(ctx):
    state = {}
    state.setdefault("answer", "Insufficient information")
    return state.get("answer", "Insufficient information")
'''
        self.assertTrue(_validator().validate(source).accepted)

    def test_allows_bounded_collection_slice(self):
        source = '''async def solve(ctx):
    values = ctx.metadata.get("values", [])
    answers = []
    for value in values[:8]:
        answers.append(str(value))
    return "\\n".join(answers)
'''
        self.assertTrue(_validator().validate(source).accepted)

    def test_markdown_source_is_unwrapped(self):
        raw = "```python\nasync def solve(ctx):\n    return 'ok'\n```"
        self.assertTrue(extract_source(raw).startswith("async def solve(ctx)"))


class GeneratedSolvePreflightTests(unittest.IsolatedAsyncioTestCase):
    def test_facts_query_signature_matches_runtime(self):
        self.assertEqual(
            inspect.signature(_PreflightContext.facts_query),
            inspect.signature(OrchestrationContext.facts_query),
        )

    async def test_facts_query_accepts_runtime_arguments(self):
        validator = _validator()
        validator.allowed_ctx_methods = validator.allowed_ctx_methods | {"facts_query"}
        for arguments in (
            "",
            'keyword="dinner", fact_ids=["fact-1"], statuses=[], rounds=[1, 2]',
        ):
            with self.subTest(arguments=arguments):
                source = (
                    "async def solve(ctx):\n"
                    f"    facts = await ctx.facts_query({arguments})\n"
                    "    return str(facts)\n"
                )
                await preflight_solve(validator.compile(source), {})

    async def test_facts_query_rejects_invalid_arguments(self):
        validator = _validator()
        validator.allowed_ctx_methods = validator.allowed_ctx_methods | {"facts_query"}
        for arguments, error in (
            ('keywords=["haram", "dinner"], statuses=[]', "keywords"),
            ('keyword="dinner", unknown=True', "unknown"),
            ('"dinner"', "positional"),
        ):
            with self.subTest(arguments=arguments):
                source = (
                    "async def solve(ctx):\n"
                    f"    facts = await ctx.facts_query({arguments})\n"
                    "    return str(facts)\n"
                )
                with self.assertRaisesRegex(TypeError, error):
                    await preflight_solve(validator.compile(source), {})

    async def test_collect_gather_returns_job_outcomes(self):
        source = '''async def solve(ctx):
    left = ctx.spawn("researcher", "left", key="left")
    right = ctx.spawn("researcher", "right", key="right")
    outcomes = await ctx.gather(left, right, failure_policy="collect")
    values = []
    for outcome in outcomes[:8]:
        if outcome.succeeded and outcome.result is not None:
            values.append(outcome.result.value)
        else:
            values.append("unavailable")
    return str(values)
'''
        validator = _validator()
        self.assertTrue(validator.validate(source).accepted)
        await preflight_solve(validator.compile(source), _request().environment.roles)

    async def test_fail_fast_gather_still_returns_plain_values(self):
        source = '''async def solve(ctx):
    handle = ctx.spawn("researcher", "research", key="research")
    values = await ctx.gather(handle)
    return str(values[0])
'''
        validator = _validator()
        self.assertTrue(validator.validate(source).accepted)
        await preflight_solve(validator.compile(source), _request().environment.roles)

    async def test_preflight_generates_values_from_declared_role_schemas(self):
        role_definitions = {
            "array_agent": {
                "result_schema": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "object_agent": {
                "result_schema": {
                    "type": "object",
                    "properties": {
                        "accepted": {"type": "boolean"},
                        "critique": {"type": "string"},
                    },
                    "required": ["accepted", "critique"],
                    "additionalProperties": False,
                },
            },
            "text_agent": {"result_schema": {}},
        }
        source = '''async def solve(ctx):
    items = await ctx.call("array_agent", "items", key="items")
    verdict = await ctx.call("object_agent", "verify", key="verify")
    text = await ctx.call("text_agent", "text", key="text")
    return str([items[0], verdict["accepted"], verdict["critique"], text])
'''
        validator = GeneratedSolveValidator(
            allowed_roles=set(role_definitions), max_parallel_jobs=2,
        )
        self.assertTrue(validator.validate(source).accepted)
        await preflight_solve(validator.compile(source), role_definitions)

    async def test_preflight_generates_nested_schema_values_without_role_branches(self):
        role_definitions = {
            "custom_worker": {
                "result_schema": {
                    "type": "object",
                    "properties": {
                        "records": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"id": {"type": "string"}},
                                "required": ["id"],
                            },
                        },
                        "choice": {
                            "oneOf": [
                                {"type": "string"},
                                {"type": "integer"},
                            ],
                        },
                    },
                    "required": ["records", "choice"],
                },
            },
        }
        source = '''async def solve(ctx):
    value = await ctx.call("custom_worker", "work", key="custom")
    return str([value["records"][0]["id"], value["choice"]])
'''
        validator = GeneratedSolveValidator(
            allowed_roles=set(role_definitions), max_parallel_jobs=2,
        )
        self.assertTrue(validator.validate(source).accepted)
        await preflight_solve(validator.compile(source), role_definitions)

    async def test_preflight_rejects_missing_role_schema_instead_of_guessing(self):
        source = '''async def solve(ctx):
    return await ctx.call("custom_worker", "work", key="custom")
'''
        validator = GeneratedSolveValidator(
            allowed_roles={"custom_worker"}, max_parallel_jobs=2,
        )
        role_definitions = {"custom_worker": {}}
        self.assertTrue(validator.validate(source).accepted)
        with self.assertRaisesRegex(PreflightSchemaError, "no result_schema"):
            await preflight_solve(validator.compile(source), role_definitions)


class GeneratedSolvePlannerTests(unittest.IsolatedAsyncioTestCase):
    async def test_natural_language_plan_retries_transient_connection_error(self):
        source = """async def solve(ctx):
    return "answer"
"""
        llm = FakePlannerLLM([
            ConnectionError("Connection error."),
            NATURAL_LANGUAGE_PLAN,
            ACCEPT_REVIEW,
            source,
            ACCEPT_REVIEW,
        ])
        planner = GeneratedSolvePlanner(
            llm, _validator(), model=llm.model, max_repair_attempts=1,
            max_tokens_per_call=100,
        )
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            result = await planner.generate(
                _request(), artifact_dir=artifact_dir,
                budget=BudgetSpec(max_tokens=10000),
            )

            self.assertIsNotNone(result.generated)
            self.assertEqual(llm.calls, 5)
            failure = json.loads(
                (artifact_dir / "plan" / "attempts" / "attempt_01" / "failure.json")
                .read_text()
            )
            self.assertEqual(failure["error_type"], "ConnectionError")

    async def test_plan_review_retries_transient_connection_error(self):
        source = """async def solve(ctx):
    return "answer"
"""
        llm = FakePlannerLLM([
            NATURAL_LANGUAGE_PLAN,
            ConnectionError("Connection error."),
            NATURAL_LANGUAGE_PLAN,
            ACCEPT_REVIEW,
            source,
            ACCEPT_REVIEW,
        ])
        planner = GeneratedSolvePlanner(
            llm, _validator(), model=llm.model, max_repair_attempts=1,
            max_tokens_per_call=100,
        )
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            result = await planner.generate(
                _request(), artifact_dir=artifact_dir,
                budget=BudgetSpec(max_tokens=10000),
            )

            self.assertIsNotNone(result.generated)
            self.assertEqual(llm.calls, 6)
            failure = json.loads(
                (artifact_dir / "plan" / "attempts" / "attempt_01" / "review" / "failure.json")
                .read_text()
            )
            self.assertEqual(failure["error_type"], "ConnectionError")

    async def test_code_generation_retries_transient_connection_error(self):
        source = """async def solve(ctx):
    try:
        return await ctx.call("solver", ctx.task)
    except Exception:
        return "Insufficient information"
"""
        llm = FakePlannerLLM([
            NATURAL_LANGUAGE_PLAN,
            ACCEPT_REVIEW,
            ConnectionError("Connection error."),
            source,
            ACCEPT_REVIEW,
        ])
        planner = GeneratedSolvePlanner(
            llm, _validator(), model=llm.model, max_repair_attempts=1,
            max_tokens_per_call=100,
        )
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            result = await planner.generate(
                _request(), artifact_dir=artifact_dir,
                budget=BudgetSpec(max_tokens=10000),
            )

            self.assertIsNotNone(result.generated)
            self.assertEqual(llm.calls, 5)
            failure = json.loads(
                (artifact_dir / "attempts" / "attempt_01" / "failure.json")
                .read_text()
            )
            self.assertEqual(failure["error_type"], "ConnectionError")
            self.assertTrue(any(
                event["event_type"] == "orchestrator_generation_retry_scheduled"
                for event in result.events
            ))

    async def test_code_review_retries_transient_connection_error(self):
        source = """async def solve(ctx):
    return "answer"
"""
        llm = FakePlannerLLM([
            NATURAL_LANGUAGE_PLAN,
            ACCEPT_REVIEW,
            source,
            ConnectionError("Connection error."),
            source,
            ACCEPT_REVIEW,
        ])
        planner = GeneratedSolvePlanner(
            llm, _validator(), model=llm.model, max_repair_attempts=1,
            max_tokens_per_call=100,
        )
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            result = await planner.generate(
                _request(), artifact_dir=artifact_dir,
                budget=BudgetSpec(max_tokens=10000),
            )

            self.assertIsNotNone(result.generated)
            self.assertEqual(llm.calls, 6)
            failure = json.loads(
                (artifact_dir / "attempts" / "attempt_01" / "review" / "failure.json")
                .read_text()
            )
            self.assertEqual(failure["error_type"], "ConnectionError")

    async def test_robustness_warnings_are_preserved_in_run_artifacts(self):
        source = """async def solve(ctx):
    return await ctx.call("solver", ctx.task)
"""
        llm = FakePlannerLLM([
            NATURAL_LANGUAGE_PLAN,
            ACCEPT_REVIEW,
            source,
            ACCEPT_REVIEW,
        ])
        planner = GeneratedSolvePlanner(
            llm, _validator(), model=llm.model, max_repair_attempts=0,
            max_tokens_per_call=100,
        )
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            result = await planner.generate(
                _request(), artifact_dir=artifact_dir,
                budget=BudgetSpec(max_tokens=10000),
            )

            self.assertIsNotNone(result.generated)
            assert result.generated is not None
            self.assertEqual(
                [item["code"] for item in result.generated.validation_warnings],
                ["unprotected_critical_call"],
            )
            validation = json.loads(
                (artifact_dir / "validation_report.json").read_text()
            )
            self.assertTrue(validation["accepted"])
            self.assertEqual(validation["issues"], [])
            self.assertEqual(
                [item["code"] for item in validation["warnings"]],
                ["unprotected_critical_call"],
            )
            manifest = json.loads((artifact_dir / "manifest.json").read_text())
            self.assertEqual(
                [item["code"] for item in manifest["validation_warnings"]],
                ["unprotected_critical_call"],
            )

    async def test_can_skip_semantic_code_review_after_preflight(self):
        source = """async def solve(ctx):
    try:
        return await ctx.call("solver", ctx.task)
    except Exception:
        return "Insufficient information"
"""
        llm = FakePlannerLLM([
            NATURAL_LANGUAGE_PLAN,
            ACCEPT_REVIEW,
            source,
        ])
        planner = GeneratedSolvePlanner(
            llm, _validator(), model=llm.model, max_repair_attempts=3,
            max_tokens_per_call=100, review_generated_code=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            result = await planner.generate(
                _request(), artifact_dir=Path(directory),
                budget=BudgetSpec(max_tokens=10000, max_llm_calls=3),
            )

            self.assertIsNotNone(result.generated)
            self.assertEqual(llm.calls, 3)
            self.assertFalse(
                (Path(result.attempts_path) / "attempt_01" / "review").exists()
            )
            self.assertTrue(any(
                event["event_type"] == "generated_code_review_skipped"
                for event in result.events
            ))
            validation = json.loads(
                (Path(result.attempts_path) / "attempt_01" / "validation.json").read_text()
            )
            self.assertTrue(validation["accepted"])

    async def test_repairs_then_executes(self):
        llm = FakePlannerLLM([
            NATURAL_LANGUAGE_PLAN,
            ACCEPT_REVIEW,
            "async def solve(ctx):\n    return open('/tmp/x')\n",
            "async def solve(ctx):\n    try:\n        return await ctx.call('solver', ctx.task)\n    except Exception:\n        return 'Insufficient information'\n",
            ACCEPT_REVIEW,
        ])
        planner = GeneratedSolvePlanner(
            llm, _validator(), model=llm.model, max_repair_attempts=3,
            max_tokens_per_call=100,
        )
        with tempfile.TemporaryDirectory() as directory:
            result = await planner.generate(
                _request(), artifact_dir=Path(directory),
                budget=BudgetSpec(max_tokens=10000),
            )
            self.assertIsNotNone(result.generated)
            assert result.generated is not None
            self.assertEqual(result.generated.generation_attempt, 2)
            self.assertEqual(result.usage.llm_calls, 5)
            self.assertTrue(result.natural_language_plan_path)
            plan_path = Path(result.natural_language_plan_path or "")
            self.assertEqual(plan_path.read_text().strip(), NATURAL_LANGUAGE_PLAN)
            self.assertTrue((plan_path.parent / "planner_prompt.md").exists())
            self.assertTrue((plan_path.parent / "raw_response.md").exists())
            self.assertTrue((plan_path.parent / "usage.json").exists())
            self.assertTrue((plan_path.parent / "review.json").exists())
            handoff_path = plan_path.parent / "code_generation_handoff.md"
            self.assertTrue(handoff_path.exists())
            handoff = handoff_path.read_text()
            self.assertIn("## Plan", handoff)
            self.assertIn("## Appendix A: Orchestration context (`ctx`) API", handoff)
            self.assertLess(handoff.index("## Plan"), handoff.index("## Appendix A"))
            self.assertIn("### Read-only context values", handoff)
            self.assertIn("### `await ctx.call(role, instruction", handoff)
            self.assertIn("**Purpose:** run one declared worker role", handoff)
            self.assertIn("**Returns:** the role's plain Python value", handoff)
            self.assertNotIn("### `ctx.spawn(role, instruction", handoff)
            self.assertTrue((
                plan_path.parent / "attempts" / "attempt_01" / "review" / "verdict.json"
            ).exists())
            plan_review_prompt = (
                plan_path.parent / "attempts" / "attempt_01" / "review" / "review_prompt.md"
            ).read_text()
            plan_review_system = (
                plan_path.parent / "attempts" / "attempt_01" / "review" / "system_prompt.md"
            ).read_text()
            self.assertIn("suggested_changes", plan_review_prompt)
            self.assertIn("selected agent exists in the supplied catalog", plan_review_prompt)
            self.assertIn("does not require all available agents", plan_review_prompt)
            self.assertIn("declared result contract", plan_review_prompt)
            self.assertIn("do not stop reviewing after the first defect", plan_review_system)
            self.assertTrue((Path(result.attempts_path) / "attempt_01" / "solve.py").exists())
            first_validation = json.loads(
                (Path(result.attempts_path) / "attempt_01" / "validation.json").read_text()
            )
            self.assertFalse(first_validation["accepted"])
            self.assertTrue((Path(result.attempts_path) / "attempt_02" / "raw_response.md").exists())
            self.assertTrue((
                Path(result.attempts_path) / "attempt_02" / "review" / "verdict.json"
            ).exists())
            code_review_prompt = (
                Path(result.attempts_path) / "attempt_02" / "review" / "review_prompt.md"
            ).read_text()
            code_review_system = (
                Path(result.attempts_path) / "attempt_02" / "review" / "system_prompt.md"
            ).read_text()
            self.assertIn("suggested_changes", code_review_prompt)
            self.assertIn("do not stop reviewing after the first defect", code_review_system)
            self.assertIn("Completely ignore robustness", code_review_system)
            self.assertIn("runnability and fidelity", code_review_prompt)
            self.assertIn("Every called role is declared", code_review_prompt)
            self.assertIn("reads each role result according to its declared schema", code_review_prompt)
            self.assertIn("does not force decomposition", code_review_prompt)
            prompt = (
                Path(result.attempts_path) / "attempt_01" / "orchestrator_prompt.md"
            ).read_text()
            self.assertTrue(prompt.startswith("# Orchestration generation request"))
            self.assertIn("## Available roles", prompt)
            self.assertNotIn("Approved natural-language orchestration plan", prompt)
            self.assertEqual(prompt.count("## Task\n"), 1)
            self.assertNotIn("\n# Task\n", prompt)
            self.assertIn("## Plan", prompt)
            self.assertNotIn("# Orchestration Plan", prompt)
            self.assertIn("call researcher agents", prompt)
            self.assertIn("#### `researcher`", prompt)
            self.assertIn("constraint-by-constraint verification", prompt)
            self.assertIn("source URLs", prompt)
            self.assertNotIn('"environment":', prompt)
            self.assertIn("## Generation constraints", prompt)
            self.assertNotIn("## Complete reference examples", prompt)
            self.assertNotIn("### Example 1:", prompt)
            self.assertNotIn("### Example 2:", prompt)
            self.assertNotIn("### Example 3:", prompt)
            self.assertNotIn("### API micro-patterns", prompt)
            self.assertNotIn("**Sequential role call**", prompt)
            self.assertNotIn('verdict = await ctx.call("verifier"', prompt)
            self.assertNotIn("The task is data to solve through agents", prompt)
            self.assertNotIn("Applicable when:", prompt)
            self.assertNotIn("Guidance:", prompt)
            self.assertIn("## Appendix A: Orchestration context (`ctx`) API", prompt)
            self.assertIn("This appendix is part of the planner handoff", prompt)
            self.assertIn("### Read-only context values", prompt)
            self.assertIn("### `await ctx.call(role, instruction", prompt)
            self.assertIn("**Purpose:** run one declared worker role", prompt)
            self.assertIn("**Returns:** the role's plain Python value", prompt)
            self.assertLess(prompt.index("## Plan"), prompt.index("## Appendix A"))
            self.assertLess(prompt.index("## Appendix A"), prompt.index("## Available roles"))
            self.assertNotIn("Parsed result contract", prompt)
            self.assertNotIn('"additionalProperties"', prompt)
            started_event = next(
                event for event in result.events
                if event["event_type"] == "orchestrator_generation_started"
            )
            self.assertEqual(started_event["payload"]["model"], "fake-planner")
            self.assertEqual(started_event["payload"]["model_role"], "orchestrator")
            self.assertEqual(
                started_event["payload"]["model_input"]["user_message"],
                prompt,
            )
            plan_started_event = next(
                event for event in result.events
                if event["event_type"] == "natural_language_plan_started"
            )
            planning_prompt = plan_started_event["payload"]["model_input"]["user_message"]
            self.assertEqual(plan_started_event["payload"]["model_role"], "planner")
            self.assertIn("when the stage begins", planning_prompt.lower())
            self.assertIn("which agent is called", planning_prompt.lower())
            self.assertNotIn("## Runtime capabilities", planning_prompt)
            self.assertNotIn("## Available tools", planning_prompt)
            self.assertNotIn("## Budget", planning_prompt)
            self.assertNotIn("ctx.", planning_prompt)
            self.assertNotIn("candidate_table", planning_prompt)
            self.assertIn("Use only agents listed above", planning_prompt)
            self.assertIn("Do not require every available agent", planning_prompt)
            self.assertIn("Preserve useful prior results", planning_prompt)
            plan_completed_event = next(
                event for event in result.events
                if event["event_type"] == "natural_language_plan_completed"
            )
            self.assertEqual(
                plan_completed_event["payload"]["model_output"],
                NATURAL_LANGUAGE_PLAN,
            )
            self.assertTrue(plan_completed_event["payload"]["review"]["accepted"])
            completed_event = next(
                event for event in result.events
                if event["event_type"] == "orchestrator_generation_completed"
            )
            self.assertIn("async def solve", completed_event["payload"]["model_output"])
            report = json.loads(
                (Path(directory) / "validation_report.json").read_text()
            )
            self.assertTrue(report["accepted"])

            solve = _validator().compile(result.generated.source)
            runtime_result = await ScriptRuntime(
                FakeWorkerExecutor(default_handler=lambda req: "answer"),
            ).run(solve, task="question")
            self.assertEqual(runtime_result.value, "answer")

    async def test_stops_after_three_repairs(self):
        bad = "async def solve(ctx):\n    return eval('1')\n"
        llm = FakePlannerLLM([NATURAL_LANGUAGE_PLAN, ACCEPT_REVIEW] + [bad] * 4)
        planner = GeneratedSolvePlanner(
            llm, _validator(), model=llm.model, max_repair_attempts=3,
            max_tokens_per_call=100,
        )
        with tempfile.TemporaryDirectory() as directory:
            result = await planner.generate(
                _request(), artifact_dir=Path(directory),
                budget=BudgetSpec(max_tokens=10000),
            )
            self.assertIsNone(result.generated)
            self.assertEqual(llm.calls, 6)
            self.assertTrue(result.fallback_reason)
            for attempt in range(1, 5):
                self.assertTrue(
                    (Path(result.attempts_path) / f"attempt_{attempt:02d}" / "solve.py").exists()
                )

    async def test_rejected_plan_is_regenerated_before_code_generation(self):
        rejected_plan = "Call the solver immediately and then return its answer."
        revised_plan = (
            "Call a researcher first. Once evidence exists, call the solver, then the "
            "verifier, and finally the synthesizer."
        )
        critique = "The plan omits evidence gathering and independent verification."
        suggested_change = "Add a research stage before candidate construction."
        source = "async def solve(ctx):\n    return 'ok'\n"
        llm = FakePlannerLLM([
            rejected_plan,
            json.dumps({
                "accepted": False,
                "critique": critique,
                "issues": [critique],
                "suggested_changes": [suggested_change],
            }),
            revised_plan,
            ACCEPT_REVIEW,
            source,
            ACCEPT_REVIEW,
        ])
        planner = GeneratedSolvePlanner(
            llm, _validator(), model=llm.model, max_repair_attempts=1,
            max_tokens_per_call=100,
        )
        with tempfile.TemporaryDirectory() as directory:
            result = await planner.generate(
                _request(), artifact_dir=Path(directory),
                budget=BudgetSpec(max_tokens=10000),
            )
            self.assertIsNotNone(result.generated)
            self.assertEqual(llm.calls, 6)
            self.assertEqual(
                Path(result.natural_language_plan_path or "").read_text().strip(),
                revised_plan,
            )
            second_plan_prompt = (
                Path(directory) / "plan" / "attempts" / "attempt_02" / "planner_prompt.md"
            ).read_text()
            self.assertIn(critique, second_plan_prompt)
            self.assertIn(suggested_change, second_plan_prompt)
            rejected_verdict = json.loads((
                Path(directory) / "plan" / "attempts" / "attempt_01"
                / "review" / "verdict.json"
            ).read_text())
            self.assertEqual(rejected_verdict["suggested_changes"], [suggested_change])
            self.assertEqual(rejected_verdict["issues"], [critique])
            self.assertIn(rejected_plan, second_plan_prompt)
            self.assertEqual(
                sum(
                    event["event_type"] == "natural_language_plan_started"
                    for event in result.events
                ),
                2,
            )
            self.assertEqual(
                sum(
                    event["event_type"] == "orchestrator_generation_started"
                    for event in result.events
                ),
                1,
            )

    async def test_rejected_code_repairs_code_without_regenerating_plan(self):
        first_source = "async def solve(ctx):\n    return 'first'\n"
        repaired_source = "async def solve(ctx):\n    return 'repaired'\n"
        critique = "The program has two material orchestration defects."
        issues = [
            "The program skips the verifier stage required by the approved plan.",
            "The program returns without calling the synthesizer.",
        ]
        suggested_changes = [
            "Call the verifier after the solver and before synthesis.",
            "Pass the accepted candidate to the synthesizer before returning.",
        ]
        llm = FakePlannerLLM([
            NATURAL_LANGUAGE_PLAN,
            ACCEPT_REVIEW,
            first_source,
            json.dumps({
                "accepted": False,
                "critique": critique,
                "issues": issues,
                "suggested_changes": suggested_changes,
            }),
            repaired_source,
            ACCEPT_REVIEW,
        ])
        planner = GeneratedSolvePlanner(
            llm, _validator(), model=llm.model, max_repair_attempts=1,
            max_tokens_per_call=100,
        )
        with tempfile.TemporaryDirectory() as directory:
            result = await planner.generate(
                _request(), artifact_dir=Path(directory),
                budget=BudgetSpec(max_tokens=10000),
            )
            self.assertIsNotNone(result.generated)
            assert result.generated is not None
            self.assertEqual(result.generated.generation_attempt, 2)
            self.assertEqual(result.generated.source, repaired_source)
            self.assertEqual(
                sum(
                    event["event_type"] == "natural_language_plan_started"
                    for event in result.events
                ),
                1,
            )
            repair_prompt = (
                Path(result.attempts_path) / "attempt_02" / "orchestrator_prompt.md"
            ).read_text()
            self.assertIn(critique, repair_prompt)
            for issue in issues:
                self.assertIn(issue, repair_prompt)
            for suggested_change in suggested_changes:
                self.assertIn(suggested_change, repair_prompt)
            self.assertIn(first_source.strip(), repair_prompt)
            self.assertEqual(
                Path(result.natural_language_plan_path or "").read_text().strip(),
                NATURAL_LANGUAGE_PLAN,
            )

    async def test_code_repairs_receive_all_attempt_history(self):
        sources = [
            "async def solve(ctx):\n    return 'first'\n",
            "async def solve(ctx):\n    return 'second'\n",
            "async def solve(ctx):\n    return 'third'\n",
        ]
        reviews = [
            {
                "accepted": False,
                "critique": "first critique",
                "issues": ["first issue"],
                "suggested_changes": ["first repair"],
            },
            {
                "accepted": False,
                "critique": "second critique",
                "issues": ["second issue"],
                "suggested_changes": ["second repair"],
            },
            json.loads(ACCEPT_REVIEW),
        ]
        llm = FakePlannerLLM([
            NATURAL_LANGUAGE_PLAN,
            ACCEPT_REVIEW,
            sources[0], json.dumps(reviews[0]),
            sources[1], json.dumps(reviews[1]),
            sources[2], json.dumps(reviews[2]),
        ])
        planner = GeneratedSolvePlanner(
            llm, _validator(), model=llm.model, max_repair_attempts=2,
            max_tokens_per_call=100,
        )
        with tempfile.TemporaryDirectory() as directory:
            result = await planner.generate(
                _request(), artifact_dir=Path(directory),
                budget=BudgetSpec(max_tokens=10000),
            )
            self.assertIsNotNone(result.generated)
            assert result.generated is not None
            self.assertEqual(result.generated.generation_attempt, 3)

            attempts_path = Path(result.attempts_path)
            second_prompt = (
                attempts_path / "attempt_02" / "orchestrator_prompt.md"
            ).read_text()
            third_prompt = (
                attempts_path / "attempt_03" / "orchestrator_prompt.md"
            ).read_text()
            second_review_prompt = (
                attempts_path / "attempt_02" / "review" / "review_prompt.md"
            ).read_text()
            third_review_prompt = (
                attempts_path / "attempt_03" / "review" / "review_prompt.md"
            ).read_text()

            self.assertIn(sources[0].strip(), second_prompt)
            self.assertIn("first issue", second_prompt)
            self.assertIn("first repair", second_prompt)
            self.assertIn(sources[0].strip(), second_review_prompt)

            self.assertIn(sources[0].strip(), third_prompt)
            self.assertIn(sources[1].strip(), third_prompt)
            self.assertIn("first repair", third_prompt)
            self.assertIn("second repair", third_prompt)
            self.assertIn(sources[0].strip(), third_review_prompt)
            self.assertIn(sources[1].strip(), third_review_prompt)
            self.assertIn(sources[2].strip(), third_review_prompt)

            history = json.loads(
                (Path(directory) / "repair_history.json").read_text()
            )
            self.assertEqual(len(history["attempts"]), 3)
            self.assertEqual(
                [entry["attempt"] for entry in history["attempts"]],
                [1, 2, 3],
            )
            self.assertEqual(
                [entry["source"] for entry in history["attempts"]], sources,
            )
            self.assertEqual(
                history["attempts"][0]["review"]["suggested_changes"],
                ["first repair"],
            )
            self.assertEqual(
                history["attempts"][1]["review"]["suggested_changes"],
                ["second repair"],
            )
            self.assertTrue(history["attempts"][2]["review"]["accepted"])
            self.assertTrue(all(
                entry["static_validation"]["accepted"]
                for entry in history["attempts"]
            ))

    async def test_respects_execution_reserve(self):
        llm = FakePlannerLLM([
            NATURAL_LANGUAGE_PLAN,
            ACCEPT_REVIEW,
            "async def solve(ctx):\n    return 'ok'\n",
            ACCEPT_REVIEW,
        ])
        planner = GeneratedSolvePlanner(
            llm, _validator(), model=llm.model, max_repair_attempts=3,
            max_tokens_per_call=8192,
        )
        with tempfile.TemporaryDirectory() as directory:
            result = await planner.generate(
                _request(), artifact_dir=Path(directory),
                budget=BudgetSpec(max_tokens=5000),
            )
        self.assertIsNone(result.generated)
        self.assertEqual(llm.calls, 0)

    async def test_empty_natural_language_plan_prevents_code_generation(self):
        llm = FakePlannerLLM([""])
        planner = GeneratedSolvePlanner(
            llm, _validator(), model=llm.model, max_repair_attempts=0,
            max_tokens_per_call=100,
        )
        with tempfile.TemporaryDirectory() as directory:
            result = await planner.generate(
                _request(), artifact_dir=Path(directory),
                budget=BudgetSpec(max_tokens=10000),
            )
            self.assertIsNone(result.generated)
            self.assertEqual(llm.calls, 1)
            self.assertIn("empty plan", result.fallback_reason or "")
            self.assertFalse((Path(directory) / "attempts" / "attempt_01").exists())
            validation = json.loads(
                (Path(directory) / "validation_report.json").read_text()
            )
            self.assertEqual(
                validation["issues"][0]["code"],
                "natural_language_plan_empty",
            )

    async def test_requires_budget_for_plan_and_code_calls(self):
        llm = FakePlannerLLM([
            NATURAL_LANGUAGE_PLAN,
            ACCEPT_REVIEW,
            "async def solve(ctx):\n    return 'ok'\n",
        ])
        planner = GeneratedSolvePlanner(
            llm, _validator(), model=llm.model, max_repair_attempts=0,
            max_tokens_per_call=100,
        )
        with tempfile.TemporaryDirectory() as directory:
            result = await planner.generate(
                _request(), artifact_dir=Path(directory),
                budget=BudgetSpec(max_tokens=10000, max_llm_calls=1),
            )
        self.assertIsNone(result.generated)
        self.assertEqual(llm.calls, 0)
        self.assertIn("reserve code-generation budget", result.fallback_reason or "")

    async def test_code_is_rejected_from_natural_language_stage(self):
        llm = FakePlannerLLM([
            "```python\nasync def solve(ctx):\n    return 'answer'\n```",
        ])
        planner = GeneratedSolvePlanner(
            llm, _validator(), model=llm.model, max_repair_attempts=0,
            max_tokens_per_call=100,
        )
        with tempfile.TemporaryDirectory() as directory:
            result = await planner.generate(
                _request(), artifact_dir=Path(directory),
                budget=BudgetSpec(max_tokens=10000),
            )
            validation = json.loads(
                (Path(directory) / "validation_report.json").read_text()
            )
        self.assertIsNone(result.generated)
        self.assertEqual(llm.calls, 1)
        self.assertEqual(
            validation["issues"][0]["code"],
            "natural_language_plan_contains_code",
        )

    async def test_research_report_keeps_success_and_failure_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact_dir = Path(directory)
            (artifact_dir / "solve.py").write_text(
                "async def solve(ctx):\n    try:\n        return await ctx.call('solver', ctx.task)\n    except Exception:\n        return 'Insufficient information'\n"
            )
            (artifact_dir / "validation_report.json").write_text(json.dumps({
                "accepted": True, "issues": [], "source_hash": "hash",
            }))
            path = write_generation_research_report(
                artifact_dir,
                question_id="1", predicted_answer="A", ground_truth="A",
                is_correct=True, strict_exact_match=True,
                unordered_team_match=False, judge_method="exact_match",
                usage={"tokens": 100}, budget_violations=[],
                pipeline_error=None, fallback_reason=None,
            )
            report = json.loads(path.read_text())
            self.assertEqual(report["classification"], "promising")
            self.assertTrue(report["eligible_for_offline_evolution"])
            self.assertEqual(report["source_analysis"]["role_calls"], {"solver": 1})


if __name__ == "__main__":
    unittest.main()
