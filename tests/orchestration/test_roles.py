"""Tests for predefined roles and the AgentHarness worker adapter."""

from __future__ import annotations

import json
import unittest

from agent_harness.core.llm import LLMResponse
from agent_harness.core.tool import Tool
from agent_harness.core.runtime.registries.agents import AgentRegistry
from agent_harness.models.agent_definition import AgentDefinition
from agent_harness.orchestration import (
    AgentHarnessWorkerExecutor,
    InvalidRoleOutput,
)
from agent_harness.orchestration.agent_executor import (
    normalize_confidence,
    parse_role_output,
)
from agent_harness.orchestration.roles import (
    BUILTIN_ORCHESTRATION_ROLES,
    DECOMPOSER_ROLE,
    RESEARCHER_ROLE,
    SOLVER_ROLE,
    SYNTHESIZER_ROLE,
    VERIFIER_ROLE,
    register_builtin_orchestration_roles,
)
from agent_harness.orchestration.types import JobRequest


class _StubLLM:
    model = "stub-model"

    def __init__(self, content: str) -> None:
        self.content = content
        self.messages = []

    async def chat(self, messages, **_kwargs):
        self.messages.append(messages)
        return LLMResponse(content=self.content, model=self.model)

    async def stream(self, messages, **kwargs):
        del messages, kwargs
        if False:
            yield


class _ToolThenAnswerLLM:
    model = "stub-model"

    def __init__(self) -> None:
        self.tools_seen = []

    async def chat(self, _messages, **kwargs):
        tools = kwargs.get("tools")
        self.tools_seen.append(tools)
        if tools:
            return LLMResponse(
                content="I will inspect the evidence.",
                tool_calls=[{
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"query": "evidence"}',
                    },
                }],
                model=self.model,
            )
        return LLMResponse(content="final evidence", model=self.model)

    async def stream(self, messages, **kwargs):
        del messages, kwargs
        if False:
            yield


class _StubbornToolLLM:
    model = "stub-model"

    def __init__(self) -> None:
        self.calls = 0
        self.tools_seen = []
        self.messages_seen = []

    async def chat(self, messages, **kwargs):
        self.calls += 1
        self.tools_seen.append(kwargs.get("tools"))
        self.messages_seen.append(messages)
        if self.calls <= 2:
            return LLMResponse(
                content="I need one more lookup.",
                tool_calls=[{
                    "id": f"call-{self.calls}",
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "arguments": '{"query": "evidence"}',
                    },
                }],
                model=self.model,
            )
        return LLMResponse(content="forced final evidence", model=self.model)

    async def stream(self, messages, **kwargs):
        del messages, kwargs
        if False:
            yield


class _RepairingLLM:
    model = "stub-model"

    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, _messages, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            return LLMResponse(
                content="Accepted. There were no errors.",
                model=self.model,
                usage={"prompt_tokens": 3, "completion_tokens": 1},
            )
        return LLMResponse(
            content=json.dumps({
                "accepted": False,
                "critique": "condition 2 needs a source",
            }),
            model=self.model,
            usage={"prompt_tokens": 4, "completion_tokens": 2},
        )

    async def stream(self, messages, **kwargs):
        del messages, kwargs
        if False:
            yield


class _StubResources:
    def __init__(self, llm, tools) -> None:
        self.llm = llm
        self.tools = tools

    def get_llm(self, _role):
        return self.llm

    def get_tools_for_role(self, _role):
        return self.tools


class RoleContractTests(unittest.TestCase):
    def test_builtin_catalog_registers_expected_roles(self) -> None:
        registry = AgentRegistry()

        register_builtin_orchestration_roles(registry)

        registered_roles = {
            definition.role_id for definition in registry.list_all()
        }
        self.assertIn("decomposer", registered_roles)
        self.assertEqual(
            registered_roles,
            {
                "decomposer",
                "researcher",
                "solver",
                "verifier",
                "synthesizer",
                "general",
            },
        )
        self.assertEqual(
            [role.role_id for role in BUILTIN_ORCHESTRATION_ROLES],
            [
                "decomposer",
                "researcher",
                "solver",
                "verifier",
                "synthesizer",
                "general",
            ],
        )

    def test_role_permissions_and_turn_limits_match_catalog(self) -> None:
        self.assertEqual(DECOMPOSER_ROLE.allowed_tools, [])
        self.assertEqual(SOLVER_ROLE.allowed_tools, ["run_python_code"])
        self.assertEqual(SYNTHESIZER_ROLE.allowed_tools, [])
        self.assertTrue(
            all(
                definition.metadata["max_turns"] == 25
                for definition in BUILTIN_ORCHESTRATION_ROLES
            )
        )

    def test_decomposer_contract_returns_string_list(self) -> None:
        value = parse_role_output(
            '["research evidence", "calculate result"]',
            DECOMPOSER_ROLE.result_schema,
        )
        self.assertEqual(value, ["research evidence", "calculate result"])

    def test_verifier_contract_returns_acceptance_and_critique(self) -> None:
        value = parse_role_output(
            '{"accepted":false,"critique":"condition 2 needs a source"}',
            VERIFIER_ROLE.result_schema,
        )
        self.assertFalse(value["accepted"])
        self.assertEqual(value["critique"], "condition 2 needs a source")

    def test_verifier_contract_rejects_missing_or_extra_fields(self) -> None:
        with self.assertRaises(InvalidRoleOutput):
            parse_role_output(
                '{"accepted": true}',
                VERIFIER_ROLE.result_schema,
            )
        with self.assertRaises(InvalidRoleOutput):
            parse_role_output(
                '{"accepted":true,"critique":"","classification":"verified"}',
                VERIFIER_ROLE.result_schema,
            )

    def test_confidence_normalization(self) -> None:
        self.assertEqual(normalize_confidence("85%"), (0.85, False))
        self.assertEqual(normalize_confidence(85), (0.85, False))
        self.assertEqual(normalize_confidence("high"), (0.8, True))

    def test_researcher_and_solver_return_plain_text(self) -> None:
        self.assertIsNone(RESEARCHER_ROLE.result_schema)
        self.assertIsNone(SOLVER_ROLE.result_schema)
        self.assertEqual(
            parse_role_output("  evidence with URL  ", RESEARCHER_ROLE.result_schema),
            "evidence with URL",
        )


class AgentHarnessWorkerExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_text_role_degrades_to_incomplete_information(self) -> None:
        registry = AgentRegistry()
        register_builtin_orchestration_roles(registry)
        executor = AgentHarnessWorkerExecutor(
            registry,
            _StubResources(_StubLLM(""), []),
        )

        async def emit(_name: str, _payload: dict) -> None:
            return None

        result = await executor.execute(
            JobRequest(
                job_id="job-empty",
                run_id="run-empty",
                key="empty",
                role="solver",
                instruction="Find evidence",
                request_digest="digest",
                max_turns=1,
            ),
            emit,
        )

        self.assertIsInstance(result.value, str)
        self.assertIn("Insufficient information", result.value)

    async def test_executor_finalizes_after_stubborn_last_turn_tool_call(
        self,
    ) -> None:
        registry = AgentRegistry()
        registry.register(AgentDefinition(
            role_id="stubborn_researcher",
            display_name="Stubborn Researcher",
            system_prompt="Research the request.",
            allowed_tools=["lookup"],
            metadata={"max_turns": 10},
        ))
        llm = _StubbornToolLLM()

        async def lookup(query: str) -> str:
            return f"result for {query}"

        executor = AgentHarnessWorkerExecutor(
            registry,
            _StubResources(
                llm,
                [Tool(
                    name="lookup",
                    description="Look up evidence.",
                    parameters={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                    func=lookup,
                )],
            ),
        )
        emitted = []

        async def emit(name: str, payload: dict) -> None:
            emitted.append((name, payload))

        result = await executor.execute(
            JobRequest(
                job_id="job-stubborn",
                run_id="run-stubborn",
                key="stubborn",
                role="stubborn_researcher",
                instruction="Find the answer",
                request_digest="digest",
                max_turns=2,
            ),
            emit,
        )

        self.assertEqual(result.value, "forced final evidence")
        self.assertGreaterEqual(result.metadata["turns"], 1)
        self.assertEqual(result.metadata["max_turns"], 2)
        self.assertEqual(result.metadata["stopped_by"], "forced_final")
        self.assertTrue(result.metadata["forced_final"])
        self.assertIn(
            "agent_loop_forced_final_started",
            [name for name, _payload in emitted],
        )
        self.assertIsNotNone(llm.tools_seen[0])
        # Reserved final turn and the evidence-summary call are both tool-free.
        self.assertEqual(llm.tools_seen[1:], [None, None])
        self.assertTrue(any(
            message.get("role") == "user"
            and "final turn" in message.get("content", "").casefold()
            for history in llm.messages_seen
            for message in history
        ))

    async def test_executor_repairs_structured_output_once(self) -> None:
        registry = AgentRegistry()
        register_builtin_orchestration_roles(registry)
        llm = _RepairingLLM()
        executor = AgentHarnessWorkerExecutor(
            registry,
            _StubResources(llm, []),
        )
        emitted = []

        async def emit(name: str, payload: dict) -> None:
            emitted.append((name, payload))

        result = await executor.execute(
            JobRequest(
                job_id="job-repair",
                run_id="run-repair",
                key="verify",
                role="verifier",
                instruction="Check this answer",
                request_digest="digest",
            ),
            emit,
        )

        self.assertFalse(result.value["accepted"])
        self.assertEqual(
            result.value["critique"],
            "condition 2 needs a source",
        )
        self.assertTrue(result.metadata["output_repaired"])
        self.assertEqual(llm.calls, 2)
        self.assertEqual(result.usage["prompt_tokens"], 7)
        self.assertEqual(result.usage["completion_tokens"], 3)
        self.assertEqual(result.metadata["llm_calls"], 2)
        self.assertEqual(
            [name for name, _payload in emitted],
            [
                "agent_loop_started",
                "role_output_repair_started",
                "role_output_repair_succeeded",
                "agent_loop_succeeded",
            ],
        )

    async def test_executor_reserves_final_turn_for_an_answer(self) -> None:
        registry = AgentRegistry()
        registry.register(AgentDefinition(
            role_id="bounded_researcher",
            display_name="Bounded Researcher",
            system_prompt="Research the request.",
            allowed_tools=["lookup"],
            metadata={"max_turns": 2},
        ))
        llm = _ToolThenAnswerLLM()

        async def lookup(query: str) -> str:
            return f"result for {query}"

        executor = AgentHarnessWorkerExecutor(
            registry,
            _StubResources(
                llm,
                [Tool(
                    name="lookup",
                    description="Look up evidence.",
                    parameters={
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                    func=lookup,
                )],
            ),
        )

        async def emit(_name: str, _payload: dict) -> None:
            return None

        result = await executor.execute(
            JobRequest(
                job_id="job-final-turn",
                run_id="run-final-turn",
                key="bounded",
                role="bounded_researcher",
                instruction="Find the answer",
                request_digest="digest",
            ),
            emit,
        )

        self.assertEqual(result.value, "final evidence")
        self.assertEqual(result.metadata["turns"], 2)
        self.assertEqual(result.metadata["stopped_by"], "no_tool")
        self.assertIsNotNone(llm.tools_seen[0])
        self.assertIsNone(llm.tools_seen[1])

if __name__ == "__main__":
    unittest.main()
