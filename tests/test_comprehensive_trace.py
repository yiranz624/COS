from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from agent_harness.components.observers.trajectory import (
    TrajectoryFileObserver,
)
from agent_harness.core.loop_types import (
    AgentLoopResult,
    LoopConfig,
    LoopPolicy,
    ToolResult,
    TurnContext,
)
from agent_harness.core.llm import LLMResponse
from agent_harness.core.tool import Tool
from agent_harness.core.runtime.loop.agent_loop import run_agent_loop
from agent_harness.core.runtime.loop.llm_client import call_llm
from agent_harness.orchestration.events import OrchestrationEventType
from agent_harness.orchestration.store import InMemoryOrchestrationStore


class ComprehensiveTraceTests(unittest.IsolatedAsyncioTestCase):
    async def test_tool_call_cap_emits_results_for_every_declared_call(self) -> None:
        test_case = self
        executed: list[str] = []

        class SixToolsThenAnswerLLM:
            model = "deepseek-test"

            def __init__(self) -> None:
                self.calls = 0

            async def chat(self, messages, **_kwargs):
                self.calls += 1
                if self.calls == 1:
                    return LLMResponse(
                        content="I will inspect six sources.",
                        tool_calls=[
                            {
                                "id": f"call_{index}",
                                "type": "function",
                                "function": {
                                    "name": "lookup",
                                    "arguments": json.dumps({"q": str(index)}),
                                },
                            }
                            for index in range(6)
                        ],
                    )

                assistant = messages[-7]
                tool_results = messages[-6:]
                test_case.assertEqual(len(assistant["tool_calls"]), 6)
                test_case.assertEqual(
                    [item["tool_call_id"] for item in tool_results],
                    [f"call_{index}" for index in range(6)],
                )
                test_case.assertIn(
                    "max_tool_calls_per_turn=5",
                    tool_results[-1]["content"],
                )
                return LLMResponse(content="final answer")

        async def lookup(q: str) -> str:
            executed.append(q)
            return f"evidence:{q}"

        result = await run_agent_loop(
            system_prompt="system",
            user_message="question",
            llm=SixToolsThenAnswerLLM(),
            tools=[Tool(
                name="lookup",
                description="lookup",
                parameters={
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
                func=lookup,
            )],
            config=LoopConfig(
                max_turns=2,
                max_tool_calls_per_turn=5,
                loop_policy=LoopPolicy(no_tool_behavior="stop"),
            ),
        )

        self.assertEqual(result.final_content, "final answer")
        self.assertEqual(executed, ["0", "1", "2", "3", "4"])

    async def test_dsml_fallback_is_canonicalized_in_history(self) -> None:
        test_case = self

        class DSMLThenAnswerLLM:
            model = "deepseek-test"

            def __init__(self) -> None:
                self.calls = 0

            async def chat(self, messages, **_kwargs):
                self.calls += 1
                if self.calls == 1:
                    return LLMResponse(content=(
                        '<｜DSML｜tool_calls>'
                        '<｜DSML｜invoke name="lookup">'
                        '<｜DSML｜parameter name="q" string="true">fact'
                        '</｜DSML｜parameter></｜DSML｜invoke>'
                        '</｜DSML｜tool_calls>'
                    ))
                assistant = messages[-2]
                tool_result = messages[-1]
                test_case.assertIn("tool_calls", assistant)
                test_case.assertEqual(
                    assistant["tool_calls"][0]["function"]["name"], "lookup",
                )
                test_case.assertEqual(tool_result["role"], "tool")
                return LLMResponse(content="final answer")

        async def lookup(q: str) -> str:
            return f"evidence:{q}"

        result = await run_agent_loop(
            system_prompt="system",
            user_message="question",
            llm=DSMLThenAnswerLLM(),
            tools=[Tool(
                name="lookup",
                description="lookup",
                parameters={
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
                func=lookup,
            )],
            config=LoopConfig(
                max_turns=2,
                loop_policy=LoopPolicy(no_tool_behavior="stop"),
            ),
        )

        self.assertEqual(result.final_content, "final answer")

    async def test_each_physical_llm_attempt_is_reported(self) -> None:
        class RetryOnceLLM:
            model = "test-model"

            def __init__(self) -> None:
                self.calls = 0

            async def chat(self, messages, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("temporary failure")
                return LLMResponse(content="成功")

        events: list[dict] = []

        async def capture(event: dict) -> None:
            events.append(event)

        response = await call_llm(
            RetryOnceLLM(),
            [{"role": "user", "content": "问题"}],
            timeout=1,
            max_retries=2,
            turn=1,
            retry_wait_fixed=0,
            on_attempt=capture,
        )
        self.assertEqual(response.content, "成功")
        self.assertEqual(
            [event["状态"] for event in events],
            ["开始", "失败", "开始", "成功"],
        )

    async def test_agent_trace_is_flushed_step_by_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            observer = TrajectoryFileObserver(
                Path(tmp),
                filename="job",
                formats=("jsonl",),
                model_name="test-model",
                system_prompt="系统提示词",
                user_message="用户问题",
            )
            await observer.on_loop_start(LoopConfig(
                task_id="task-1", role_id="solver", max_turns=3,
            ))
            path = Path(tmp) / "job.jsonl"
            self.assertTrue(path.exists())
            self.assertEqual(len(path.read_text().splitlines()), 1)

            ctx = TurnContext(
                turn=1,
                max_turns=3,
                task_id="task-1",
                role_id="solver",
                ai_text="需要搜索",
                thinking="分析过程",
                tool_calls=[{
                    "id": "call-1",
                    "name": "web_search",
                    "args": {"q": "测试"},
                }],
                messages=[],
                usage={"prompt_tokens": 10, "completion_tokens": 5},
                metadata={},
                llm_input_messages=[
                    {"role": "system", "content": "系统提示词"},
                    {"role": "user", "content": "用户问题"},
                ],
                llm_tools=[{"type": "function"}],
                llm_duration_ms=123,
                rejected_tool_calls=[{
                    "id": "rejected-1",
                    "name": "web_search",
                    "args": {"q": "不得执行"},
                }],
            )
            await observer.on_llm_response(ctx)
            await observer.on_tool_call(ctx, ctx.tool_calls[0])
            await observer.on_tool_result(ctx, ToolResult(
                name="web_search",
                args={"q": "测试"},
                result="搜索结果",
                duration_ms=45,
                tool_call_id="call-1",
                is_error=False,
            ))

            records = [
                json.loads(line) for line in path.read_text().splitlines()
            ]
            self.assertEqual(
                [record["t"] for record in records],
                ["start", "llm", "tool_call", "result"],
            )
            self.assertEqual(records[1]["模型调用耗时毫秒"], 123)
            self.assertEqual(
                records[1]["最终轮被拒绝的工具调用"][0]["name"],
                "web_search",
            )
            self.assertIn("未执行", records[1]["工具调用拒绝原因"])
            self.assertEqual(records[1]["模型输入消息"][1]["content"], "用户问题")
            self.assertEqual(records[2]["工具输入参数"], {"q": "测试"})
            self.assertEqual(records[3]["工具返回信息"], "搜索结果")

            await observer.on_loop_end(AgentLoopResult(
                messages=[],
                final_content="最终答案",
                turns_used=1,
                tool_calls_count=1,
                stopped_by="no_tool",
            ))
            final = json.loads(path.read_text().splitlines()[-1])
            self.assertEqual(final["事件"], "Agent循环结束")
            self.assertEqual(final["最终输出"], "最终答案")

    async def test_no_tool_events_when_no_tool_was_called(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            observer = TrajectoryFileObserver(
                Path(tmp), filename="job", formats=("jsonl",),
            )
            await observer.on_loop_start(LoopConfig(task_id="task-1"))
            await observer.on_llm_response(TurnContext(
                turn=1,
                max_turns=1,
                task_id="task-1",
                role_id="general",
                ai_text="直接答案",
                thinking="",
                tool_calls=[],
                messages=[],
                usage={"total_tokens": 12},
                metadata={},
                llm_input_messages=[{"role": "user", "content": "问题"}],
                llm_duration_ms=10,
            ))
            records = [
                json.loads(line)
                for line in (Path(tmp) / "job.jsonl").read_text().splitlines()
            ]
            self.assertNotIn("tool_call", [record["t"] for record in records])
            self.assertNotIn("模型请求的工具调用", records[-1])
            await observer.on_loop_end(AgentLoopResult(
                messages=[],
                final_content="直接答案",
                turns_used=1,
                stopped_by="no_tool",
            ))

    async def test_orchestration_events_are_written_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "orchestration_live.jsonl"
            store = InMemoryOrchestrationStore(
                "run-1", live_trace_path=path,
            )
            store.append_event(
                OrchestrationEventType.RUN_STARTED,
                payload={"任务": "测试"},
            )
            self.assertTrue(path.exists())
            first = json.loads(path.read_text().splitlines()[0])
            self.assertEqual(first["事件"], "编排运行开始")
            self.assertEqual(first["详细信息"], {"任务": "测试"})


if __name__ == "__main__":
    unittest.main()
