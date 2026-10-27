from __future__ import annotations

import unittest

from agent_harness.core.llm import LLMResponse
from agent_harness.core.runtime.loop.model_profile import (
    DefaultMessageNormalizer,
    DefaultThinkingParser,
    HistoryPolicy,
    ModelProfile,
)
from agent_harness.core.runtime.loop.tool_call_parser import (
    MultiFormatToolCallParser,
)


class DeepSeekHistoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = ModelProfile(
            model_id="deepseek-v4-flash",
            provider="deepseek",
            thinking_format="reasoning_content",
        )
        self.parser = DefaultThinkingParser()
        self.normalizer = DefaultMessageNormalizer()

    def test_tool_call_history_echoes_reasoning_content(self) -> None:
        response = LLMResponse(
            content="I will search.",
            reasoning_content="I need current evidence.",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": '{"q":"example"}',
                    },
                }
            ],
        )

        parsed = self.parser.extract(response, self.profile)
        history = self.normalizer.to_history(
            response,
            parsed,
            HistoryPolicy(),
        )

        self.assertEqual(
            history["reasoning_content"],
            "I need current evidence.",
        )
        self.assertEqual(history["tool_calls"], response.tool_calls)

    def test_answer_only_history_omits_reasoning_content(self) -> None:
        response = LLMResponse(
            content="Final answer",
            reasoning_content="Internal reasoning",
        )

        parsed = self.parser.extract(response, self.profile)
        history = self.normalizer.to_history(
            response,
            parsed,
            HistoryPolicy(),
        )

        self.assertNotIn("reasoning_content", history)

    def test_dsml_text_tool_call_is_parsed_and_stripped(self) -> None:
        response = LLMResponse(content=(
            '<｜DSML｜tool_calls><｜DSML｜invoke name="web_search">'
            '<｜DSML｜parameter name="q" string="true">joint query'
            '</｜DSML｜parameter><｜DSML｜parameter name="num" '
            'string="false">10</｜DSML｜parameter>'
            '</｜DSML｜invoke></｜DSML｜tool_calls>'
        ))

        calls = MultiFormatToolCallParser().parse(
            response, {"web_search"},
        )

        self.assertEqual(calls[0]["name"], "web_search")
        self.assertEqual(calls[0]["args"], {"q": "joint query", "num": 10})
        self.assertEqual(response.content, "")

    def test_dsml_open_url_aliases_to_web_fetch(self) -> None:
        text = (
            '<｜DSML｜tool_calls><｜DSML｜invoke name="open_url">'
            '<｜DSML｜parameter name="url" string="true">'
            'https://example.com</｜DSML｜parameter>'
            '</｜DSML｜invoke></｜DSML｜tool_calls>'
        )

        calls = MultiFormatToolCallParser().parse_text(text, {"web_fetch"})

        self.assertEqual(calls[0]["name"], "web_fetch")
        self.assertEqual(calls[0]["args"]["url"], "https://example.com")
        self.assertIn("info_to_extract", calls[0]["args"])


if __name__ == "__main__":
    unittest.main()
