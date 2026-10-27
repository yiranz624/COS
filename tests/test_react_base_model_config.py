from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from plugins.tools.web_fetch import _summary_llm_base_url, _summary_llm_config
from workflows.react_base.profile import create_llm, load_profile


class ReactBaseModelConfigTests(unittest.TestCase):
    def test_model_settings_are_shared_by_agent_profiles(self) -> None:
        env = {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_BASE_URL": "https://example.test/v1",
        }
        with patch.dict(os.environ, env, clear=False):
            os.environ["REACT_BASE_MODEL_CONFIG"] = "gpt_5_mini.yaml"
            default = load_profile("default")
            keep5 = load_profile("keep5")

        for profile in (default, keep5):
            self.assertEqual(profile["llm"]["model"], "gpt-5-mini")
            self.assertEqual(profile["llm"]["provider"], "openai")
            self.assertEqual(profile["llm"]["api_key"], "test-key")
            self.assertEqual(
                profile["llm"]["base_url"],
                "https://example.test/v1",
            )
            self.assertEqual(profile["llm"]["max_tokens"], 32768)
            self.assertEqual(profile["summary_llm"]["model"], "gpt-5-mini")
            self.assertEqual(profile["summary_llm"]["provider"], "openai")
            self.assertEqual(
                profile["summary_llm"]["base_url"],
                "https://example.test/v1",
            )

        self.assertEqual(default["agent"]["keep_tool_result"], -1)
        self.assertEqual(keep5["agent"]["keep_tool_result"], 5)

    def test_closed_book_profile_disables_all_tools(self) -> None:
        env = {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_BASE_URL": "https://example.test/v1",
            "REACT_BASE_MODEL_CONFIG": (
                "openrouter_deepseek_v4_flash.yaml"
            ),
        }
        with patch.dict(os.environ, env, clear=False):
            profile = load_profile("closed_book")

        self.assertEqual(profile["agent"]["allowed_tools"], [])
        self.assertEqual(profile["agent"]["main_max_turns"], 25)

    def test_deepseek_v4_flash_model_settings(self) -> None:
        env = {
            "REACT_BASE_MODEL_CONFIG": "deepseek_v4_flash.yaml",
            "DEEPSEEK_API_KEY": "deepseek-test-key",
            "DEEPSEEK_BASE_URL": "https://api.deepseek.test",
        }
        with patch.dict(os.environ, env, clear=False):
            profile = load_profile("default")
            client = create_llm(profile)

        llm = profile["llm"]
        self.assertEqual(llm["model"], "deepseek-v4-flash")
        self.assertEqual(llm["provider"], "deepseek")
        self.assertEqual(llm["api_key"], "deepseek-test-key")
        self.assertEqual(llm["base_url"], "https://api.deepseek.test")
        self.assertEqual(llm["max_tokens_param"], "max_tokens")
        self.assertEqual(
            llm["extra_body"],
            {
                "thinking": {"type": "enabled"},
                "reasoning_effort": "high",
            },
        )
        self.assertEqual(client.model, "deepseek-v4-flash")
        self.assertEqual(client.max_tokens_param, "max_tokens")
        self.assertIsNone(client.parallel_tool_calls)
        summary = profile["summary_llm"]
        self.assertEqual(summary["model"], "deepseek-v4-flash")
        self.assertEqual(summary["provider"], "deepseek")
        self.assertEqual(summary["api_key"], "deepseek-test-key")
        self.assertEqual(summary["base_url"], "https://api.deepseek.test")

    def test_web_fetch_summary_model_comes_from_selected_yaml(self) -> None:
        env = {
            "REACT_BASE_MODEL_CONFIG": "deepseek_v4_flash.yaml",
            "DEEPSEEK_API_KEY": "deepseek-test-key",
            "DEEPSEEK_BASE_URL": "https://api.deepseek.test/v1",
            "SUMMARY_LLM_MODEL": "stale-model",
            "SUMMARY_LLM_MODEL_NAME": "stale-model-name",
        }
        with patch.dict(os.environ, env, clear=False):
            summary = _summary_llm_config()

        self.assertEqual(summary["model"], "deepseek-v4-flash")
        self.assertEqual(summary["provider"], "deepseek")
        self.assertEqual(summary["api_key"], "deepseek-test-key")
        self.assertEqual(
            _summary_llm_base_url(summary),
            "https://api.deepseek.test/v1/chat/completions",
        )

    def test_model_descriptor_can_be_used_as_the_profile(self) -> None:
        env = {
            "DEEPSEEK_API_KEY": "deepseek-test-key",
            "DEEPSEEK_BASE_URL": "https://api.deepseek.test",
        }
        with patch.dict(os.environ, env, clear=False):
            profile = load_profile("deepseek_v4_flash")

        self.assertEqual(profile["llm"]["model"], "deepseek-v4-flash")
        self.assertEqual(profile["llm"]["provider"], "deepseek")
        self.assertEqual(profile["summary_llm"]["model"], "deepseek-v4-flash")
        self.assertEqual(profile["summary_llm"]["provider"], "deepseek")
        self.assertEqual(profile["agent"]["keep_tool_result"], -1)


if __name__ == "__main__":
    unittest.main()
