"""Validate a container before starting an expensive benchmark run."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from benchmarks.core.registry import get_config, load_questions
from benchmarks.judges import get_semantic_judge_model, has_judge
from workflows.react_base.profile import load_profile

_REQUIRED_ENV = (
    "SERPER_API_KEY",
    "JINA_API_KEY",
    "E2B_API_KEY",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="browsecomp_200")
    args = parser.parse_args()

    errors: list[str] = []
    cfg = get_config(args.benchmark)
    profile = load_profile("default")
    agent_llm = profile["llm"]
    summary_llm = profile["summary_llm"]

    try:
        questions = load_questions(args.benchmark)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        errors.append(str(exc))
        questions = []

    missing_env = [name for name in _REQUIRED_ENV if not os.getenv(name)]
    judge_api_key_set = bool(os.getenv("JUDGE_API_KEY"))
    judge_base_url_set = bool(os.getenv("JUDGE_BASE_URL"))
    dedicated_judge = judge_api_key_set and judge_base_url_set
    fallback_judge = bool(
        os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_BASE_URL")
    )
    if missing_env:
        errors.append("missing environment variables: " + ", ".join(missing_env))
    if not agent_llm.get("api_key"):
        errors.append(
            f"agent API key is empty for model {agent_llm.get('model')!r}"
        )
    if not agent_llm.get("base_url"):
        errors.append(
            f"agent base URL is empty for model {agent_llm.get('model')!r}"
        )
    if not summary_llm.get("api_key"):
        errors.append(
            f"summary API key is empty for model {summary_llm.get('model')!r}"
        )
    if not summary_llm.get("base_url"):
        errors.append(
            f"summary base URL is empty for model {summary_llm.get('model')!r}"
        )
    if judge_api_key_set != judge_base_url_set:
        errors.append(
            "JUDGE_API_KEY and JUDGE_BASE_URL must either both be set or both be absent"
        )
    if not dedicated_judge and not fallback_judge:
        errors.append(
            "the judge requires either JUDGE_API_KEY/JUDGE_BASE_URL or "
            "OPENAI_API_KEY/OPENAI_BASE_URL"
        )
    if not has_judge(args.benchmark):
        errors.append(f"no judge registered for {args.benchmark!r}")
    if len(questions) != 200:
        errors.append(
            f"expected 200 questions for {args.benchmark!r}, found {len(questions)}"
        )

    print(f"benchmark: {args.benchmark}")
    print(f"dataset:   {Path('benchmarks/datasets') / cfg.key / cfg.jsonl}")
    print(f"questions: {len(questions)}")
    print(f"pipeline:  {cfg.default_pipeline}")
    print(f"agent LLM: {agent_llm.get('model')}")
    print(f"summary:   {summary_llm.get('model')}")
    print(f"judge:     {'registered' if has_judge(args.benchmark) else 'missing'}")
    print(f"judge API: {'JUDGE_*' if dedicated_judge else 'OPENAI_* fallback'}")
    print(f"judge LLM: {get_semantic_judge_model()}")
    print(f"env:       {'complete' if not missing_env else 'incomplete'}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    print("preflight: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
