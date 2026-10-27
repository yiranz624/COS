"""Run one question through the real react_base single-agent pipeline."""

from __future__ import annotations

import argparse
import asyncio
import json

from dotenv import load_dotenv

from benchmarks.core.kernel_adapter import BenchmarkSession


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--question",
        default="只回答最终数字：17乘以23等于多少？",
        help="Question sent to the single ReAct agent.",
    )
    parser.add_argument(
        "--profile",
        default="smoke",
        help="react_base behavior profile (default: smoke).",
    )
    return parser.parse_args()


async def _run(question: str, profile: str) -> dict:
    async with BenchmarkSession() as session:
        return await session.run(
            question,
            pipeline_id="react_base",
            meta={
                "profile": profile,
                "question_id": "single-agent-smoke",
            },
        )


def main() -> None:
    args = _parse_args()
    load_dotenv()
    state = asyncio.run(_run(args.question, args.profile))
    answer = str(state.get("final_answer") or "").strip()
    if not answer:
        raise RuntimeError(
            "react_base completed without final_answer; "
            f"errors={state.get('errors', [])!r}"
        )
    print("\n========== FINAL ANSWER ==========")
    print(answer)
    print("\n========== RUN SUMMARY ==========")
    print(json.dumps({
        "profile": args.profile,
        "react_steps": len(state.get("react_steps", [])),
        "errors": state.get("errors", []),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
