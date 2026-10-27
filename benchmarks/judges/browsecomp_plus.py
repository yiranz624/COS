"""BrowseComp-Plus semantic-equivalence judge.

Adapted from ``texttron/BrowseComp-Plus``'s
``scripts_evaluation/evaluate_run.py``:
https://github.com/texttron/BrowseComp-Plus/blob/main/scripts_evaluation/evaluate_run.py

Copyright (c) 2025 texttron. Used under the MIT License.

The upstream evaluator runs Qwen through vLLM in batches. AgentHarness keeps
the upstream grading contract and parser, but sends each request through the
framework's OpenAI-compatible judge client so the scorer works in normal,
subprocess, orchestration, and rejudge flows without a local GPU runtime.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any

from benchmarks.judges._common import (
    Verdict,
    _build_judge_kwargs,
    _judge_call_with_effort_fallback,
)

logger = logging.getLogger(__name__)


BROWSECOMP_PLUS_GRADER_TEMPLATE = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response].

[correct_answer]: Repeat the [correct_answer] given above.

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], in the context of this [question]. You should judge whether the extracted_final_answer is semantically equivalent to [correct_answer], allowing the extracted_final_answer to be string variations of [correct_answer]. You should also allow the extracted_final_answer to be more precise or verbose than [correct_answer], as long as its additional details are correct. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers are semantically equivalent.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available.
""".strip()


def get_semantic_judge_model() -> str:
    """Resolve the semantic judge model without breaking the legacy env name."""
    return (
        os.getenv("SEMANTIC_JUDGE_MODEL")
        or os.getenv("BROWSECOMP_PLUS_JUDGE_MODEL")
        or os.getenv("BROWSECOMP_JUDGE_MODEL")
        or (os.getenv("OPENAI_MODEL") if not os.getenv("JUDGE_BASE_URL") else None)
        or "gpt-4.1-2025-04-14"
    )


def create_judge_prompt(question: str, response: str, correct_answer: str) -> str:
    """Build the BrowseComp-Plus semantic-equivalence prompt."""
    return BROWSECOMP_PLUS_GRADER_TEMPLATE.format(
        question=question,
        response=response,
        correct_answer=correct_answer,
    )


def _line_value(text: str, field: str) -> str | None:
    match = re.search(
        rf"^\s*(?:\*\*)?{re.escape(field)}\s*"
        rf"(?::\*\*|\*\*\s*:|:)\s*(.*?)\s*$",
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    return match.group(1).strip() if match else None


def parse_judge_response(judge_response: str) -> dict[str, Any]:
    """Parse the structured upstream response, including Markdown variants."""
    result: dict[str, Any] = {
        "extracted_final_answer": None,
        "reasoning": None,
        "correct": None,
        "confidence": None,
        "parse_error": False,
    }
    if not judge_response:
        result["parse_error"] = True
        return result

    result["extracted_final_answer"] = _line_value(
        judge_response, "extracted_final_answer",
    )

    reasoning_match = re.search(
        r"^\s*(?:\*\*)?reasoning\s*(?::\*\*|\*\*\s*:|:)\s*(.*?)"
        r"(?=^\s*(?:\*\*)?correct\s*(?::\*\*|\*\*\s*:|:)|\Z)",
        judge_response,
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1).strip()

    correct_value = _line_value(judge_response, "correct")
    if correct_value:
        match = re.match(r"(yes|no)\b", correct_value, re.IGNORECASE)
        if match:
            result["correct"] = match.group(1).lower() == "yes"

    confidence_value = _line_value(judge_response, "confidence")
    if confidence_value:
        match = re.search(r"-?\d+(?:\.\d+)?", confidence_value)
        if match:
            result["confidence"] = min(100.0, max(0.0, float(match.group())))

    if result["correct"] is None:
        result["parse_error"] = True
    return result


def verdict_from_judgement(judgement: dict[str, Any]) -> Verdict:
    """Convert a detailed BrowseComp-Plus judgement to the framework verdict."""
    if judgement.get("parse_error") or judgement.get("correct") is None:
        return "NOT_ATTEMPTED"
    return "CORRECT" if judgement["correct"] else "INCORRECT"


async def judge_semantic_equivalence(
    question: str,
    target: str,
    predicted: str,
    *,
    max_attempts: int = 10,
    retry_delay_seconds: float = 5.0,
) -> dict[str, Any]:
    """Return the complete semantic judgement for one answer pair."""
    prompt = create_judge_prompt(question, predicted, target)
    model = get_semantic_judge_model()
    last: dict[str, Any] = {
        "extracted_final_answer": None,
        "reasoning": None,
        "correct": None,
        "confidence": None,
        "parse_error": True,
        "raw_response": "",
        "model": model,
        "attempts": 0,
    }
    for attempt in range(1, max_attempts + 1):
        kwargs = _build_judge_kwargs(
            prompt,
            max_completion_tokens=4096,
            model_override=model,
        )
        content = await _judge_call_with_effort_fallback(
            kwargs, label="BrowseComp-Plus semantic",
        )
        parsed = parse_judge_response(content)
        last = {
            **parsed,
            "raw_response": content,
            "model": model,
            "attempts": attempt,
        }
        if not parsed["parse_error"]:
            return last
        if attempt < max_attempts:
            logger.info(
                "BrowseComp-Plus judge response was not parseable — retrying %d/%d",
                attempt,
                max_attempts,
            )
            await asyncio.sleep(retry_delay_seconds)
    return last


async def verify_semantic_equivalence(
    question: str, target: str, predicted: str,
) -> Verdict:
    """Framework-compatible binary wrapper around the detailed judge."""
    judgement = await judge_semantic_equivalence(question, target, predicted)
    return verdict_from_judgement(judgement)


__all__ = [
    "BROWSECOMP_PLUS_GRADER_TEMPLATE",
    "create_judge_prompt",
    "get_semantic_judge_model",
    "judge_semantic_equivalence",
    "parse_judge_response",
    "verdict_from_judgement",
    "verify_semantic_equivalence",
]
