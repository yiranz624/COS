"""V4 terminal judge, preserved from commit 7868e716cf8380d08cdc944df5d3719bb262e3bc."""

from agent_harness.models.agent_definition import AgentDefinition
from agent_harness.orchestration.roles import _ALL_RESEARCH_TOOLS

EVOLUTION_JUDGE_ROLE = AgentDefinition(
    role_id="evolution_judge",
    display_name="Evolution Judge",
    system_prompt=(
        "Audit the completed orchestration round as the final veto gate before "
        "synthesis. Read the complete original task, the selected candidate, its "
        "ordinary Verifier verdict, inherited facts, prior failure report, current "
        "plan, and the complete structured round trace. You may use research tools "
        "only to test a material weakness or contradiction exposed by that record. "
        "You have veto authority only: you may reject a candidate that an ordinary "
        "Verifier accepted, but you must never approve an unverified candidate, "
        "reverse an ordinary Verifier rejection, replace the selected candidate, "
        "or introduce an alternative answer. Decide approve_synthesis only when the "
        "supplied candidate is already ordinary-Verifier-eligible and the complete "
        "record contains no unresolved identity-binding gap, answer-binding gap, "
        "reliable contradiction, entity-chain break, or material process failure. "
        "Decide evolve otherwise. When evolving, write a concrete failure report "
        "that preserves source-backed facts, disputes conflicting facts, records "
        "failed routes and rejected candidates, and tells the next Planner what to "
        "change without copying the prior plan or Python source. Confirmed facts "
        "must be self-contained, directly relevant, and cite URLs present in the "
        "supplied trace or obtained by your own tool calls. Return JSON only."
    ),
    allowed_tools=_ALL_RESEARCH_TOOLS,
    result_schema={
        "type": "object",
        "properties": {
            "decision": {
                "type": "string",
                "enum": ["approve_synthesis", "evolve"],
            },
            "rationale": {"type": "string"},
            "confirmed_facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string"},
                        "source_urls": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string"},
                        },
                        "candidate_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "claim", "source_urls", "candidate_ids",
                    ],
                    "additionalProperties": False,
                },
            },
            "failure_report": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "properties": {
                            "failure_stage": {"type": "string"},
                            "summary": {"type": "string"},
                            "failed_routes": {
                                "type": "array", "items": {},
                            },
                            "rejected_candidates": {
                                "type": "array", "items": {},
                            },
                            "inheritable_facts": {
                                "type": "array", "items": {},
                            },
                            "disputed_facts": {
                                "type": "array", "items": {},
                            },
                            "unverified_identity_conditions": {
                                "type": "array", "items": {},
                            },
                            "evidence_conflicts": {
                                "type": "array", "items": {},
                            },
                            "verifier_critique": {"type": "string"},
                            "next_round_priorities": {
                                "type": "array", "items": {},
                            },
                            "continue_evolution": {"type": "boolean"},
                        },
                        "required": [
                            "failure_stage", "summary", "failed_routes",
                            "rejected_candidates", "inheritable_facts",
                            "disputed_facts", "unverified_identity_conditions",
                            "evidence_conflicts", "verifier_critique",
                            "next_round_priorities", "continue_evolution",
                        ],
                        "additionalProperties": False,
                    },
                ],
            },
        },
        "required": [
            "decision", "rationale", "confirmed_facts", "failure_report",
        ],
        "additionalProperties": False,
    },
    description=(
        "Vetoes unsafe round proposals or approves already-verified candidates "
        "for synthesis."
    ),
    metadata={"max_turns": 25, "max_completion_tokens": 8192},
)

EVOLUTION_JUDGE_V4_POLICY = """
V4 persistent relaxed review (rounds 3-5 only):
This policy is the limited exception to the strict completeness requirements
above and in the audit instruction. You still cannot replace the selected
candidate, invent an answer, or override the CURRENT ordinary Verifier gate.
Read the complete Runner-supplied prior_round_failures and current trace.
Rounds 3, 4, and 5 ALWAYS use this relaxed standard. No repeated-gap trigger or
three-round same-candidate/answer/verification sequence is required. Earlier
generation/execution failures, empty candidates, candidate changes, and Verifier
rejections do NOT reset later rounds to strict review. They remain evidence to
consider, not reasons to disable relaxed review. No extra review call is needed.

Classify the CURRENT result internally:
a. Reliable answer and reliable evidence chain: approve if ordinarily eligible.
b. Reliable answer with gaps in parts of its proof chain: approve if ordinarily
   eligible and none of the prohibitions below applies; a fully closed chain is
   not required.
c. A concrete answer candidate with incomplete support for the answer itself:
   approve if ordinarily eligible, supported by actual task-relevant evidence,
   and none of the prohibitions below applies. Do not veto solely because answer
   support is incomplete or confidence is below the original strict threshold.
   A nonempty string, repeated answer, or unsupported guess is not evidence.
d. Process failure leaving insufficient results, no concrete requested answer,
   or missing/failed necessary verification: evolve for THIS round. Later rounds
   still use relaxed review.

Mandatory prohibitions: unresolved reliable counterevidence directly negating
an applicable candidate condition; evidence-backed competing interpretations or
candidates that materially change the answer; entity mixing; a missing requested
answer field; missing, mismatched, or compromised critical verification. CURRENT
ordinary Verifier acceptance remains mandatory. A rejected or unverified proposal
cannot be approved, including class c. Check the actual executed verification
against the supplied candidate, not just its reported accepted boolean. If the
question requests a number, a candidate naming only an entity is not an answer.
Do not recover a previous candidate when the current selection is empty.

Reassess prior 'core' labels from evidence rather than treating them as permanent
vetoes. Existing evidence can justify correcting a prior classification; new
evidence is not mandatory. Missing support, source silence, or failed search is
not direct counterevidence. Conversely, repetition never proves a question wrong
or erases reliable counterevidence anywhere in the history. Auxiliary failures
that leave answer evidence and critical verification intact are not independent
vetoes. Do not demand exhaustive searches to prove uniqueness, but do reject
material competing answers supported by the record. Interpret the original
wording without adding stricter requirements or bending it to fit the candidate.

Return the existing JSON schema only. In rationale state the round number,
'V4 relaxed', class a/b/c/d, the actual evidence and remaining uncertainty,
prohibition checks, and the approval/rejection reason. On evolve return the
existing actionable failure_report for the next Planner. Do not add schema fields.
""".strip()


def evolution_judge_v4(round_number: int) -> AgentDefinition:
    """Isolated V4 role: original strict prompt first, persistent relaxation later."""
    prompt = EVOLUTION_JUDGE_ROLE.system_prompt
    if round_number >= 3:
        prompt += "\n\n" + EVOLUTION_JUDGE_V4_POLICY
    return EVOLUTION_JUDGE_ROLE.model_copy(deep=True, update={"system_prompt": prompt})


