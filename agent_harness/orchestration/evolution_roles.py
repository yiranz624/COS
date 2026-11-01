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
