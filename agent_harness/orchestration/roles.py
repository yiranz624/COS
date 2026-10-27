"""Built-in role catalog for generated orchestration scripts."""

from __future__ import annotations

from agent_harness.models.agent_definition import AgentDefinition

_ALL_RESEARCH_TOOLS = [
    "web_search",
    "web_fetch",
    "run_python_code",
]

DECOMPOSER_ROLE = AgentDefinition(
    role_id="decomposer",
    display_name="Decomposer",
    system_prompt=(
        "Break the user's task into a concise list of independent, actionable "
        "subproblems. Preserve every important requirement and avoid redundant "
        "subproblems."
    ),
    allowed_tools=[],
    result_schema={
        "type": "array",
        "items": {"type": "string"},
    },
    description="Splits a task into independent subproblems.",
    metadata={"max_turns": 25},
)

RESEARCHER_ROLE = AgentDefinition(
    role_id="researcher",
    display_name="Researcher",
    system_prompt=(
        "Investigate the requested question using reliable evidence. Use tools "
        "when needed, prefer primary sources, reconcile conflicting evidence, "
        "and return a concise research result with source URLs. Stop searching "
        "and report once the requested claims have adequate evidence."
    ),
    allowed_tools=_ALL_RESEARCH_TOOLS,
    description="Finds and evaluates external evidence.",
    metadata={"max_turns": 25},
)

SOLVER_ROLE = AgentDefinition(
    role_id="solver",
    display_name="Solver",
    system_prompt=(
        "Construct the strongest possible solution to the user's problem. "
        "Follow all constraints and return a self-contained candidate "
        "solution. Use Python only when the task genuinely requires a "
        "calculation; do not use it to reorganize or restate supplied text."
    ),
    allowed_tools=["run_python_code"],
    description="Constructs candidate solutions.",
    metadata={"max_turns": 25},
)

VERIFIER_ROLE = AgentDefinition(
    role_id="verifier",
    display_name="Verifier",
    system_prompt=(
        "Independently verify the candidate solution against the stated "
        "problem. Search or calculate when needed. Accept only if the solution "
        "is correct, complete, and follows every requirement; otherwise give "
        "specific actionable criticism. Stop checking and return the verdict "
        "once every material claim has adequate evidence."
    ),
    allowed_tools=_ALL_RESEARCH_TOOLS,
    result_schema={
        "type": "object",
        "properties": {
            "accepted": {"type": "boolean"},
            "critique": {"type": "string"},
        },
        "required": ["accepted", "critique"],
        "additionalProperties": False,
    },
    description="Checks correctness and returns an acceptance decision.",
    metadata={"max_turns": 25},
)

SYNTHESIZER_ROLE = AgentDefinition(
    role_id="synthesizer",
    display_name="Synthesizer",
    system_prompt=(
        "Combine the supplied intermediate results into one coherent final "
        "answer. Resolve inconsistencies, preserve important qualifications, "
        "and follow the original task's requested format."
    ),
    allowed_tools=[],
    description="Combines intermediate work into a final answer.",
    metadata={"max_turns": 25},
)

GENERAL_ROLE = AgentDefinition(
    role_id="general",
    display_name="General Reasoner",
    system_prompt=(
        "Complete the user's delegated task carefully and return a concise, "
        "self-contained result."
    ),
    allowed_tools=_ALL_RESEARCH_TOOLS,
    description="General-purpose fallback for delegated tasks.",
    metadata={"max_turns": 25},
)

BUILTIN_ORCHESTRATION_ROLES: tuple[AgentDefinition, ...] = (
    DECOMPOSER_ROLE,
    RESEARCHER_ROLE,
    SOLVER_ROLE,
    VERIFIER_ROLE,
    SYNTHESIZER_ROLE,
    GENERAL_ROLE,
)


def register_builtin_orchestration_roles(agent_registry) -> None:
    """Register built-ins without overwriting application-defined roles."""
    for definition in BUILTIN_ORCHESTRATION_ROLES:
        if not agent_registry.has(definition.role_id):
            agent_registry.register(definition.model_copy(deep=True))


__all__ = [
    "BUILTIN_ORCHESTRATION_ROLES",
    "DECOMPOSER_ROLE",
    "GENERAL_ROLE",
    "RESEARCHER_ROLE",
    "SOLVER_ROLE",
    "SYNTHESIZER_ROLE",
    "VERIFIER_ROLE",
    "register_builtin_orchestration_roles",
]
