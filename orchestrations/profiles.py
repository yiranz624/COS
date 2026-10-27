"""Load explicit per-role model routes for orchestration workflows."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_model_profile(profile_id: str) -> dict[str, str]:
    path = Path(__file__).parent / "profiles" / f"{profile_id}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"orchestration model profile not found: {path}")
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    roles = document.get("roles", {})
    if not isinstance(roles, dict) or not roles:
        raise ValueError("orchestration model profile must define roles")
    return {str(role): str(model) for role, model in roles.items()}


def build_role_llms(profile_id: str, config) -> dict[str, object]:
    """Build separate clients so future role changes need no workflow edit."""
    from agent_harness.infra.llm_adapter import create_llm

    return {
        role: create_llm(config.model_copy(update={"openai_model": model}))
        for role, model in load_model_profile(profile_id).items()
    }
